"""Session Recording window for the GripVT desktop app.

Opened from the Sensor Calibration window's "Continue". Shows the live
USB feed (undistorted + ROI-cropped, same pipeline as the pre-baseline
viewer preview) alongside session/trial progress and target-force readouts.
Visual design ported from the Modernist design system (Claude Design:
"GripVT Session Recording").

The right panel is a two-phase gate on Tracker.capture_baseline(): phase 1
(gate box) blocks Start until a real 15-frame burst locks >= MIN_ACCEPTABLE
of the 99 Kalman states; phase 2 (baseline strip) shows the locked count and
unlocks Start. A below-threshold capture can be overridden (flags the saved
record) or recaptured -- but Re-capture is permanently disabled once Start is
pressed, since Kalman states are locked once per session (see CLAUDE.md).

The REC timer, camera-loss detection/retry, and the {n} / 99 TRACKED
indicator are real: the timer is wall-clock elapsed time anchored on the
confirmed Start (E6), camera loss is a debounced cv2.VideoCapture read
failure (E9) that survives on the locked Kalman states without rebuilding
the tracker, and the tracked count reads MarkerRecord.autofilled from a
throttled Tracker.process_frame() once the baseline is locked (E2/E3).

CSV recording, trial sequencing, and target-force values stay static
placeholders. In particular, the TRIAL kicker's counter and its inline Undo
button (E7) are in-window UI state only -- Next Trial / Undo move a local
counter and progress bar, write nothing, and move no real trial boundary;
there is no trial sequence yet for them to advance. A future pass wiring
real trial sequencing must not mistake this counter for it. Start/Stop/Next
Trial otherwise remain stubs beyond the baseline gate, the D2 Re-capture
lockout, and the bookend confirm flow.
"""
import logging
import time
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.calibration import CameraCalibration
from core.detector_search import MIN_ACCEPTABLE, TARGET_MARKERS, SearchResult, region_counts
from core.roi import DEFAULT_ALPHA, ROICrop
from core.tracker import BASELINE_FRAMES, MarkerRecord, Tracker
from output import participants
from output.participants import ParticipantRecord
from output.writer import (
    CSVWriter, EventLog, TrialLog, anchor_wall_clock, make_session_dir, now_ms,
)
from ui import theme
from ui.overlay import draw_overlay

_log = logging.getLogger(__name__)

WINDOW_W = 1100
WINDOW_H = 820
FEED_W = 450
FEED_H = 600

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps, matching viewer_window

# A single dropped read at 30fps is normal jitter, not a disconnection --
# only a sustained streak means the USB device actually went away (E9).
CAMERA_FAIL_POLLS = 15  # ~0.5s at POLL_MS

# Full detect+Kalman+Hungarian per frame at 30fps is more work than the
# indicator needs -- ~10Hz is well past human-perceptible for a count badge.
TRACK_EVERY = 3

# D3 (session CSV recording plan): while a recording is active, process_frame()
# runs every poll instead of throttled at TRACK_EVERY -- the CSV needs every
# frame, unlike the count badge. Bump to 2 (~15Hz) only after an empirical rig
# test shows 30Hz drops frames; see docs/logs/20260728_121200_session_csv_
# recording_plan.md D3. A change away from 1 logs one rate_fallback event at
# Start (static config check, not a dynamic heuristic).
RECORDING_TRACK_EVERY = 1

# D7: force-commit buffered markers.csv rows before they grow past this many
# buffered frames (~60s at 30fps), so an unbroken recording with no Next
# Trial press can't hold an unbounded amount of uncommitted data in RAM.
MAX_BUFFER_FRAMES = 1800

_TOAST_TIMEOUT_MS = 5000
_FADE_STEPS = 8
_FADE_INTERVAL_MS = 50

_LETTERBOX_BG = tuple(int(theme.SURFACE.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

_BASELINE_IDLE: dict = {
    "status": "none", "frame": 0, "count": None,
    "overridden": False, "corners": None, "bands": None, "subline": "",
}

# Phase 7 (E2/E3): {n} / 99 TRACKED badge tiers, keyed by untracked-marker
# count. "border" is None for the two tiers that render borderless -- see
# _paint_tracked, which toggles highlightthickness rather than always
# drawing one.
_TRACKED_TIERS: dict[str, dict[str, str | None]] = {
    "calm": {"fill": theme.INK, "fg": theme.BG, "border": None, "dot": theme.ACCENT},
    "warning": {
        "fill": theme.ACCENT_10, "fg": theme.ACCENT_700,
        "border": theme.ACCENT_600, "dot": theme.ACCENT_700,
    },
    "alarm": {"fill": theme.ACCENT, "fg": theme.BG, "border": None, "dot": theme.BG},
}


class SessionWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Callable[[str | None], None] | None = None,
        detector_params: dict | None = None,
        calibration_result: SearchResult | None = None,
        participant: ParticipantRecord | None = None,
    ) -> None:
        super().__init__(master)
        # E1: on_back now takes an optional post-navigation toast message --
        # Restart's confirmations pass one through to the Calibration window
        # (landing.restore_cal); ordinary Back always passes None.
        self._on_back = on_back or (lambda message=None: print("Back clicked"))
        self._calibration_result = calibration_result
        self._participant = participant

        self.title("GripVT — Session Recording")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._current_img: ImageTk.PhotoImage | None = None
        self._after_id: str | None = None
        self._toast_after: str | None = None
        self._cap: cv2.VideoCapture | None = None

        # E6: REC timer -- _rec_elapsed accumulates wall-clock time across
        # pauses; _rec_segment_start anchors the currently-running segment
        # (None while paused/stopped), so the displayed elapsed time excludes
        # dead gaps once Phase 6 wires _pause_timer/_resume_timer to camera
        # loss/reconnect. Reset on every confirmed Start/Stop, never mid-trial.
        self._rec_after_id: str | None = None
        self._rec_segment_start: float | None = None
        self._rec_elapsed: float = 0.0
        self._timer_paused = False

        # Modal state (Phase 3) -- at most one open at a time; see _open_modal.
        self._modal_scrim: tk.Toplevel | None = None
        self._modal_panel: tk.Toplevel | None = None
        self._modal_configure_id: str | None = None
        # Rig-feedback correction (2026-07-27): pointer offset from the
        # panel's top-left at drag start, re-applied on every <B1-Motion>.
        self._modal_drag_offset: tuple[int, int] = (0, 0)

        # Phase 6 (E9): consecutive failed cap.read()s since the last good
        # frame; _set_camera_disconnected fires once this hits
        # CAMERA_FAIL_POLLS. _disconnect_at anchors the reconnect-gap toast.
        self._read_fail_streak = 0
        self._disconnect_at: float = 0.0

        # Phase 7 (E2/E3): throttled process_frame() bookkeeping. A per-frame
        # exception disables further attempts (_tracking_disabled) rather
        # than risking the poll loop.
        self._frame_tick = 0
        self._tracking_disabled = False
        self._tracked_tier: str | None = None
        self._tracked_blink_after: str | None = None
        self._tracked_blink_state = False

        self._baseline: dict = dict(_BASELINE_IDLE)
        self._baseline_frames: list[np.ndarray] = []
        self._recording_started = False
        # _recording (Phase 5, DCLogic) toggles Start<->Stop within a trial
        # block; _recording_started (D2) latches True forever on first Start
        # and permanently disables Re-capture -- the two must not merge.
        self._recording = False
        # Phase 3 (session CSV recording plan): writer/log handles exist only
        # while a session dir is open -- _finalize_recording nulls all three
        # together, so "is a writer open" always has one source of truth
        # (self._writer is not None), never a second flag to drift out of sync.
        self._writer: CSVWriter | None = None
        self._trial_log: TrialLog | None = None
        self._event_log: EventLog | None = None
        # Phase 5: the still-open camera_loss event dict, or None -- appended
        # to self._event_log in _set_camera_disconnected and closed (end_ts
        # set) in retry_camera or _finalize_recording. EventLog.append() just
        # appends this same dict reference to its internal list, so mutating
        # it in place after append is safe and needs no second lookup.
        self._open_camera_loss_event: dict | None = None
        # D2: monotonic trial-segment ordinal, distinct from self._trial (the
        # operator-facing counter) -- never reused, never decremented by Undo.
        self._segment_id = 0
        # Latest process_frame() records, refreshed on every call (recording
        # or not) -- _finalize_recording needs these for the overlay PNG.
        self._last_records: list[MarkerRecord] | None = None
        # D1: anchored together at confirmed Start via anchor_wall_clock();
        # every buffered timestamp derives from these two via now_ms().
        self._t0_wall_ms: float = 0.0
        self._t0_monotonic: float = 0.0
        # E7: in-window placeholder trial state only -- no real sequencing
        # exists yet. Named attributes (not literals) so Phase 8 only has to
        # assign real values instead of rewiring _refresh_transport.
        self._trial = 12
        self._trial_max = 30
        # Phase 8 (E7): set True by Next Trial, False by Undo -- drives the
        # Undo button's own enabled state, independent of the transport rules.
        self._undo_available = False
        self._strip_recapture_btn: tk.Frame | None = None
        self._capture_frame_label: tk.Label | None = None
        self._capture_fill: tk.Frame | None = None

        self._init_fonts()
        self._build_body()
        theme.apply_dark_titlebar(self)

        self._roi = ROICrop()
        # Undistort with whatever alpha the ROI was selected under, or the
        # crop lands on the wrong pixels -- see viewer_window for the same.
        roi_config = self._roi.config
        self._alpha = roi_config.alpha if roi_config is not None else DEFAULT_ALPHA
        self._cal = CameraCalibration(alpha=self._alpha)
        self._tracker = Tracker(self._cal, self._roi)
        if detector_params is not None:
            self._tracker.params = detector_params

        if self._open_camera():
            self._poll()
        else:
            self._show_placeholder()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _center(self, w: int, h: int) -> None:
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _init_fonts(self) -> None:
        self._f_eyebrow = theme.make_font(self, "semibold", 11)
        self._f_kicker = theme.make_font(self, "semibold", 11)
        self._f_progress_value = theme.make_font(self, "heading", 16)
        self._f_rec = theme.make_font(self, "heading", 12)
        self._f_target_num = theme.make_font(self, "heading", 64)
        self._f_target_unit = theme.make_font(self, "heading", 24)
        self._f_small = theme.make_font(self, "body", 12)
        self._f_button = theme.make_font(self, "heading", 14)
        self._f_toast = theme.make_font(self, "body", 11)

        # baseline gate / strip
        self._f_gate_heading = theme.make_font(self, "heading", 20)
        self._f_gate_body = theme.make_font(self, "body", 13)
        self._f_gate_value = theme.make_font(self, "heading", 13)
        self._f_gate_fail_heading = theme.make_font(self, "heading", 18)
        self._f_gate_count = theme.make_font(self, "heading", 48)
        self._f_gate_count_unit = theme.make_font(self, "heading", 18)
        self._f_gate_footnote = theme.make_font(self, "body", 11)
        self._f_strip_label = theme.make_font(self, "heading", 11)
        self._f_strip_tag = theme.make_font(self, "heading", 10)
        self._f_strip_subline = theme.make_font(self, "body", 12)

        # restart modal (Phase 4)
        self._f_restart_row_heading = theme.make_font(self, "semibold", 14)
        self._f_restart_danger_heading = theme.make_font(self, "heading", 15)

        # bookend confirm modal (Phase 5) -- 19px, distinct from the restart
        # modal's 20px "Restart?" heading.
        self._f_bookend_heading = theme.make_font(self, "heading", 19)

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=32, pady=(22, 20))

        self._build_topbar(body)

        content = tk.Frame(body, bg=theme.BG)
        content.pack(fill="both", expand=True, pady=(18, 0))
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._build_feed(content)
        self._build_panel(content)
        self._build_toast(content)

    def _build_topbar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=theme.BG)
        bar.pack(fill="x")
        bar.columnconfigure(1, weight=1)

        back = self._build_button(bar, "Back", "secondary", "arrow-left", self._handle_back)
        back.grid(row=0, column=0, sticky="w")

        eyebrow = tk.Frame(bar, bg=theme.BG)
        eyebrow.grid(row=0, column=1)
        tk.Frame(eyebrow, bg=theme.ACCENT, width=8, height=8).pack(side="left")
        tk.Label(
            eyebrow, text="GRIPVT", bg=theme.BG, fg=theme.INK_65, font=self._f_eyebrow
        ).pack(side="left", padx=(8, 0))

        # Fills the previously-empty 76px right spacer (E1/E10). No weight on
        # this column, so it keeps its natural width and the eyebrow's
        # weighted centre column (1) stays centred, same as before.
        restart = self._build_button(
            bar, "Restart", "ghost", "rotate", self._open_restart_modal,
            fg_override=theme.INK_60, hover_fg=theme.ACCENT_700,
        )
        restart.grid(row=0, column=2, sticky="e")

    # ── left: camera feed slot ─────────────────────────────────────────────
    def _build_feed(self, parent: tk.Frame) -> None:
        slot = tk.Frame(
            parent, bg=theme.BG, width=FEED_W, height=FEED_H,
            highlightthickness=2, highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        slot.grid(row=0, column=0, sticky="n")
        slot.grid_propagate(False)

        self._feed_label = tk.Label(slot, bg=theme.SURFACE)
        self._feed_label.pack(fill="both", expand=True)

        # Phase 6/7: REC (or REC PAUSED) and the {n}/99 TRACKED indicator
        # share one left-to-right container so they sit beside each other
        # per the design, always at (12, 12) in the feed slot. The container
        # itself is placed unconditionally -- with nothing packed inside it
        # collapses to 0x0, so it draws nothing until a badge is packed.
        self._badge_container = tk.Frame(slot, bg=theme.BG)
        self._badge_container.place(x=12, y=12)

        # Built once; only packed once recording starts -- see _confirm_start.
        badge = tk.Frame(self._badge_container, bg=theme.INK)
        badge_inner = tk.Frame(badge, bg=theme.INK)
        badge_inner.pack(padx=12, pady=6)
        dot = tk.Canvas(badge_inner, width=9, height=9, bg=theme.INK, highlightthickness=0)
        dot.create_oval(0, 0, 9, 9, fill=theme.ACCENT, outline=theme.ACCENT)
        dot.pack(side="left")
        self._rec_label = tk.Label(
            badge_inner, text="REC 00:00", bg=theme.INK, fg=theme.BG, font=self._f_rec,
        )
        self._rec_label.pack(side="left", padx=(6, 0))
        self._rec_badge = badge

        # REC PAUSED variant (Phase 6) -- same fill/text colours as REC, but
        # a square INK_60 dot (not round, not blinking) and no timer text.
        # Swapped in/out for _rec_badge rather than reconfigured in place,
        # per the plan's "build it as a second badge widget" guidance.
        paused = tk.Frame(self._badge_container, bg=theme.INK)
        paused_inner = tk.Frame(paused, bg=theme.INK)
        paused_inner.pack(padx=12, pady=6)
        tk.Frame(paused_inner, bg=theme.INK_60, width=9, height=9).pack(side="left")
        tk.Label(
            paused_inner, text="REC PAUSED", bg=theme.INK, fg=theme.BG, font=self._f_rec,
        ).pack(side="left", padx=(6, 0))
        self._rec_paused_badge = paused

        # {n} / 99 TRACKED indicator (Phase 7) -- built once, reconfigured in
        # place per-tier (_apply_tracked_tier), never destroyed/rebuilt. Only
        # packed the first time a tier is computed (E2: gated on
        # baseline_set), and never unpacked again once shown.
        tracked = tk.Frame(self._badge_container, bg=theme.INK, highlightthickness=0)
        tracked_inner = tk.Frame(tracked, bg=theme.INK)
        tracked_inner.pack(padx=12, pady=6)
        tracked_dot = tk.Canvas(tracked_inner, width=9, height=9, bg=theme.INK, highlightthickness=0)
        self._tracked_dot_id = tracked_dot.create_oval(0, 0, 9, 9, fill=theme.ACCENT, outline=theme.ACCENT)
        tracked_dot.pack(side="left")
        self._tracked_label = tk.Label(
            tracked_inner, text=f"{TARGET_MARKERS} / {TARGET_MARKERS} TRACKED",
            bg=theme.INK, fg=theme.BG, font=self._f_rec,
        )
        self._tracked_label.pack(side="left", padx=(6, 0))
        self._tracked_badge = tracked
        self._tracked_inner = tracked_inner
        self._tracked_dot = tracked_dot

        # Disconnected feed panel (Phase 6, E9) -- replaces the live feed
        # while the camera is lost. Built once, hidden until
        # _set_camera_disconnected shows it; raised above _feed_label but
        # kept below _badge_container so REC PAUSED / TRACKED stay visible.
        self._disconnected_panel = tk.Frame(
            slot, bg=theme.ACCENT_10, highlightthickness=2,
            highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        dc_wrap = tk.Frame(self._disconnected_panel, bg=theme.ACCENT_10)
        dc_wrap.place(relx=0.5, rely=0.5, anchor="center")
        self._warning_icon(dc_wrap, theme.ACCENT_10, size=32).pack(pady=(0, 14))
        tk.Label(
            dc_wrap, text="Camera Disconnected", bg=theme.ACCENT_10, fg=theme.ACCENT_700,
            font=self._f_bookend_heading,  # 19px heading -- same size as this file's other 19px heading
        ).pack(pady=(0, 8))
        tk.Label(
            dc_wrap,
            text="Recording continues — trial data collected so far is not lost.",
            bg=theme.ACCENT_10, fg=theme.ACCENT_700, font=self._f_gate_body,
            wraplength=260, justify="center",
        ).pack(pady=(0, 20))
        self._build_button(
            dc_wrap, "Retry", "primary", "rotate", self.retry_camera,
        ).pack()

    def _show_placeholder(self) -> None:
        self._feed_label.configure(
            image="", text="Camera unavailable", fg=theme.INK_55, bg=theme.SURFACE,
            font=self._f_small,
        )
        self._current_img = None

    # ── right: control panel ───────────────────────────────────────────────
    def _build_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=theme.BG)
        panel.grid(row=0, column=1, sticky="nsew", padx=(28, 0))

        self._progress_block(panel, "SESSION", "3 / 10", 0.30)

        # Paused banner (Phase 6) -- built once, packed only while the camera
        # is disconnected AND a recording is in progress (_set_camera_
        # disconnected / retry_camera). `before=` at pack-time lets it slot
        # in above _baseline_container regardless of build order.
        self._paused_banner = tk.Frame(
            panel, bg=theme.ACCENT_10, highlightthickness=2,
            highlightbackground=theme.ACCENT_600, highlightcolor=theme.ACCENT_600,
        )
        pb_inner = tk.Frame(self._paused_banner, bg=theme.ACCENT_10)
        pb_inner.pack(fill="x", padx=14, pady=12)
        pb_head = tk.Frame(pb_inner, bg=theme.ACCENT_10)
        pb_head.pack(fill="x")
        self._icon(pb_head, "pause", theme.ACCENT_700, theme.ACCENT_10).pack(
            side="left", anchor="n", padx=(0, 10),
        )
        pb_text = tk.Frame(pb_head, bg=theme.ACCENT_10)
        pb_text.pack(side="left", fill="x", expand=True)
        tk.Label(
            pb_text, text="Recording Paused — Camera Signal Lost", bg=theme.ACCENT_10,
            fg=theme.ACCENT_700, font=self._f_gate_value,  # 13px heading
            anchor="w", justify="left", wraplength=280,
        ).pack(fill="x")
        tk.Label(
            pb_text,
            text="Session data captured so far is safe. Retry the feed to "
                 "resume recording.",
            bg=theme.ACCENT_10, fg=theme.ACCENT_700, font=self._f_small,  # 12px body
            anchor="w", justify="left", wraplength=280,
        ).pack(fill="x", pady=(4, 0))

        self._baseline_container = tk.Frame(panel, bg=theme.BG)
        self._baseline_container.pack(fill="both", expand=True)
        self._build_phase1(self._baseline_container)
        self._build_phase2(self._baseline_container)

        row = tk.Frame(panel, bg=theme.BG)
        row.pack(fill="x")
        row.columnconfigure(0, weight=1, uniform="ss")
        row.columnconfigure(1, weight=1, uniform="ss")
        self._start_btn = self._build_button(
            row, "Start", "primary", "play", self._on_start, enabled=False,
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._stop_btn = self._build_button(
            row, "Stop", "secondary", "stop", self._on_stop, enabled=False,
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self._next_btn = self._build_button(
            panel, "Next Trial", "secondary", "arrow-right", self._on_next_trial, enabled=False,
        )
        self._next_btn.pack(fill="x", pady=(12, 0))

        self._render_baseline()

    def _build_toast(self, parent: tk.Frame) -> None:
        # Top-right of the content frame -- content is already inset by
        # body's own (22, 20)/32 padding, so flush-to-corner here reproduces
        # the design's top:22px; right:32px without a second offset.
        self._toast_parent = parent
        self._toast = tk.Frame(
            parent, bg=theme.ACCENT_10, highlightthickness=1,
            highlightbackground=theme.ACCENT, highlightcolor=theme.ACCENT,
        )
        self._toast_inner = tk.Frame(self._toast, bg=theme.ACCENT_10)
        self._toast_inner.pack(padx=10, pady=7)
        self._toast_label = tk.Label(
            self._toast_inner, text="", bg=theme.ACCENT_10, fg=theme.ACCENT_700,
            font=self._f_toast, wraplength=220, justify="left",
        )
        self._toast_label.pack()

    def _progress_block(
        self, parent: tk.Frame, kicker: str, value: str, frac: float, pady_top: int = 0,
    ) -> None:
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(fill="x", pady=(pady_top, 0))

        row = tk.Frame(wrap, bg=theme.BG)
        row.pack(fill="x")
        tk.Label(row, text=kicker, bg=theme.BG, fg=theme.INK_60, font=self._f_kicker).pack(side="left")
        tk.Label(row, text=value, bg=theme.BG, fg=theme.INK, font=self._f_progress_value).pack(side="right")

        track = tk.Frame(wrap, bg=theme.NEUTRAL_200, height=6)
        track.pack(fill="x", pady=(8, 0))
        fill = tk.Frame(track, bg=theme.ACCENT)
        fill.place(relx=0, rely=0, relwidth=frac, relheight=1)

    def _build_trial_block(self, parent: tk.Frame) -> None:
        """TRIAL gets its own rendering path rather than _progress_block
        (Phase 8, E7): it carries a live value/bar plus an inline Undo
        button, so the value label and bar fill are stored for
        _refresh_trial_block to update without a rebuild. SESSION above
        keeps using _progress_block unchanged."""
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(fill="x")

        row = tk.Frame(wrap, bg=theme.BG)
        row.pack(fill="x")
        kicker_row = tk.Frame(row, bg=theme.BG)
        kicker_row.pack(side="left")
        tk.Label(
            kicker_row, text="TRIAL", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker,
        ).pack(side="left")
        # Compact ghost Undo, inline after the kicker (design: margin-left
        # 10px, 10.5px label -- self._f_strip_tag is this file's existing
        # 10px heading font, close enough to reuse rather than add a new one).
        self._undo_btn = self._build_button(
            kicker_row, "Undo", "ghost", "undo", self._on_undo_trial,
            enabled=self._undo_available, font=self._f_strip_tag,
        )
        self._undo_btn.pack(side="left", padx=(10, 0))

        self._trial_value_label = tk.Label(
            row, text=self._trial_value_text(), bg=theme.BG, fg=theme.INK,
            font=self._f_progress_value,
        )
        self._trial_value_label.pack(side="right")

        track = tk.Frame(wrap, bg=theme.NEUTRAL_200, height=6)
        track.pack(fill="x", pady=(8, 0))
        self._trial_fill = tk.Frame(track, bg=theme.ACCENT)
        self._trial_fill.place(relx=0, rely=0, relwidth=self._trial_frac(), relheight=1)

    def _trial_value_text(self) -> str:
        return f"{self._trial} / {self._trial_max}"

    def _trial_frac(self) -> float:
        return self._trial / self._trial_max if self._trial_max else 0.0

    def _refresh_trial_block(self) -> None:
        """Repaints the TRIAL value/bar/Undo in place -- called after every
        Next Trial / Undo (Phase 8), never rebuilds _build_trial_block's
        widgets."""
        self._trial_value_label.configure(text=self._trial_value_text())
        self._trial_fill.place_configure(relwidth=self._trial_frac())
        self._undo_btn.set_enabled(self._undo_available)

    # ── phase 1: baseline gate box ─────────────────────────────────────────
    def _build_phase1(self, parent: tk.Frame) -> None:
        self._phase1_frame = tk.Frame(parent, bg=theme.BG)
        self._phase1_frame.pack(fill="both", expand=True)

        self._gate_box = tk.Frame(
            self._phase1_frame, bg=theme.SURFACE, highlightthickness=2,
            highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        self._gate_box.pack(fill="both", expand=True)

        tk.Frame(self._phase1_frame, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(16, 18))

    # ── phase 2: baseline strip + trial + target force ────────────────────
    def _build_phase2(self, parent: tk.Frame) -> None:
        self._phase2_frame = tk.Frame(parent, bg=theme.BG)
        # Not packed yet -- phase 1 (unlocked) is the initial state.

        self._strip = tk.Frame(self._phase2_frame, highlightthickness=1)
        self._strip.pack(fill="x", pady=(0, 16))

        self._build_trial_block(self._phase2_frame)
        tk.Frame(self._phase2_frame, bg=theme.DIVIDER, height=2).pack(fill="x", pady=18)

        target = tk.Frame(self._phase2_frame, bg=theme.BG)
        target.pack(fill="both", expand=True)
        tk.Label(
            target, text="TARGET FORCE", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker,
        ).pack()
        num_row = tk.Frame(target, bg=theme.BG)
        num_row.pack(pady=(6, 0))
        tk.Label(
            num_row, text="18.4", bg=theme.BG, fg=theme.INK, font=self._f_target_num,
        ).pack(side="left")
        tk.Label(
            num_row, text="kg", bg=theme.BG, fg=theme.INK, font=self._f_target_unit,
        ).pack(side="left", anchor="s", pady=(0, 10))
        tk.Label(
            target, text="40% MVC", bg=theme.BG, fg=theme.INK_55, font=self._f_small,
        ).pack(pady=(4, 0))

        tk.Frame(self._phase2_frame, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(18, 0))

    # ── baseline state machine ─────────────────────────────────────────────
    def _locked(self) -> bool:
        b = self._baseline
        return b["status"] == "pass" or (b["status"] == "fail" and b["overridden"])

    def _render_baseline(self) -> None:
        """Swap the pre-built phase-1 / phase-2 frames via pack_forget()/pack()
        -- both are built once in _build_phase1/_build_phase2, never rebuilt
        here. Only the frames' *contents* (gate box substate, strip variant)
        are torn down and rebuilt, and only from this method -- never from
        the ~30fps _poll() loop (see _update_capturing_progress)."""
        locked = self._locked()
        if locked:
            self._phase1_frame.pack_forget()
            self._phase2_frame.pack(fill="both", expand=True)
            self._render_strip()
        else:
            self._phase2_frame.pack_forget()
            self._phase1_frame.pack(fill="both", expand=True)
            self._render_gate_box()
        self._refresh_transport()

    def _refresh_transport(self) -> None:
        """Recompute Start/Stop/Next Trial enabled state per the design's
        DCLogic (Phase 5). Called from _render_baseline (locked flips) and
        from _confirm_start/_confirm_stop (recording flips) -- every state
        change that feeds these rules. trial/trial_max are E7 placeholders.

        Rig-feedback correction (2026-07-27, E7 revision): Next Trial stays
        usable throughout an active recording -- Start/Stop bracket the
        *entire* session, not one trial, so Next Trial marks trial
        boundaries mid-recording.

        2026-07-28 correction: that recording-independence does not extend
        to *before* Start. self._trial's value at Start becomes the
        persisted trial_index for the first segment (see D9-addendum), so a
        pre-Start Next Trial/Undo click would silently mislabel it -- gate
        on self._recording too."""
        locked = self._locked()
        start_disabled = not locked or self._recording
        stop_disabled = not self._recording
        next_trial_disabled = not locked or not self._recording or self._trial >= self._trial_max
        self._start_btn.set_enabled(not start_disabled)
        self._stop_btn.set_enabled(not stop_disabled)
        self._next_btn.set_enabled(not next_trial_disabled)

    def _render_gate_box(self) -> None:
        for child in self._gate_box.winfo_children():
            child.destroy()
        status = self._baseline["status"]
        if status == "fail":
            self._gate_box.configure(
                bg=theme.ACCENT_10, highlightbackground=theme.ACCENT_600,
                highlightcolor=theme.ACCENT_600,
            )
            self._render_gate_fail()
        else:
            self._gate_box.configure(
                bg=theme.SURFACE, highlightbackground=theme.INK, highlightcolor=theme.INK,
            )
            if status == "capturing":
                self._render_gate_capturing()
            else:
                self._render_gate_none()

    def _render_gate_none(self) -> None:
        wrap = tk.Frame(self._gate_box, bg=theme.SURFACE)
        wrap.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            wrap, text="BASELINE · REQUIRED", bg=theme.SURFACE, fg=theme.INK_70,
            font=self._f_kicker,
        ).pack(pady=(0, 14))
        self._crosshair_icon(wrap, theme.SURFACE).pack(pady=(0, 14))
        tk.Label(
            wrap, text="Capture Baseline", bg=theme.SURFACE, fg=theme.INK,
            font=self._f_gate_heading,
        ).pack(pady=(0, 8))
        tk.Label(
            wrap, text="Locks the zero-reference for all 30 trials. Required before Start.",
            bg=theme.SURFACE, fg=theme.INK_60, font=self._f_gate_body,
            wraplength=230, justify="center",
        ).pack(pady=(0, 20))
        self._build_button(
            wrap, "Capture Baseline", "primary", "dot", self._on_capture_baseline_click,
        ).pack(fill="x")

    def _render_gate_capturing(self) -> None:
        wrap = tk.Frame(self._gate_box, bg=theme.SURFACE)
        wrap.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            wrap, text="BASELINE · CAPTURING", bg=theme.SURFACE, fg=theme.INK_70,
            font=self._f_kicker,
        ).pack(pady=(0, 14))
        tk.Label(
            wrap, text="Hold the Reference Position", bg=theme.SURFACE, fg=theme.INK,
            font=self._f_gate_heading,
        ).pack(pady=(0, 20))

        progress = tk.Frame(wrap, bg=theme.SURFACE)
        progress.pack(fill="x")
        prow = tk.Frame(progress, bg=theme.SURFACE)
        prow.pack(fill="x")
        tk.Label(
            prow, text="FRAMES CAPTURED", bg=theme.SURFACE, fg=theme.INK_70,
            font=self._f_kicker,
        ).pack(side="left")
        self._capture_frame_label = tk.Label(
            prow, text=f"{self._baseline['frame']} / {BASELINE_FRAMES}",
            bg=theme.SURFACE, fg=theme.INK, font=self._f_gate_value,
        )
        self._capture_frame_label.pack(side="right")
        track = tk.Frame(progress, bg=theme.NEUTRAL_200, height=6)
        track.pack(fill="x", pady=(6, 0))
        self._capture_fill = tk.Frame(track, bg=theme.ACCENT)
        self._capture_fill.place(
            relx=0, rely=0, relwidth=self._baseline["frame"] / BASELINE_FRAMES, relheight=1,
        )

        # D1: baseline is the relaxed reference pose, not a grip -- the
        # design's "do not release grip" copy would instruct the wrong pose.
        tk.Label(
            wrap, text="Keep the forearm at rest until the burst completes.",
            bg=theme.SURFACE, fg=theme.INK_60, font=self._f_gate_footnote,
        ).pack(pady=(16, 0))

    def _render_gate_fail(self) -> None:
        b = self._baseline
        wrap = tk.Frame(self._gate_box, bg=theme.ACCENT_10)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        head = tk.Frame(wrap, bg=theme.ACCENT_10)
        head.pack(pady=(0, 14))
        self._warning_icon(head, theme.ACCENT_10).pack(side="left", padx=(0, 8))
        tk.Label(
            head, text="Below Threshold", bg=theme.ACCENT_10, fg=theme.ACCENT_700,
            font=self._f_gate_fail_heading,
        ).pack(side="left")

        count_row = tk.Frame(wrap, bg=theme.ACCENT_10)
        count_row.pack()
        tk.Label(
            count_row, text=str(b["count"]), bg=theme.ACCENT_10, fg=theme.ACCENT_700,
            font=self._f_gate_count,
        ).pack(side="left")
        tk.Label(
            count_row, text=f"/ {TARGET_MARKERS}", bg=theme.ACCENT_10, fg=theme.ACCENT_700,
            font=self._f_gate_count_unit,
        ).pack(side="left", anchor="s", pady=(0, 6))

        subline = b["subline"] or f"{b['count']} / {TARGET_MARKERS} markers detected"
        tk.Label(
            wrap, text=f"{subline} — recapture for a clean zero-reference.",
            bg=theme.ACCENT_10, fg=theme.INK_70, font=self._f_gate_body,
            wraplength=250, justify="center",
        ).pack(pady=(8, 20))

        row = tk.Frame(wrap, bg=theme.ACCENT_10)
        row.pack()
        # Natural widths, not equal columns -- "Continue Anyway" is much wider
        # than "Re-capture", and forcing them to share a uniform grid clips it.
        # D2: recording hasn't started yet in this substate (Start can't be
        # pressed until locked), but guard uniformly for defense in depth.
        enabled = not self._recording_started
        self._build_button(
            row, "Re-capture", "primary", None, self._on_recapture_baseline_click,
            enabled=enabled,
        ).pack(side="left", padx=(0, 6))
        self._build_button(
            row, "Continue Anyway", "ghost", None, self._on_override_baseline_click,
            fg_override=theme.ACCENT_700, bg_override=theme.ACCENT_10,
        ).pack(side="left", padx=(6, 0))

        tk.Label(
            wrap, text="Overriding flags this baseline in the session record.",
            bg=theme.ACCENT_10, fg=theme.ACCENT_700, font=self._f_gate_footnote,
        ).pack(pady=(10, 0))

    def _render_strip(self) -> None:
        for child in self._strip.winfo_children():
            child.destroy()
        b = self._baseline
        overridden_locked = b["status"] == "fail" and b["overridden"]
        fill = theme.ACCENT_10 if overridden_locked else theme.SURFACE
        border = theme.ACCENT_600 if overridden_locked else theme.DIVIDER
        icon_bg = theme.ACCENT_700 if overridden_locked else theme.INK
        label = "BASELINE OVERRIDDEN" if overridden_locked else "BASELINE LOCKED"

        self._strip.configure(bg=fill, highlightbackground=border, highlightcolor=border)
        row = tk.Frame(self._strip, bg=fill)
        row.pack(fill="x", padx=14, pady=10)

        # D2: Re-capture is disabled the moment Start is pressed, permanently.
        # Packed before `left` so it claims its natural width first -- `left`
        # fills+expands into the remainder, so a long subline never squeezes
        # the button down to an unreadable sliver.
        enabled = not self._recording_started
        self._strip_recapture_btn = self._build_button(
            row, "Re-capture", "ghost", None, self._on_recapture_baseline_click,
            enabled=enabled, bg_override=fill,
        )
        self._strip_recapture_btn.pack(side="right")

        left = tk.Frame(row, bg=fill)
        left.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._check_icon_square(left, icon_bg).pack(side="left", padx=(0, 10))

        text_col = tk.Frame(left, bg=fill)
        text_col.pack(side="left", fill="x", expand=True)
        label_row = tk.Frame(text_col, bg=fill)
        label_row.pack(fill="x", anchor="w")
        tk.Label(
            label_row, text=label, bg=fill, fg=theme.INK, font=self._f_strip_label,
        ).pack(side="left")
        if overridden_locked:
            tag = tk.Frame(label_row, bg=theme.ACCENT_10)
            tag.pack(side="left", padx=(6, 0))
            tk.Label(
                tag, text="FLAGGED", bg=theme.ACCENT_10, fg=theme.ACCENT_700,
                font=self._f_strip_tag,
            ).pack(padx=6, pady=2)

        count_text = f"{b['count']} / {TARGET_MARKERS} markers"
        if b["subline"]:
            count_text += f" · {b['subline']}"
        sub = tk.Label(
            text_col, text=count_text, bg=fill, fg=theme.INK_60, font=self._f_strip_subline,
            anchor="w", justify="left",
        )
        sub.pack(fill="x", pady=(2, 0))
        # The column's width isn't known until Tk lays the strip out, and a
        # guessed wraplength wider than it clips instead of wrapping -- so
        # track the real width rather than hardcoding one.
        text_col.bind(
            "<Configure>", lambda e, w=sub: w.configure(wraplength=max(e.width, 1)),
        )

    def _compute_subline(self, corners: dict[str, int]) -> str:
        result = self._calibration_result
        assert result is not None
        ref = result.reference_corners
        weak = [cell for cell in sorted(ref) if corners.get(cell, 0) < ref[cell]]
        if not weak:
            return "corner coverage full"
        return f"{len(weak)} of {len(ref)} corners weak"

    # ── baseline actions ────────────────────────────────────────────────────
    def _on_capture_baseline_click(self) -> None:
        if self._baseline["status"] == "capturing":
            return
        self._baseline_frames = []
        self._baseline = {
            "status": "capturing", "frame": 0, "count": None,
            "overridden": False, "corners": None, "bands": None, "subline": "",
        }
        self._render_baseline()

    def _on_recapture_baseline_click(self) -> None:
        if self._recording_started:
            return
        self._baseline_frames = []
        self._baseline = dict(_BASELINE_IDLE)
        self._render_baseline()

    def _on_override_baseline_click(self) -> None:
        if self._baseline["status"] != "fail":
            return
        self._baseline["overridden"] = True
        self._save_baseline_record()
        self._render_baseline()

    def _capture_baseline_frame(self, raw_frame: np.ndarray) -> None:
        self._baseline_frames.append(raw_frame)
        self._baseline["frame"] = len(self._baseline_frames)
        self._update_capturing_progress()
        if len(self._baseline_frames) >= BASELINE_FRAMES:
            self._finish_baseline_capture()

    def _update_capturing_progress(self) -> None:
        frame = self._baseline["frame"]
        if self._capture_frame_label is not None:
            self._capture_frame_label.configure(text=f"{frame} / {BASELINE_FRAMES}")
        if self._capture_fill is not None:
            self._capture_fill.place_configure(relwidth=frame / BASELINE_FRAMES)

    def _finish_baseline_capture(self) -> None:
        count = self._tracker.capture_baseline(self._baseline_frames)
        self._baseline_frames = []
        passed = count >= MIN_ACCEPTABLE

        corners = bands = None
        subline = ""
        if self._calibration_result is not None:
            # D5: capture_baseline() returns only a count -- rebuild
            # (x, y, area) detections from the locked Kalman states to score
            # corner coverage against the calibration-relaxed geometry.
            # Kalman states are full-frame (tracker._detect_full_frame maps
            # them back); the geometry is ROI-relative, so undo the offset or
            # every marker lands in the wrong cell.
            detections = [
                (*self._roi.to_crop(*s.baseline_pos), s.baseline_area)
                for s in self._tracker.kalman.states.values()
            ]
            counts = region_counts(detections, self._calibration_result.geometry)
            corners, bands = counts["corners"], counts["bands"]
            subline = self._compute_subline(corners)

        self._baseline = {
            "status": "pass" if passed else "fail",
            "frame": BASELINE_FRAMES,
            "count": count,
            "overridden": False,
            "corners": corners,
            "bands": bands,
            "subline": subline,
        }
        if passed:
            self._save_baseline_record()
        self._render_baseline()

    def _save_baseline_record(self) -> None:
        if self._participant is None:
            return
        b = self._baseline
        participants.save_baseline(
            self._participant,
            locked_count=b["count"],
            passed=b["status"] == "pass",
            overridden=b["overridden"],
            corners=b["corners"],
            bands=b["bands"],
            params=self._tracker.params,
        )

    # ── decorative icons (gate box / strip -- not button icons) ───────────
    def _crosshair_icon(self, parent: tk.Misc, bg: str) -> tk.Canvas:
        c = tk.Canvas(parent, width=30, height=30, bg=bg, highlightthickness=0)
        s = 30 / 24

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        c.create_oval(*p(3, 3, 21, 21), outline=theme.INK, width=1.6)
        c.create_line(*p(12, 2, 12, 6), fill=theme.INK, width=1.6, capstyle="round")
        c.create_line(*p(12, 18, 12, 22), fill=theme.INK, width=1.6, capstyle="round")
        c.create_line(*p(2, 12, 6, 12), fill=theme.INK, width=1.6, capstyle="round")
        c.create_line(*p(18, 12, 22, 12), fill=theme.INK, width=1.6, capstyle="round")
        return c

    def _warning_icon(self, parent: tk.Misc, bg: str, size: int = 24) -> tk.Canvas:
        # size=24 (default) is the shipped baseline-fail-box glyph; size=32 is
        # for the camera-disconnect panel (Phase 6) -- coordinates scale by
        # `s`, so the 24px call sites render pixel-identically to before.
        c = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0)
        s = size / 24

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        c.create_polygon(
            *p(12, 3, 22, 20, 2, 20), fill="", outline=theme.ACCENT_700, width=1.8,
            joinstyle="round",
        )
        c.create_line(*p(12, 9, 12, 13), fill=theme.ACCENT_700, width=1.8, capstyle="round")
        c.create_oval(
            *p(11.2, 15.2, 12.8, 16.8), fill=theme.ACCENT_700, outline=theme.ACCENT_700,
        )
        return c

    def _check_icon_square(self, parent: tk.Misc, bg: str) -> tk.Frame:
        sq = tk.Frame(parent, bg=bg, width=20, height=20)
        sq.pack_propagate(False)
        c = tk.Canvas(sq, width=12, height=12, bg=bg, highlightthickness=0)
        c.place(relx=0.5, rely=0.5, anchor="center")
        c.create_line(0, 6, 5, 11, fill=theme.BG, width=3, capstyle="round", joinstyle="round")
        c.create_line(5, 11, 12, 1, fill=theme.BG, width=3, capstyle="round", joinstyle="round")
        return sq

    # ── button/icon idiom (copied from new_participant_window, not imported,
    #    so the windows stay independent) ──────────────────────────────────
    def _build_button(
        self, parent: tk.Misc, label: str, kind: str, icon: str | None,
        command: Callable[[], None], enabled: bool = True,
        fg_override: str | None = None, bg_override: str | None = None,
        hover_fg: str | None = None, font: tkfont.Font | None = None,
    ) -> tk.Frame:
        fill, fg, border = self._button_style(kind, enabled, fg_override, bg_override)
        button_font = font or self._f_button

        btn = tk.Frame(
            parent, bg=fill, highlightthickness=1,
            highlightbackground=border, highlightcolor=border,
        )
        inner = tk.Frame(btn, bg=fill)
        # .btn-ghost carries padding-inline: var(--space-1) in the design
        # system, not the 14px the filled variants use.
        inner.pack(padx=4 if kind == "ghost" else 14, pady=8)

        widgets: list[tk.Widget] = [btn, inner]
        icon_widget: tk.Canvas | None = None
        # rotate (Restart/Retry) and undo (Undo) render on the left, same as
        # the original four -- Phase 6/8 are the first callers to pass them.
        if icon in ("arrow-left", "play", "stop", "dot", "rotate", "undo"):
            icon_widget = self._icon(inner, icon, fg, fill)
            icon_widget.pack(side="left", padx=(0, 6))
            widgets.append(icon_widget)
        text = tk.Label(inner, text=label, bg=fill, fg=fg, font=button_font)
        text.pack(side="left")
        widgets.append(text)
        if icon == "arrow-right":
            icon_widget = self._icon(inner, icon, fg, fill)
            icon_widget.pack(side="left", padx=(6, 0))
            widgets.append(icon_widget)

        wired = False

        def repaint(en: bool) -> tuple[str, str]:
            fill_, fg_, border_ = self._button_style(kind, en, fg_override, bg_override)
            for w in widgets:
                w.configure(bg=fill_)
            text.configure(fg=fg_)
            btn.configure(highlightbackground=border_, highlightcolor=border_)
            if icon_widget is not None:
                icon_widget.delete("all")
                self._paint_icon_shape(icon_widget, icon, fg_)
            return fill_, fg_

        def set_enabled(en: bool) -> None:
            nonlocal wired
            fill_, fg_ = repaint(en)
            if en and not wired:
                hover, active = self._button_hover_active(kind)
                self._wire_button(
                    btn, widgets, fill_, hover, active, command,
                    text=text, icon_widget=icon_widget, icon=icon,
                    base_fg=fg_, hover_fg=hover_fg,
                )
                wired = True
            elif not en and wired:
                self._unwire_button(btn, widgets)
                wired = False
            btn.enabled = en

        btn.set_enabled = set_enabled
        set_enabled(enabled)
        return btn

    def _button_style(
        self, kind: str, enabled: bool,
        fg_override: str | None = None, bg_override: str | None = None,
    ) -> tuple[str, str, str]:
        if not enabled:
            fill = bg_override if bg_override is not None else theme.BG
            return fill, theme.INK_45, theme.INK_45
        if kind == "primary":
            return theme.ACCENT, theme.BG, theme.ACCENT
        if kind == "ghost":
            fill = bg_override if bg_override is not None else theme.BG
            return fill, fg_override or theme.ACCENT, fill
        if kind == "danger":
            # The design's "Restart Entire Testing" button -- a raw solid
            # accent-700 fill, not one of the three .btn variants (E10a).
            return theme.ACCENT_700, theme.BG, theme.ACCENT_700
        fill = bg_override if bg_override is not None else theme.BG
        return fill, theme.INK, theme.DIVIDER

    def _button_hover_active(self, kind: str) -> tuple[str, str]:
        if kind == "primary":
            return theme.ACCENT_600, theme.ACCENT_700
        if kind == "ghost":
            return theme.ACCENT_10, theme.ACCENT_18
        if kind == "danger":
            return theme.ACCENT_600, theme.ACCENT
        return theme.HOVER, theme.ACTIVE

    def _icon(self, parent: tk.Misc, kind: str, color: str, bg: str) -> tk.Canvas:
        c = tk.Canvas(parent, width=14, height=14, bg=bg, highlightthickness=0)
        self._paint_icon_shape(c, kind, color)
        return c

    def _paint_icon_shape(self, canvas: tk.Canvas, kind: str | None, color: str) -> None:
        s = 14 / 24

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        if kind == "arrow-right":
            canvas.create_line(*p(5, 12, 19, 12), fill=color, width=2, capstyle="round")
            canvas.create_line(*p(12, 5, 19, 12, 12, 19), fill=color, width=2,
                                capstyle="round", joinstyle="round")
        elif kind == "arrow-left":
            canvas.create_line(*p(19, 12, 5, 12), fill=color, width=2, capstyle="round")
            canvas.create_line(*p(12, 19, 5, 12, 12, 5), fill=color, width=2,
                                capstyle="round", joinstyle="round")
        elif kind == "play":
            canvas.create_polygon(*p(6, 3, 20, 12, 6, 21), fill=color, outline=color)
        elif kind == "stop":
            canvas.create_rectangle(*p(5, 5, 19, 19), fill=color, outline=color)
        elif kind == "dot":
            canvas.create_oval(*p(4, 4, 20, 20), fill=color, outline=color)
        elif kind == "rotate":
            # Restart's refresh icon -- design SVG:
            # <path d="M3 12a9 9 0 1 0 2.6-6.4L2 9"/><path d="M2 4v5h5"/>
            # A ~315deg circle sweep (open at the upper-left) plus the two
            # line segments that draw the arrowhead sitting in that gap.
            canvas.create_arc(
                *p(3, 3, 21, 21), start=180, extent=315, style="arc",
                outline=color, width=2,
            )
            canvas.create_line(*p(2, 4, 2, 9), fill=color, width=2, capstyle="round")
            canvas.create_line(*p(2, 9, 7, 9), fill=color, width=2, capstyle="round")
        elif kind == "pause":
            # <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
            canvas.create_rectangle(*p(6, 4, 10, 20), fill=color, outline=color)
            canvas.create_rectangle(*p(14, 4, 18, 20), fill=color, outline=color)
        elif kind == "undo":
            # design SVG: <path d="M9 14 4 9l5-5"/>
            # <path d="M4 9h10.5a5.5 5.5 0 1 1 0 11H11"/>
            # Arrowhead chevron pointing left, then a line that runs right,
            # loops through a semicircle, and returns left along the bottom.
            canvas.create_line(*p(9, 14, 4, 9), fill=color, width=2,
                                capstyle="round", joinstyle="round")
            canvas.create_line(*p(4, 9, 9, 4), fill=color, width=2,
                                capstyle="round", joinstyle="round")
            canvas.create_line(*p(4, 9, 14.5, 9), fill=color, width=2, capstyle="round")
            canvas.create_arc(
                *p(9, 9, 20, 20), start=90, extent=-180, style="arc",
                outline=color, width=2,
            )
            canvas.create_line(*p(14.5, 20, 11, 20), fill=color, width=2, capstyle="round")

    def _wire_button(
        self, btn: tk.Frame, widgets: list[tk.Widget], base: str,
        hover: str, active: str, command: Callable[[], None],
        text: tk.Label | None = None, icon_widget: tk.Canvas | None = None,
        icon: str | None = None, base_fg: str | None = None, hover_fg: str | None = None,
    ) -> None:
        def paint(colour: str) -> None:
            for w in widgets:
                w.configure(bg=colour)

        def paint_fg(colour: str | None) -> None:
            # Only the Restart ghost button passes hover_fg today (E10b) --
            # every other caller leaves this a no-op, so their label/icon
            # colour stays exactly what repaint() already set.
            if colour is None:
                return
            if text is not None:
                text.configure(fg=colour)
            if icon_widget is not None:
                icon_widget.delete("all")
                self._paint_icon_shape(icon_widget, icon, colour)

        def on_enter(_e: object) -> None:
            paint(hover)
            paint_fg(hover_fg)

        def on_leave(_e: object) -> None:
            paint(base)
            paint_fg(base_fg)

        def on_press(_e: object) -> None:
            paint(active)

        def on_release(_e: object) -> None:
            x, y = btn.winfo_pointerxy()
            hovered = btn.winfo_containing(x, y)
            if hovered is not None and (hovered is btn or str(hovered).startswith(str(btn))):
                paint(hover)
                paint_fg(hover_fg)
                command()
            else:
                paint(base)
                paint_fg(base_fg)

        for w in widgets:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<ButtonPress-1>", on_press)
            w.bind("<ButtonRelease-1>", on_release)
            w.configure(cursor="hand2")

    def _unwire_button(self, btn: tk.Frame, widgets: list[tk.Widget]) -> None:
        for w in widgets:
            w.unbind("<Enter>")
            w.unbind("<Leave>")
            w.unbind("<ButtonPress-1>")
            w.unbind("<ButtonRelease-1>")
            w.configure(cursor="arrow")

    # ── stub handlers ───────────────────────────────────────────────────────
    def _on_start(self) -> None:
        # Mirrors _refresh_transport's startDisabled rule -- guards against a
        # stray call (e.g. a key binding) reaching a disabled button.
        start_disabled = not self._locked() or self._recording
        if start_disabled:
            return
        self._open_start_bookend_modal()

    def _on_stop(self) -> None:
        stop_disabled = not self._recording
        if stop_disabled:
            return
        self._open_stop_bookend_modal()

    def _on_next_trial(self) -> None:
        # Mirrors _refresh_transport's nextTrialDisabled rule (Phase 8, E7,
        # recording-independent per the rig-feedback correction) -- guards a
        # stray call the same way _on_start/_on_stop already do.
        next_trial_disabled = not self._locked() or self._trial >= self._trial_max
        if next_trial_disabled:
            return
        if self._trial_log is not None:
            # D9: persist the trial boundary only while a recording is open
            # -- guard on _trial_log, not self._recording, so the two can
            # never disagree. Commits the segment that's ending, then opens
            # the next one under a fresh, never-reused segment_id (D2).
            assert self._writer is not None
            self._writer.write_window()
            self._trial_log.close_trial(
                self._segment_id, now_ms(self._t0_wall_ms, self._t0_monotonic),
            )
            self._segment_id += 1
        self._trial = min(self._trial_max, self._trial + 1)
        if self._trial_log is not None:
            self._trial_log.open_trial(
                trial_id=self._segment_id, trial_index=self._trial,
                start_ts=now_ms(self._t0_wall_ms, self._t0_monotonic),
            )
        self._undo_available = True
        self._refresh_trial_block()
        self._refresh_transport()

    def _on_undo_trial(self) -> None:
        if not self._undo_available:
            return
        if self._trial_log is not None:
            # D2 tombstone: void the segment Next Trial just closed and
            # committed (self._segment_id - 1) -- its markers.csv rows stay
            # on disk untouched, only trials.csv's voided flag flips.
            assert self._writer is not None
            self._trial_log.void_trial(self._segment_id - 1)
            # The just-opened segment (self._segment_id) is still in
            # progress -- discard whatever it has buffered so far (could be
            # zero or a few frames) and re-open it fresh for the redo.
            # open_trial() only stages a row in memory (trials.csv is only
            # written by flush()), so overwriting that entry here leaves no
            # "completed" row behind for a segment that never happened.
            # self._segment_id itself is never decremented (D2) -- only
            # self._trial, the displayed counter, moves back.
            self._writer.discard_window()
            redo_trial_index = max(1, self._trial - 1)
            self._trial_log.open_trial(
                trial_id=self._segment_id, trial_index=redo_trial_index,
                start_ts=now_ms(self._t0_wall_ms, self._t0_monotonic),
            )
        self._trial = max(1, self._trial - 1)
        self._undo_available = False
        self._refresh_trial_block()
        self._refresh_transport()

    # ── bookend confirm modals (Phase 5) ────────────────────────────────────
    def _open_start_bookend_modal(self) -> None:
        def build(parent: tk.Frame) -> None:
            self._build_bookend_content(
                parent,
                title="Confirm Start Bookend",
                body="Perform the triple-tap bookend on the dynamometer now, "
                     "before recording begins.",
                on_confirm=self._confirm_start,
            )

        self._open_modal(build)

    def _open_stop_bookend_modal(self) -> None:
        def build(parent: tk.Frame) -> None:
            self._build_bookend_content(
                parent,
                title="Confirm End Bookend",
                body="Perform the closing triple-tap bookend on the dynamometer "
                     "now, before ending the session.",
                on_confirm=self._confirm_stop,
            )

        self._open_modal(build)

    def _build_bookend_content(
        self, parent: tk.Frame, title: str, body: str, on_confirm: Callable[[], None],
    ) -> None:
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(padx=28, pady=24)
        # 420px panel -- forced via a zero-height spacer, the same idiom the
        # topbar uses for its fixed-width Restart column.
        tk.Frame(wrap, bg=theme.BG, width=364, height=1).pack()

        tk.Label(
            wrap, text=title, bg=theme.BG, fg=theme.INK, font=self._f_bookend_heading,
            anchor="w", justify="left",
        ).pack(fill="x", pady=(4, 12))
        tk.Label(
            wrap, text=body, bg=theme.BG, fg=theme.INK_70, font=self._f_gate_body,
            wraplength=364, justify="left", anchor="w",
        ).pack(fill="x")
        tk.Label(
            wrap,
            text="A missed bookend makes this session's force labels unrecoverable.",
            bg=theme.BG, fg=theme.INK_70, font=self._f_small, wraplength=364,
            justify="left", anchor="w",
        ).pack(fill="x", pady=(8, 20))

        btn_row = tk.Frame(wrap, bg=theme.BG)
        btn_row.pack(fill="x")
        self._build_button(btn_row, "Confirm", "primary", None, on_confirm).pack(side="right")

    def _confirm_start(self) -> None:
        self._close_modal()
        self._recording = True
        self._recording_started = True
        self._show_rec_badge()
        if self._strip_recapture_btn is not None:
            self._strip_recapture_btn.set_enabled(False)
        self._start_timer()
        self._refresh_transport()
        print("Start clicked")

        # D5/D8: degrade to a no-op, mirroring _save_baseline_record's guard
        # (self._participant is None is the headless/test-fixture path) --
        # never crash, never write to the project root.
        if self._participant is None:
            return
        session_dir = make_session_dir(self._participant.folder)
        self._writer = CSVWriter(session_dir)
        self._trial_log = TrialLog(session_dir)
        self._event_log = EventLog(
            session_dir,
            participant_id=self._participant.participant_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
            frame_index_at_start=self._tracker.frame_index,
        )
        self._t0_wall_ms, self._t0_monotonic = anchor_wall_clock()
        self._segment_id = 0
        self._trial_log.open_trial(
            trial_id=self._segment_id, trial_index=self._trial,
            start_ts=now_ms(self._t0_wall_ms, self._t0_monotonic),
        )
        if RECORDING_TRACK_EVERY != 1:
            # D3 approved addition: a static config check, not a dynamic
            # heuristic -- fires once, here, if RECORDING_TRACK_EVERY is ever
            # manually bumped after a rig test shows 30Hz drops frames.
            self._event_log.append({
                "type": "rate_fallback",
                "target_hz": 30 / RECORDING_TRACK_EVERY,
                "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
            })

    def _confirm_stop(self) -> None:
        self._close_modal()
        self._recording = False
        self._hide_rec_badges()
        # Defensive: a Stop while disconnected is not a designed flow, but
        # leaving the paused banner up with no recording in progress would
        # be a stale, misleading state.
        self._paused_banner.pack_forget()
        self._stop_timer()
        self._refresh_transport()
        print("Stop clicked")
        self._finalize_recording("stopped")

    def _buffer_recording_frame(self, records: list[MarkerRecord]) -> None:
        """D7: appends one frame to the CSVWriter's in-RAM buffer and force-
        commits early if it has grown past MAX_BUFFER_FRAMES, so a long
        unbroken recording can't hold unbounded uncommitted data."""
        assert self._writer is not None and self._event_log is not None
        ts = now_ms(self._t0_wall_ms, self._t0_monotonic)
        self._writer.buffer_frame(records, self._tracker.frame_index, ts, trial_id=self._segment_id)
        if self._writer.pending_count >= MAX_BUFFER_FRAMES:
            self._writer.write_window()
            self._event_log.append({
                "type": "buffer_flushed",
                "frame_count": MAX_BUFFER_FRAMES,
                "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
            })

    def _finalize_recording(self, reason: str, discard_open_segment: bool = False) -> None:
        """General-purpose recording finalizer (D8) -- Phase 3's only caller
        is _confirm_stop, always with the default (commit, not discard).
        Later phases (camera-loss hooks, exit paths, Restart) call this too,
        some with discard_open_segment=True; `reason` is accepted now so
        those callers don't need a second finalize signature -- Phase 3 does
        not itself log it anywhere.
        """
        if self._writer is None:
            return
        assert self._trial_log is not None and self._event_log is not None
        if self._open_camera_loss_event is not None:
            self._open_camera_loss_event["end_ts"] = now_ms(self._t0_wall_ms, self._t0_monotonic)
            self._open_camera_loss_event = None
        self._trial_log.close_trial(self._segment_id, now_ms(self._t0_wall_ms, self._t0_monotonic))
        if discard_open_segment:
            self._writer.discard_window()
            # Rig-found 2026-07-28: without this, a Restart-discarded segment
            # still showed up in trials.csv as a real, non-voided, completed
            # trial (valid start_ts/end_ts) despite having zero markers.csv
            # rows -- indistinguishable from genuine (empty) data.
            self._trial_log.void_trial(self._segment_id)
        else:
            self._writer.write_window()

        if self._last_records is not None and self._tracker.last_undistorted is not None:
            # Best-effort visualization -- must never block the data-integrity
            # flushes below it. A failure here previously left trials.csv and
            # events.json unwritten even though markers.csv had already
            # committed (found via rig verification 2026-07-28: closing the
            # window mid-recording produced markers.csv alone).
            try:
                overlay = draw_overlay(
                    self._tracker.last_undistorted, self._last_records, True,
                    self._tracker.frame_index,
                )
                cv2.imwrite(self._writer.png_path, overlay)
            except Exception:
                _log.exception("overlay.png write failed during _finalize_recording")

        self._trial_log.flush()
        self._event_log.flush()
        self._writer.close()
        self._writer = None
        self._trial_log = None
        self._event_log = None

    # ── REC / REC PAUSED / TRACKED badge helpers (Phase 6/7) ────────────────
    def _show_rec_badge(self) -> None:
        """Packs the live REC timer badge, swapping out REC PAUSED if it was
        showing. `before=` keeps REC ahead of the TRACKED badge in the
        container's left-to-right order even if TRACKED was packed first
        (it is packed once, on first tier computation, and never unpacked)."""
        self._rec_paused_badge.pack_forget()
        kwargs = {"side": "left"}
        if self._tracked_badge.winfo_manager() == "pack":
            kwargs["before"] = self._tracked_badge
        self._rec_badge.pack(**kwargs)

    def _show_rec_paused_badge(self) -> None:
        self._rec_badge.pack_forget()
        kwargs = {"side": "left"}
        if self._tracked_badge.winfo_manager() == "pack":
            kwargs["before"] = self._tracked_badge
        self._rec_paused_badge.pack(**kwargs)

    def _hide_rec_badges(self) -> None:
        self._rec_badge.pack_forget()
        self._rec_paused_badge.pack_forget()

    # ── REC timer (Phase 5, E6) ──────────────────────────────────────────────
    def _start_timer(self) -> None:
        """Begin the 1s after-tick from the Start confirm. Fresh state every
        Start -- this is a wall-clock display, not tied to CSV frame
        timestamps (see the module docstring)."""
        self._rec_elapsed = 0.0
        self._rec_segment_start = time.monotonic()
        self._timer_paused = False
        self._tick_timer()

    def _tick_timer(self) -> None:
        elapsed = self._rec_elapsed
        if self._rec_segment_start is not None:
            elapsed += time.monotonic() - self._rec_segment_start
        minutes, seconds = divmod(int(elapsed), 60)
        self._rec_label.configure(text=f"REC {minutes:02d}:{seconds:02d}")
        self._rec_after_id = self.after(1000, self._tick_timer)

    def _pause_timer(self) -> None:
        """Freezes the displayed elapsed time. Phase 6 calls this the moment
        camera loss is confirmed, so the dead gap is never counted."""
        if self._timer_paused or self._rec_segment_start is None:
            return
        self._rec_elapsed += time.monotonic() - self._rec_segment_start
        self._rec_segment_start = None
        self._timer_paused = True

    def _resume_timer(self) -> None:
        """Re-anchors the running segment. Phase 6 calls this on reconnect."""
        if not self._timer_paused:
            return
        self._rec_segment_start = time.monotonic()
        self._timer_paused = False

    def _stop_timer(self) -> None:
        if self._rec_after_id is not None:
            self.after_cancel(self._rec_after_id)
            self._rec_after_id = None
        self._rec_segment_start = None
        self._timer_paused = False
        self._rec_elapsed = 0.0
        self._rec_label.configure(text="REC 00:00")

    # ── restart modal (Phase 4, E1/E8) ──────────────────────────────────────
    def _open_restart_modal(self) -> None:
        """Two asymmetric options -- Restart Session costs one session,
        Restart Entire Testing costs the whole visit. Both close this window
        via self._on_back(message); the toast lands on Calibration (E1)."""
        state: dict = {"reason": None, "chips": {}, "note_entry": None, "note_state": None}

        def repaint_chip(key: str) -> None:
            chip, label = state["chips"][key]
            selected = state["reason"] == key
            fill = theme.ACCENT_08 if selected else theme.BG
            fg = theme.ACCENT_700 if selected else theme.INK
            border = theme.ACCENT_600 if selected else theme.DIVIDER
            chip.configure(bg=fill, highlightbackground=border, highlightcolor=border)
            label.configure(bg=fill, fg=fg)

        def on_chip_click(key: str) -> None:
            # Single-select toggle -- clicking the selected chip deselects it.
            state["reason"] = None if state["reason"] == key else key
            for k in state["chips"]:
                repaint_chip(k)

        def note_text() -> str:
            entry = state["note_entry"]
            note_state = state["note_state"]
            if entry is None or note_state is None or note_state["placeholder"]:
                return ""
            return entry.get()

        def confirm_session() -> None:
            # D6: Restart aborts the in-progress segment -- log the reason/
            # note as a session_aborted event, then discard the uncommitted
            # buffer (already-committed trials stay on disk untouched). No
            # event log exists if Restart was pressed before any Start, in
            # which case there is nothing to record.
            if self._event_log is not None:
                self._event_log.append({
                    "type": "session_aborted",
                    "scope": "session",
                    "reason": state["reason"],
                    "note": note_text(),
                    "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
                })
                self._finalize_recording("restart", discard_open_segment=True)
            self._close_modal()
            self.release_camera()
            self._on_back(
                "Session 3 abandoned — returned to Sensor Calibration. Sessions 1-2 kept."
            )

        def confirm_all() -> None:
            # D6: same abort handling as confirm_session, scope="all".
            if self._event_log is not None:
                self._event_log.append({
                    "type": "session_aborted",
                    "scope": "all",
                    "reason": state["reason"],
                    "note": note_text(),
                    "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
                })
                self._finalize_recording("restart", discard_open_segment=True)
            self._close_modal()
            self.release_camera()
            self._on_back(
                "All sessions abandoned — returned to Sensor Calibration. "
                "Counter reset to 1 of 10."
            )

        def build(parent: tk.Frame) -> None:
            wrap = tk.Frame(parent, bg=theme.BG)
            wrap.pack(padx=28, pady=28)
            # 560px panel, same fixed-width spacer idiom as the bookend modal.
            tk.Frame(wrap, bg=theme.BG, width=504, height=1).pack()

            tk.Label(
                wrap, text="Restart?", bg=theme.BG, fg=theme.INK, font=self._f_gate_heading,
                anchor="w", justify="left",
            ).pack(fill="x")
            tk.Label(
                wrap,
                text="Choose what to abandon. Neither option deletes data — this "
                     "can't be undone once confirmed.",
                bg=theme.BG, fg=theme.INK_70, font=self._f_gate_body, wraplength=504,
                justify="left", anchor="w",
            ).pack(fill="x", pady=(6, 20))

            # Restart Session -- costs one session, kept.
            row1 = tk.Frame(
                wrap, bg=theme.BG, highlightthickness=1,
                highlightbackground=theme.DIVIDER, highlightcolor=theme.DIVIDER,
            )
            row1.pack(fill="x", pady=(0, 12))
            row1_in = tk.Frame(row1, bg=theme.BG)
            row1_in.pack(fill="x", padx=16, pady=16)
            row1_text = tk.Frame(row1_in, bg=theme.BG)
            row1_text.pack(side="left", fill="x", expand=True, padx=(0, 12))
            tk.Label(
                row1_text, text="Restart Session", bg=theme.BG, fg=theme.INK,
                font=self._f_restart_row_heading, anchor="w",
            ).pack(fill="x")
            tk.Label(
                row1_text,
                text="Abandons session 3 and returns to Sensor Calibration. "
                     "Sessions 1-2 are kept. The next attempt is still session "
                     "3 of 10.",
                bg=theme.BG, fg=theme.INK_70, font=self._f_small, wraplength=320,
                justify="left", anchor="w",
            ).pack(fill="x", pady=(4, 0))
            self._build_button(
                row1_in, "Restart Session", "secondary", None, confirm_session,
            ).pack(side="right")

            # Restart Entire Testing -- costs the whole visit, flagged accent
            # treatment so the asymmetry with the row above reads at a glance.
            row2 = tk.Frame(
                wrap, bg=theme.ACCENT_10, highlightthickness=2,
                highlightbackground=theme.ACCENT_600, highlightcolor=theme.ACCENT_600,
            )
            row2.pack(fill="x")
            row2_in = tk.Frame(row2, bg=theme.ACCENT_10)
            row2_in.pack(fill="x", padx=18, pady=18)
            row2_text = tk.Frame(row2_in, bg=theme.ACCENT_10)
            row2_text.pack(side="left", fill="x", expand=True, padx=(0, 12))
            tk.Label(
                row2_text, text="Restart Entire Testing", bg=theme.ACCENT_10,
                fg=theme.ACCENT_700, font=self._f_restart_danger_heading, anchor="w",
            ).pack(fill="x")
            tk.Label(
                row2_text,
                text="Abandons all sessions for this participant's visit and "
                     "returns to Sensor Calibration. Nothing is deleted; every "
                     "session is kept and flagged. The counter resets to "
                     "session 1 of 10.",
                bg=theme.ACCENT_10, fg=theme.ACCENT_700, font=self._f_small,
                wraplength=320, justify="left", anchor="w",
            ).pack(fill="x", pady=(4, 0))
            self._build_button(
                row2_in, "Restart Entire Testing", "danger", None, confirm_all,
            ).pack(side="right")

            tk.Frame(wrap, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(20, 16))

            tk.Label(
                wrap, text="REASON", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker,
                anchor="w",
            ).pack(fill="x", pady=(0, 8))
            chips_row = tk.Frame(wrap, bg=theme.BG)
            chips_row.pack(fill="x", pady=(0, 16))
            for key in ("Sensor slipped", "Camera fault", "Participant request", "Other"):
                chip = tk.Frame(
                    chips_row, bg=theme.BG, highlightthickness=1,
                    highlightbackground=theme.DIVIDER, highlightcolor=theme.DIVIDER,
                )
                chip.pack(side="left", padx=(0, 8))
                label = tk.Label(chip, text=key, bg=theme.BG, fg=theme.INK, font=self._f_small)
                label.pack(padx=12, pady=7)
                state["chips"][key] = (chip, label)
                for w in (chip, label):
                    w.bind("<Button-1>", lambda _e, k=key: on_chip_click(k))
                    w.configure(cursor="hand2")

            note_entry = tk.Entry(
                wrap, bg=theme.BG, fg=theme.INK, font=self._f_small, relief="flat",
                highlightthickness=1, highlightbackground=theme.DIVIDER,
                highlightcolor=theme.DIVIDER,
            )
            note_entry.pack(fill="x", ipady=6, pady=(0, 20))
            state["note_entry"] = note_entry
            state["note_state"] = _add_placeholder(note_entry, "Add a note (optional)")

            cancel_row = tk.Frame(wrap, bg=theme.BG)
            cancel_row.pack(fill="x")
            self._build_button(
                cancel_row, "Cancel", "ghost", None, self._close_modal,
            ).pack(side="right")

        self._open_modal(build)

    # ── toast (copied from calibration_window, not imported, so the windows
    #    stay independent) ────────────────────────────────────────────────────
    def _show_toast(self, message: str) -> None:
        if self._toast_after is not None:
            self.after_cancel(self._toast_after)
        self._toast_label.configure(text=message)
        self._reset_toast_colors()
        self._toast.place(in_=self._toast_parent, relx=1.0, x=0, y=0, anchor="ne")
        self._toast.lift()
        self._toast_after = self.after(_TOAST_TIMEOUT_MS, lambda: self._fade_toast(1))

    def _fade_toast(self, step: int) -> None:
        t = step / _FADE_STEPS
        bg = _lerp(theme.ACCENT_10, theme.BG, t)
        edge = _lerp(theme.ACCENT, theme.BG, t)
        fg = _lerp(theme.ACCENT_700, theme.BG, t)
        self._toast.configure(bg=bg, highlightbackground=edge, highlightcolor=edge)
        self._toast_inner.configure(bg=bg)
        self._toast_label.configure(bg=bg, fg=fg)
        if step < _FADE_STEPS:
            self._toast_after = self.after(_FADE_INTERVAL_MS, lambda: self._fade_toast(step + 1))
        else:
            self._toast_after = None
            self._toast.place_forget()
            self._reset_toast_colors()

    def _reset_toast_colors(self) -> None:
        self._toast.configure(
            bg=theme.ACCENT_10, highlightbackground=theme.ACCENT, highlightcolor=theme.ACCENT
        )
        self._toast_inner.configure(bg=theme.ACCENT_10)
        self._toast_label.configure(bg=theme.ACCENT_10, fg=theme.ACCENT_700)

    # ── modal (Phase 3, E4/E5) ──────────────────────────────────────────────
    def _open_modal(self, build_content: Callable[[tk.Frame], None]) -> None:
        """Blocking modal as two borderless Toplevels -- tkinter frames have
        no alpha, so the design's rgba(32,30,29,0.55) scrim can't be a single
        widget (E4). Only one modal may be open at a time; opening a second
        closes the first."""
        self._close_modal()

        scrim = tk.Toplevel(self)
        scrim.overrideredirect(True)
        scrim.transient(self)
        scrim.attributes("-alpha", 0.55)
        scrim.configure(bg=theme.INK)

        panel = tk.Toplevel(self)
        panel.overrideredirect(True)
        panel.transient(self)
        panel.configure(
            bg=theme.BG, highlightthickness=2,
            highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        inner = tk.Frame(panel, bg=theme.BG)
        inner.pack()
        build_content(inner)

        self._modal_scrim = scrim
        self._modal_panel = panel
        self._reposition_modal()
        # Rig-feedback correction (2026-07-27): on Windows a freshly-created
        # overrideredirect Toplevel's initial geometry doesn't reliably
        # "stick" before the next event-loop tick -- on the rig this landed
        # the modal pinned at the desktop's top-left instead of centered.
        # Re-applying once more once the loop goes idle fixes it. Safe even
        # if the modal has since closed: _reposition_modal no-ops when
        # _modal_scrim/_modal_panel are None.
        self.after_idle(self._reposition_modal)
        scrim.lift()
        panel.lift()

        # The window is resizable(False, False) but can still be dragged, so
        # both Toplevels have to follow it to stay aligned over the session.
        self._modal_configure_id = self.bind(
            "<Configure>", lambda _e: self._reposition_modal(),
        )

        # Manual drag-to-move (rig-feedback correction, 2026-07-27): an
        # overrideredirect Toplevel has no OS titlebar to grab, so dragging
        # the panel is the permanent fallback regardless of whether the idle
        # re-position above also fixes the underlying race. Bound directly
        # to the panel Toplevel itself, not its content children, so
        # grabbing empty panel background moves it while clicks on the
        # buttons/entries/chips built by build_content keep their own
        # bindings untouched. Only the panel moves -- the scrim is left
        # alone here and keeps tracking the session window via the
        # <Configure> binding above, so it still covers the whole window.
        # A fresh bind per _open_modal call (panel is a new Toplevel every
        # time); destroying the Toplevel in _close_modal removes these
        # bindings with it, so there is nothing to unbind explicitly.
        panel.bind("<ButtonPress-1>", self._on_modal_drag_start)
        panel.bind("<B1-Motion>", self._on_modal_drag_motion)

        panel.grab_set()

    def _reposition_modal(self) -> None:
        if self._modal_scrim is None or self._modal_panel is None:
            return
        self.update_idletasks()
        x, y = self.winfo_rootx(), self.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        self._modal_scrim.geometry(f"{w}x{h}+{x}+{y}")

        panel = self._modal_panel
        panel.update_idletasks()
        pw, ph = panel.winfo_reqwidth(), panel.winfo_reqheight()
        px, py = x + (w - pw) // 2, y + (h - ph) // 2
        panel.geometry(f"{pw}x{ph}+{px}+{py}")

    def _on_modal_drag_start(self, event: tk.Event) -> None:
        """<ButtonPress-1> on the panel Toplevel -- event.x/y are already
        panel-relative because the binding is on the panel itself, so this
        is directly the pointer's offset from the panel's top-left."""
        self._modal_drag_offset = (event.x, event.y)

    def _on_modal_drag_motion(self, event: tk.Event) -> None:
        """<B1-Motion> on the panel Toplevel. Moves only the panel -- the
        scrim is meant to keep covering the whole session window and stays
        aligned to it via the <Configure> binding in _open_modal, not via
        drag. Guarded the same way _reposition_modal is, in case a queued
        motion event still lands after the modal has closed mid-drag."""
        if self._modal_panel is None:
            return
        offset_x, offset_y = self._modal_drag_offset
        x = event.x_root - offset_x
        y = event.y_root - offset_y
        self._modal_panel.geometry(f"+{x}+{y}")

    def _close_modal(self) -> None:
        """Safe to call when no modal is open. Called unconditionally from
        both _on_close and _handle_back (E5) so a modal held open at window
        teardown can never leave a dangling grab."""
        if self._modal_configure_id is not None:
            self.unbind("<Configure>", self._modal_configure_id)
            self._modal_configure_id = None
        if self._modal_panel is not None:
            self._modal_panel.grab_release()
            self._modal_panel.destroy()
            self._modal_panel = None
        if self._modal_scrim is not None:
            self._modal_scrim.destroy()
            self._modal_scrim = None

    # ── camera lifecycle ────────────────────────────────────────────────────
    def _open_camera(self) -> bool:
        """Factored out of __init__ (Phase 6, mirrors
        calibration_window._open_camera) so retry_camera() can reopen the
        capture identically after a disconnect. Returns whether the camera
        opened; callers own the resulting UI state -- this method touches
        only self._cap."""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            cap.release()
            return False
        cal_w, cal_h = self._cal.image_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self._cap = cap
        return True

    def _poll(self) -> None:
        # Mirrors calibration_window._poll's early-return shape -- a stopped
        # capture must never reschedule itself (Phase 6).
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            self._read_fail_streak = 0
            if self._baseline["status"] == "capturing":
                self._capture_baseline_frame(frame)
            display = self._tracker.undistort(frame)
            display = self._roi.crop(display)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = _fit_letterbox(Image.fromarray(rgb), FEED_W, FEED_H)
            photo = ImageTk.PhotoImage(img)
            self._feed_label.configure(image=photo)
            self._current_img = photo  # plain tk.Label drops the image without this ref
            self._update_tracked_indicator(frame)
        else:
            # E9: a lone dropped read is normal 30fps jitter -- only a
            # sustained streak means the camera actually disconnected.
            self._read_fail_streak += 1
            if self._read_fail_streak >= CAMERA_FAIL_POLLS:
                self._set_camera_disconnected()
                return  # disconnected state owns rescheduling from here (none, until retry)
        self._after_id = self.after(POLL_MS, self._poll)

    def _set_camera_disconnected(self) -> None:
        """E9: fires only after CAMERA_FAIL_POLLS consecutive failed reads.
        Never touches self._tracker -- the baseline and all locked Kalman
        states must survive a disconnect/reconnect cycle untouched
        (CLAUDE.md; decision 5 of the design brief)."""
        self._after_id = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._disconnect_at = time.monotonic()
        self._read_fail_streak = 0

        self._feed_label.lower()
        self._disconnected_panel.place(x=0, y=0, relwidth=1, relheight=1)
        self._disconnected_panel.lift()
        self._badge_container.lift()  # stays above the disconnected panel

        if self._recording:
            self._pause_timer()
            self._show_rec_paused_badge()
            self._paused_banner.pack(fill="x", pady=(0, 16), before=self._baseline_container)
            if self._event_log is not None:
                self._open_camera_loss_event = {
                    "type": "camera_loss",
                    "start_ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
                }
                self._event_log.append(self._open_camera_loss_event)

    def retry_camera(self) -> None:
        """Public (Phase 6, mirrors CalibrationWindow.retry_camera): no-op
        while a capture is already open; the disconnected panel's Retry
        button is the only caller today."""
        if self._cap is not None:
            return
        if not self._open_camera():
            self._show_toast("Camera still unavailable — check the USB connection.")
            return

        self._disconnected_panel.place_forget()
        self._paused_banner.pack_forget()
        if self._recording:
            self._show_rec_badge()
            self._resume_timer()
            if self._open_camera_loss_event is not None:
                self._open_camera_loss_event["end_ts"] = now_ms(self._t0_wall_ms, self._t0_monotonic)
                self._open_camera_loss_event = None
        self._read_fail_streak = 0
        gap = round(time.monotonic() - self._disconnect_at)
        self._show_toast(f"Camera reconnected — {gap}s gap")
        self._poll()

    # ── tracked-marker indicator (Phase 7, E2/E3) ───────────────────────────
    def _update_tracked_indicator(self, raw_frame: np.ndarray) -> None:
        """Throttled Tracker.process_frame() driving the {n}/99 TRACKED
        badge. E2: gated on baseline_set -- before Capture Baseline there
        are no Kalman states and process_frame() raises RuntimeError. Any
        per-frame exception disables further attempts rather than risking
        the live feed's poll loop.

        D3 (session CSV recording plan): while self._recording, the
        process_frame() call itself runs every poll (RECORDING_TRACK_EVERY)
        instead of throttled at TRACK_EVERY, so no frame is lost to the CSV
        -- but the badge repaint stays throttled at TRACK_EVERY regardless,
        since a count label doesn't need to redraw at 30Hz.
        """
        if not self._tracker.baseline_set or self._tracking_disabled:
            return
        self._frame_tick += 1
        track_every = RECORDING_TRACK_EVERY if self._recording else TRACK_EVERY
        should_process = self._frame_tick % track_every == 0
        should_repaint = self._frame_tick % TRACK_EVERY == 0
        if not should_process and not should_repaint:
            return
        try:
            records = self._tracker.process_frame(raw_frame)
        except Exception as e:
            _log.exception("process_frame failed -- disabling tracked-marker indicator")
            self._tracking_disabled = True
            if self._recording:
                # D8: a dead tracker must not leave a REC badge counting over
                # a writer that will never receive another frame -- log the
                # failure and finalize now, same as any other exit path.
                if self._event_log is not None:
                    self._event_log.append({
                        "type": "tracking_failed",
                        "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
                        "error": str(e),
                    })
                self._show_toast("Tracking failed — recording stopped.")
                self._finalize_recording("tracking_failed")
                # Phase 5 follow-up: mirror _confirm_stop's UI teardown so the
                # REC badge/timer don't keep counting over a finalized writer.
                # Done after finalize -- neither touches self._recording, and
                # _finalize_recording's no-op guard checks self._writer, not
                # self._recording, so ordering here can't affect it.
                self._recording = False
                self._hide_rec_badges()
                self._paused_banner.pack_forget()
                self._stop_timer()
                self._refresh_transport()
            return
        self._last_records = records
        if self._recording and self._writer is not None:
            self._buffer_recording_frame(records)
        if should_repaint:
            untracked = sum(r.autofilled for r in records)
            self._set_tracked_badge(untracked)

    def _set_tracked_badge(self, untracked: int) -> None:
        tracked = TARGET_MARKERS - untracked
        self._tracked_label.configure(text=f"{tracked} / {TARGET_MARKERS} TRACKED")

        if untracked == 0:
            tier = "calm"
        elif untracked <= 5:
            tier = "warning"
        else:
            tier = "alarm"
        if tier != self._tracked_tier:
            self._tracked_tier = tier
            self._apply_tracked_tier(tier)

        if self._tracked_badge.winfo_manager() != "pack":
            self._tracked_badge.pack(side="left", padx=(8, 0))

    def _apply_tracked_tier(self, tier: str) -> None:
        """Reconfigures the existing badge widgets in place -- never
        destroys/rebuilds them, since this runs at ~10Hz (TRACK_EVERY)."""
        self._cancel_tracked_blink()
        spec = _TRACKED_TIERS[tier]
        self._paint_tracked(spec["fill"], spec["fg"], spec["border"], spec["dot"])
        if tier == "alarm":
            # E3: blink only the alarm tier -- REC's dot stays static, and
            # the brief permits nothing more than this one deliberate pulse.
            self._tracked_blink_state = False
            self._tick_tracked_blink()

    def _paint_tracked(
        self, fill: str, fg: str, border: str | None, dot: str,
    ) -> None:
        self._tracked_badge.configure(bg=fill)
        self._tracked_inner.configure(bg=fill)
        self._tracked_label.configure(bg=fill, fg=fg)
        self._tracked_dot.configure(bg=fill)
        self._tracked_dot.itemconfigure(self._tracked_dot_id, fill=dot, outline=dot)
        if border is not None:
            self._tracked_badge.configure(
                highlightthickness=1, highlightbackground=border, highlightcolor=border,
            )
        else:
            self._tracked_badge.configure(highlightthickness=0)

    def _tick_tracked_blink(self) -> None:
        # ~600ms half-period -> ~1.2s full cycle, matching the design.
        # Toggles only the fill (ACCENT <-> ACCENT_600); fg/dot already
        # contrast against both, so they're left alone.
        self._tracked_blink_state = not self._tracked_blink_state
        fill = theme.ACCENT if self._tracked_blink_state else theme.ACCENT_600
        self._tracked_badge.configure(bg=fill)
        self._tracked_inner.configure(bg=fill)
        self._tracked_label.configure(bg=fill)
        self._tracked_dot.configure(bg=fill)
        self._tracked_blink_after = self.after(600, self._tick_tracked_blink)

    def _cancel_tracked_blink(self) -> None:
        if self._tracked_blink_after is not None:
            self.after_cancel(self._tracked_blink_after)
            self._tracked_blink_after = None

    def release_camera(self) -> None:
        for attr in ("_after_id", "_toast_after", "_rec_after_id", "_tracked_blink_after"):
            pending = getattr(self, attr)
            if pending is not None:
                self.after_cancel(pending)
                setattr(self, attr, None)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _handle_back(self) -> None:
        self._close_modal()  # E5: never leave a grab held by a dead widget
        self._finalize_recording("closed")  # D8: every exit path finalizes
        self.release_camera()
        self._on_back(None)  # ordinary Back carries no toast message (E1)

    def _on_close(self) -> None:
        """Default WM_DELETE_WINDOW handler -- SessionWindow's own fallback
        if constructed without an overriding caller (e.g. headless tests).
        landing.py's real app flow registers its own protocol handler after
        construction (2026-07-28 rig finding: that override skipped
        _finalize_recording entirely, since Tk keeps only one handler per
        protocol name and the later registration wins) -- see landing.py's
        open_session() for the real X-button path."""
        self._close_modal()  # E5: same guard as _handle_back
        self._finalize_recording("closed")  # D8: every exit path finalizes
        self.release_camera()
        self.destroy()


def _add_placeholder(entry: tk.Entry, text: str) -> dict:
    """Grey placeholder that clears on focus and restores while empty.

    Copied from new_participant_window.py, not imported -- see the
    _build_button section's comment on this file's independence convention.
    Returns the mutable state dict so callers can tell whether the field is
    still showing placeholder text (i.e. effectively empty) rather than input.
    """
    state = {"placeholder": True}

    def show() -> None:
        entry.delete(0, "end")
        entry.insert(0, text)
        entry.configure(fg=theme.INK_55)
        state["placeholder"] = True

    def on_focus_in(_e: object) -> None:
        if state["placeholder"]:
            entry.delete(0, "end")
            entry.configure(fg=theme.INK)
            state["placeholder"] = False

    def on_focus_out(_e: object) -> None:
        if not entry.get():
            show()

    show()
    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)
    return state


def _lerp(from_hex: str, to_hex: str, t: float) -> str:
    a = tuple(int(from_hex[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(to_hex[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _fit_letterbox(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (box_w, box_h), _LETTERBOX_BG)
    canvas.paste(resized, ((box_w - new_w) // 2, (box_h - new_h) // 2))
    return canvas
