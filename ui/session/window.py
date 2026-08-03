"""Session Recording window for the GripVT desktop app.

This is the ui/session/ package: SessionWindow used to be a single 2272-line
tk.Toplevel subclass (ui/session_window.py); it is now split by
responsibility into six mixins, composed here, per
docs/logs/20260729_141951_codebase_reorg_and_session_split_plan.md (D8):

  - window.py (this file) -- the single __init__, scaffold/widget-tree
    building, button/icon builders, lifecycle methods.
  - progress.py (ProgressMixin) -- SESSION/TRIAL progress blocks, TARGET
    FORCE display.
  - baseline.py (BaselineMixin) -- the Capture Baseline gate state machine.
  - recording.py (RecordingMixin) -- trial start/stop/undo, bookend modals,
    CSV finalize, REC timer.
  - modals.py (ModalMixin) -- generic modal/toast plumbing, Restart modal.
  - camera_feed.py (CameraFeedMixin) -- camera polling, tracked-marker
    indicator.

The split is mechanical (verbatim method moves, no logic changes): all six
mixins share one `self` namespace on the composed SessionWindow instance,
so cross-group calls/attribute access are left as plain, unqualified
`self.foo` (D1/D2) -- only __init__ (D3) stays single, here.

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

CSV recording, trial sequencing, and target-force values are real. A
confirmed Start generates and locks a 30-trial randomized %MVC sequence
(core/trial_sequence.py), persists it to sequence.json, and drives the
TRIAL kicker and TARGET FORCE block off sequence[trial_index - 1] --
self._trial is the sole index into that sequence, so Next Trial / Undo
(E7) can never desync from it (see D2 in the trial-sequencing plan). Both
also open/close/void real segments in trials.csv via TrialLog, alongside
markers.csv (per-frame) and events.json (bookend/annotation) writers.
"""
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable

import cv2
import numpy as np
from PIL import ImageTk

from core.calibration import CameraCalibration
from core.detector_search import TARGET_MARKERS, SearchResult
from core.roi import DEFAULT_ALPHA, ROICrop
from core.tracker import MarkerRecord, Tracker
from core.trial_sequence import TRIAL_COUNT
from output.participants import ParticipantRecord
from output.video import VideoLogWriter
from ui import theme

from .baseline import BaselineMixin, _BASELINE_IDLE
from .camera_feed import FEED_H, FEED_W, CameraFeedMixin
from .modals import ModalMixin
from .progress import ProgressMixin
from .recording import RecordingMixin

WINDOW_W = 1100
WINDOW_H = 820


class SessionWindow(
    tk.Toplevel, ProgressMixin, BaselineMixin, RecordingMixin, ModalMixin, CameraFeedMixin,
):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Callable[[str | None], None] | None = None,
        detector_params: dict | None = None,
        calibration_result: SearchResult | None = None,
        participant: ParticipantRecord | None = None,
        session_num: int = 0,
        pending_calibration: dict | None = None,
        pending_supersede_session: str | None = None,
    ) -> None:
        super().__init__(master)
        # E1: on_back now takes an optional post-navigation toast message --
        # Restart's confirmations pass one through to the Calibration window
        # (landing.restore_cal); ordinary Back always passes None.
        self._on_back = on_back or (lambda message=None: print("Back clicked"))
        self._calibration_result = calibration_result
        self._participant = participant
        # F6a (D3): the calibration bundle built in CalibrationWindow._continue
        # -- held in memory, never written, until _confirm_start (recording.py)
        # flushes it into the session directory. None in the headless/test
        # fixture path, same as self._participant.
        self._pending_calibration = pending_calibration
        # D10 (resume-supersede plan): bare folder name of the trailing
        # incomplete session to supersede on the first confirmed Start,
        # consumed exactly once in _confirm_start (recording.py) and set to
        # None there.
        self._pending_supersede_session = pending_supersede_session
        # D2 (session-indicator plan): _session_num is mutable and live;
        # _session_num_at_open is a frozen snapshot Restart Session rolls
        # back to -- see _confirm_start's single increment site.
        self._session_num = session_num
        self._session_num_at_open = session_num

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
        # Post-session unmount/rest modal (docs/logs/20260803_post_session_
        # unmount_rest_modal_plan.md, D10): owned solely by the modal's own
        # Continue handler in modals.py -- no other close path starts this
        # timer, so no other close path needs to cancel it.
        self._rest_after_id: str | None = None

        # Phase 6 (E9): consecutive failed cap.read()s since the last good
        # frame; _set_camera_disconnected fires once this hits
        # CAMERA_FAIL_POLLS. _disconnect_at anchors the reconnect-gap toast.
        self._read_fail_streak = 0
        self._disconnect_at: float = 0.0

        # Phase 7 (E2/E3): throttled process_frame() bookkeeping. A per-frame
        # exception disables further attempts (_tracking_disabled) rather
        # than risking the poll loop.
        self._frame_tick = 0
        # Phase 4 (framerate spike plan, D6/D7): counts every poll (not just
        # tracked ones, unlike _frame_tick) so the display-repaint throttle
        # in camera_feed.py's _poll() has its own reliable counter.
        self._display_tick = 0
        self._tracking_disabled = False
        self._tracked_tier: str | None = None
        self._tracked_blink_after: str | None = None
        self._tracked_blink_state = False

        self._baseline: dict = dict(_BASELINE_IDLE)
        self._baseline_frames: list[np.ndarray] = []
        # Holds the winning burst until _save_baseline_record persists it (D13).
        self._baseline_clip_frames: list[np.ndarray] = []
        # F6b (D3): the save_baseline() kwargs bundle built by
        # _save_baseline_record -- held in memory, never written, until
        # _confirm_start (recording.py) flushes it into the session
        # directory. A later _save_baseline_record call (pass, or override,
        # possibly after a recapture) simply overwrites this dict, so only
        # the winning attempt is ever flushed (same guarantee the .mp4 path
        # already gave). None until a baseline capture locks.
        self._pending_baseline: dict | None = None
        self._recording_started = False
        # _recording (Phase 5, DCLogic) toggles Start<->Stop within a trial
        # block; _recording_started (D2) latches True forever on first Start
        # and permanently disables Re-capture -- the two must not merge.
        self._recording = False
        # Phase 3 (session CSV recording plan): writer/log handles exist only
        # while a session dir is open -- _finalize_recording nulls all three
        # together, so "is a writer open" always has one source of truth
        # (self._writer is not None), never a second flag to drift out of sync.
        self._writer = None
        self._trial_log = None
        self._event_log = None
        # Phase 4 (raw video log plan): shares self._writer's lifetime --
        # opened in _confirm_start, closed and nulled in _finalize_recording
        # (D9), no second "is it open" flag.
        self._video: VideoLogWriter | None = None
        # Phase 5: the still-open camera_loss event dict, or None -- appended
        # to self._event_log in _set_camera_disconnected and closed (end_ts
        # set) in retry_camera or _finalize_recording. EventLog.append() just
        # appends this same dict reference to its internal list, so mutating
        # it in place after append is safe and needs no second lookup.
        self._open_camera_loss_event: dict | None = None
        # Phase 3 (trial sequencing plan): the locked 30-trial sequence,
        # assigned once at confirmed Start (D6) -- None until then.
        self._sequence = None
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
        self._trial = 0
        self._trial_max = TRIAL_COUNT
        # Phase 8 (E7): set True by Next Trial, False by Undo -- drives the
        # Undo button's own enabled state, independent of the transport rules.
        self._undo_available = False
        # F2: True once this segment's grip boundary has been marked. Reset at
        # confirmed Start and on every Next Trial / Undo, since each opens a
        # fresh segment with its own unmarked boundary.
        self._grip_marked = False
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
        # D7 (session-indicator plan): gated on self._session_num >= 1 --
        # nothing to restart before the first confirmed Start.
        self._restart_btn = self._build_button(
            bar, "Restart", "ghost", "rotate", self._open_restart_modal,
            fg_override=theme.INK_60, hover_fg=theme.ACCENT_700,
            enabled=self._session_num >= 1,
        )
        self._restart_btn.grid(row=0, column=2, sticky="e")

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

        self._build_session_block(panel)

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

        # F2: marks where rest ends and grip begins inside the current
        # segment. "dot" rather than a transport icon -- it marks an instant,
        # it does not move the session forward.
        self._grip_btn = self._build_button(
            panel, "Grip Start", "secondary", "dot", self._on_grip_start, enabled=False,
        )
        self._grip_btn.pack(fill="x", pady=(8, 0))

        self._render_baseline()

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
        self._target_num_label = tk.Label(
            num_row, text=self._target_num_text(), bg=theme.BG, fg=self._target_color(),
            font=self._f_target_num,
        )
        self._target_num_label.pack(side="left")
        self._target_unit_label = tk.Label(
            num_row, text="kg", bg=theme.BG, fg=theme.INK, font=self._f_target_unit,
        )
        self._target_unit_label.pack(side="left", anchor="s", pady=(0, 10))
        self._target_sub_label = tk.Label(
            target, text=self._target_sub_text(), bg=theme.BG, fg=self._target_color(),
            font=self._f_small,
        )
        self._target_sub_label.pack(pady=(4, 0))
        # Advisory accept range (F1) -- display only, never gates or scores.
        self._target_band_label = tk.Label(
            target, text=self._target_band_text(), bg=theme.BG, fg=theme.INK_60,
            font=self._f_small,
        )
        self._target_band_label.pack(pady=(2, 0))

        tk.Frame(self._phase2_frame, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(18, 0))

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

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _handle_back(self) -> None:
        self._cancel_rest_timer()  # D10: the post-session modal's grab does
        # not reliably block every exit path -- guard against a dangling
        # after() callback on a window this is about to leave/destroy.
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
        self._cancel_rest_timer()  # D10: see _handle_back
        self._close_modal()  # E5: same guard as _handle_back
        self._finalize_recording("closed")  # D8: every exit path finalizes
        self.release_camera()
        self.destroy()

    def _cancel_rest_timer(self) -> None:
        """D10: the post-session modal's rest countdown is owned by its own
        Continue handler (modals.py), but a WM_DELETE_WINDOW close (or Back,
        should the window manager ever deliver one past the modal's grab)
        must not leave a self.after() callback scheduled against a window
        that is about to be destroyed."""
        if self._rest_after_id is not None:
            self.after_cancel(self._rest_after_id)
            self._rest_after_id = None
