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

CSV recording, trial sequencing, the REC timer, and target-force values stay
static placeholders -- Start/Stop/Next Trial are stubs beyond the baseline
gate and the D2 Re-capture lockout.
"""
import tkinter as tk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.calibration import CameraCalibration
from core.detector_search import MIN_ACCEPTABLE, TARGET_MARKERS, SearchResult, region_counts
from core.roi import DEFAULT_ALPHA, ROICrop
from core.tracker import BASELINE_FRAMES, Tracker
from output import participants
from output.participants import ParticipantRecord
from ui import theme

WINDOW_W = 880
WINDOW_H = 740
FEED_W = 450
FEED_H = 600

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps, matching viewer_window

_LETTERBOX_BG = tuple(int(theme.SURFACE.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

_BASELINE_IDLE: dict = {
    "status": "none", "frame": 0, "count": None,
    "overridden": False, "corners": None, "bands": None, "subline": "",
}


class SessionWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Callable[[], None] | None = None,
        detector_params: dict | None = None,
        calibration_result: SearchResult | None = None,
        participant: ParticipantRecord | None = None,
    ) -> None:
        super().__init__(master)
        self._on_back = on_back or (lambda: print("Back clicked"))
        self._calibration_result = calibration_result
        self._participant = participant

        self.title("GripVT — Session Recording")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._current_img: ImageTk.PhotoImage | None = None
        self._after_id: str | None = None
        self._cap: cv2.VideoCapture | None = None

        self._baseline: dict = dict(_BASELINE_IDLE)
        self._baseline_frames: list[np.ndarray] = []
        self._recording_started = False
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

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cal_w, cal_h = self._cal.image_size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            self._cap = cap
            self._poll()
        else:
            cap.release()
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

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=32, pady=(18, 18))

        self._build_topbar(body)

        content = tk.Frame(body, bg=theme.BG)
        content.pack(fill="both", expand=True, pady=(18, 0))
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._build_feed(content)
        self._build_panel(content)

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

        tk.Frame(bar, bg=theme.BG, width=76, height=1).grid(row=0, column=2, sticky="e")

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

        # Built once; only placed once recording starts -- see _on_start.
        badge = tk.Frame(slot, bg=theme.INK)
        badge_inner = tk.Frame(badge, bg=theme.INK)
        badge_inner.pack(padx=12, pady=6)
        dot = tk.Canvas(badge_inner, width=9, height=9, bg=theme.INK, highlightthickness=0)
        dot.create_oval(0, 0, 9, 9, fill=theme.ACCENT, outline=theme.ACCENT)
        dot.pack(side="left")
        tk.Label(
            badge_inner, text="REC 00:00", bg=theme.INK, fg=theme.BG, font=self._f_rec,
        ).pack(side="left", padx=(6, 0))
        self._rec_badge = badge

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
        stop = self._build_button(row, "Stop", "secondary", "stop", self._on_stop, enabled=False)
        stop.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        next_btn = self._build_button(
            panel, "Next Trial", "secondary", "arrow-right", self._on_next_trial, enabled=False,
        )
        next_btn.pack(fill="x", pady=(12, 0))

        self._render_baseline()

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

        self._progress_block(self._phase2_frame, "TRIAL", "12 / 30", 0.40)
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
        self._start_btn.set_enabled(locked)

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
        row.pack(fill="x")
        row.columnconfigure(0, weight=1, uniform="fail")
        row.columnconfigure(1, weight=1, uniform="fail")
        # D2: recording hasn't started yet in this substate (Start can't be
        # pressed until locked), but guard uniformly for defense in depth.
        enabled = not self._recording_started
        self._build_button(
            row, "Re-capture", "primary", None, self._on_recapture_baseline_click,
            enabled=enabled,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._build_button(
            row, "Continue Anyway", "ghost", None, self._on_override_baseline_click,
            fg_override=theme.ACCENT_700, bg_override=theme.ACCENT_10,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

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
        tk.Label(
            text_col, text=count_text, bg=fill, fg=theme.INK_60, font=self._f_strip_subline,
            anchor="w", wraplength=220, justify="left",
        ).pack(fill="x", pady=(2, 0))

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
            detections = [
                (s.baseline_pos[0], s.baseline_pos[1], s.baseline_area)
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

    def _warning_icon(self, parent: tk.Misc, bg: str) -> tk.Canvas:
        c = tk.Canvas(parent, width=24, height=24, bg=bg, highlightthickness=0)
        c.create_polygon(
            12, 3, 22, 20, 2, 20, fill="", outline=theme.ACCENT_700, width=1.8, joinstyle="round",
        )
        c.create_line(12, 9, 12, 13, fill=theme.ACCENT_700, width=1.8, capstyle="round")
        c.create_oval(11.2, 15.2, 12.8, 16.8, fill=theme.ACCENT_700, outline=theme.ACCENT_700)
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
    ) -> tk.Frame:
        fill, fg, border = self._button_style(kind, enabled, fg_override, bg_override)

        btn = tk.Frame(
            parent, bg=fill, highlightthickness=1,
            highlightbackground=border, highlightcolor=border,
        )
        inner = tk.Frame(btn, bg=fill)
        inner.pack(padx=14, pady=8)

        widgets: list[tk.Widget] = [btn, inner]
        icon_widget: tk.Canvas | None = None
        if icon in ("arrow-left", "play", "stop", "dot"):
            icon_widget = self._icon(inner, icon, fg, fill)
            icon_widget.pack(side="left", padx=(0, 6))
            widgets.append(icon_widget)
        text = tk.Label(inner, text=label, bg=fill, fg=fg, font=self._f_button)
        text.pack(side="left")
        widgets.append(text)
        if icon == "arrow-right":
            icon_widget = self._icon(inner, icon, fg, fill)
            icon_widget.pack(side="left", padx=(6, 0))
            widgets.append(icon_widget)

        wired = False

        def repaint(en: bool) -> str:
            fill_, fg_, border_ = self._button_style(kind, en, fg_override, bg_override)
            for w in widgets:
                w.configure(bg=fill_)
            text.configure(fg=fg_)
            btn.configure(highlightbackground=border_, highlightcolor=border_)
            if icon_widget is not None:
                icon_widget.delete("all")
                self._paint_icon_shape(icon_widget, icon, fg_)
            return fill_

        def set_enabled(en: bool) -> None:
            nonlocal wired
            fill_ = repaint(en)
            if en and not wired:
                hover, active = self._button_hover_active(kind)
                self._wire_button(btn, widgets, fill_, hover, active, command)
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
        fill = bg_override if bg_override is not None else theme.BG
        return fill, theme.INK, theme.DIVIDER

    def _button_hover_active(self, kind: str) -> tuple[str, str]:
        if kind == "primary":
            return theme.ACCENT_600, theme.ACCENT_700
        if kind == "ghost":
            return theme.ACCENT_10, theme.ACCENT_18
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

    def _wire_button(
        self, btn: tk.Frame, widgets: list[tk.Widget], base: str,
        hover: str, active: str, command: Callable[[], None],
    ) -> None:
        def paint(colour: str) -> None:
            for w in widgets:
                w.configure(bg=colour)

        def on_enter(_e: object) -> None:
            paint(hover)

        def on_leave(_e: object) -> None:
            paint(base)

        def on_press(_e: object) -> None:
            paint(active)

        def on_release(_e: object) -> None:
            x, y = btn.winfo_pointerxy()
            hovered = btn.winfo_containing(x, y)
            if hovered is not None and (hovered is btn or str(hovered).startswith(str(btn))):
                paint(hover)
                command()
            else:
                paint(base)

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
        if not self._locked():
            return
        self._recording_started = True
        self._rec_badge.place(x=12, y=12)
        if self._strip_recapture_btn is not None:
            self._strip_recapture_btn.set_enabled(False)
        print("Start clicked")

    def _on_stop(self) -> None:
        print("Stop clicked")

    def _on_next_trial(self) -> None:
        print("Next Trial clicked")

    # ── camera lifecycle ────────────────────────────────────────────────────
    def _poll(self) -> None:
        ret, frame = self._cap.read()
        if ret:
            if self._baseline["status"] == "capturing":
                self._capture_baseline_frame(frame)
            display = self._tracker.undistort(frame)
            display = self._roi.crop(display)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = _fit_letterbox(Image.fromarray(rgb), FEED_W, FEED_H)
            photo = ImageTk.PhotoImage(img)
            self._feed_label.configure(image=photo)
            self._current_img = photo  # plain tk.Label drops the image without this ref
        self._after_id = self.after(POLL_MS, self._poll)

    def release_camera(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _handle_back(self) -> None:
        self.release_camera()
        self._on_back()

    def _on_close(self) -> None:
        self.release_camera()
        self.destroy()


def _fit_letterbox(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (box_w, box_h), _LETTERBOX_BG)
    canvas.paste(resized, ((box_w - new_w) // 2, (box_h - new_h) // 2))
    return canvas
