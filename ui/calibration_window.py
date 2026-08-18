"""Sensor Calibration window for the GripVT desktop app.

Sits between New Participant and Session Recording as the per-wear calibration
step: captures the sensor at three hand poses, searches thresh/min_area for one
param set that keeps all 99 markers detectable at the worst pose, records the
per-pose diagnostics into the participant's folder, and hands the winning params
to the session. Visual design ported from the Modernist design system (Claude
Design: "GripVT Sensor Calibration").

Baseline capture stays a Session-window control -- this window only settles
detector params.
"""
import threading
import time
import tkinter as tk
from dataclasses import replace
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.calibration import CameraCalibration
from core.detector import denoise, detect, load_params, preprocess
from core.detector_search import (
    MIN_AREA_RANGE,
    POSE_KEYS,
    TARGET_MARKERS,
    THRESH_RANGE,
    AnchorError,
    SearchResult,
    clamp_manual_params,
    rescore,
    search,
)
from core.roi import DEFAULT_ALPHA, ROICrop
from core.tracker import BASELINE_FRAMES, Tracker
from output import participants
from output.participants import ParticipantRecord
from ui import theme

WINDOW_W = 880
# 740 was the pre-Phase-4 height. 800 covered the collapsed override toggle
# but not the panel itself open and diverged from auto (measured in-rig
# 2026-08-19: _actions' natural bottom sits ~27px past 800, clipping
# "Continue anyway"). 850 covers that with real margin -- screen is 1920x1080
# here, so 850 still leaves ~230px of headroom below the window.
WINDOW_H = 850
FEED_W = 450
FEED_H = 600

CAMERA_INDEX = 1
POLL_MS = 33
# A single dropped read at 30fps is normal jitter -- only a sustained streak
# means the camera actually disconnected. Mirrors ui/session/camera_feed.py:47.
CAMERA_FAIL_POLLS = 15  # ~0.5s at POLL_MS
# Live detection is visual feedback only, so it runs at ~10Hz -- illumination
# flattening (CLAHE + a large top-hat) costs more than the 33ms frame budget.
DETECT_EVERY = 3

_TOAST_TIMEOUT_MS = 5000
_FADE_STEPS = 8
_FADE_INTERVAL_MS = 50

# Manual-override stepper bounds (Phase 4, D15) -- derived from the sweep's
# own range so the widget can never drift from clamp_manual_params.
THRESH_LO, THRESH_HI, _ = THRESH_RANGE
AREA_LO, AREA_HI, _ = MIN_AREA_RANGE
_RESCORE_DEBOUNCE_MS = 250

_LETTERBOX_BG = tuple(int(theme.SURFACE.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
_RING_BGR = tuple(int(theme.ACCENT.lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))

POSES: list[tuple[str, str, str]] = [
    ("relaxed", "RELAXED", "Rest the forearm and let the hand hang naturally open, and hold "
                           "still. Capture the session baseline from this same pose."),
    ("full_open", "FULL OPEN", "Extend and spread the fingers as wide as comfortably "
                               "possible, palm facing away, and hold."),
    ("max_grip", "MAX GRIP", "Squeeze the dynamometer at maximum effort and hold until the "
                             "burst completes."),
]
POSE_NAMES = {key: name for key, name, _instruction in POSES}
POSE_SHORT = {"relaxed": "relaxed", "full_open": "open", "max_grip": "grip"}


class CalibrationWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        record: ParticipantRecord,
        on_back: Callable[[], None] | None = None,
        on_continue: Callable[[SearchResult, dict], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._record = record
        self._on_back = on_back or (lambda: print("Back clicked"))
        # F6a: on_continue now also receives the pending-calibration bundle
        # (see _continue below) -- nothing is written to disk here anymore,
        # so the caller (main.open_session) is responsible for flushing it
        # into the session directory once one exists (D3).
        self._on_continue = on_continue or (
            lambda result, pending: print(f"Continue {result.params['thresh']}")
        )

        self.title("GripVT — Sensor Calibration")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._pose_state: dict[str, str] = {key: "pending" for key in POSE_KEYS}
        self._pose_counts: dict[str, int] = {}
        self._pose_frames: dict[str, np.ndarray] = {}
        self._pose_dets: dict[str, list] = {}
        self._selected = POSE_KEYS[0]
        self._review: str | None = None

        self._status = "idle"
        self._result: SearchResult | None = None
        self._error_message: str | None = None
        self._overridden = False
        self._manual_override = False
        self._auto_params: dict | None = None
        self._auto_worst_count: int | None = None

        # Manual-override panel (Phase 4 UI). _manual_thresh/_manual_min_area
        # hold the stepper's live values while the panel is open; they start
        # at auto_params on open, so the panel shows no divergence until the
        # operator actually adjusts something.
        self._override_open = False
        self._manual_thresh = 0
        self._manual_min_area = 0.0
        self._manual_scoring = False
        self._rescore_after: str | None = None

        self._burst: list[np.ndarray] = []
        self._capturing: str | None = None

        self._params: dict = load_params()
        self._live_dets: list = []
        self._frame_tick = 0

        self._current_img: ImageTk.PhotoImage | None = None
        self._after_id: str | None = None
        self._search_after: str | None = None
        self._toast_after: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._read_fail_streak = 0
        self._disconnect_at: float | None = None

        self._init_fonts()
        self._build_body()
        theme.apply_dark_titlebar(self)

        self._roi = ROICrop()
        roi_config = self._roi.config
        self._alpha = roi_config.alpha if roi_config is not None else DEFAULT_ALPHA
        self._cal = CameraCalibration(alpha=self._alpha)
        self._tracker = Tracker(self._cal, self._roi)

        self._open_camera()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _center(self, w: int, h: int) -> None:
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _init_fonts(self) -> None:
        self._f_eyebrow = theme.make_font(self, "semibold", 11)
        self._f_chip = theme.make_font(self, "body", 11)
        self._f_link = theme.make_font(self, "body", 11)
        self._f_link.configure(underline=True)
        self._f_badge = theme.make_font(self, "heading", 12)
        self._f_pose_num = theme.make_font(self, "heading", 11)
        self._f_pose = theme.make_font(self, "heading", 13)
        self._f_count = theme.make_font(self, "body", 12)
        self._f_instruction = theme.make_font(self, "body", 13)
        self._f_kicker = theme.make_font(self, "body", 11)
        self._f_burst = theme.make_font(self, "heading", 13)
        self._f_big = theme.make_font(self, "heading", 64)
        self._f_big_idle = theme.make_font(self, "heading", 56)
        self._f_unit = theme.make_font(self, "heading", 24)
        self._f_small = theme.make_font(self, "body", 12)
        self._f_fail = theme.make_font(self, "semibold", 12)
        self._f_button = theme.make_font(self, "heading", 14)
        self._f_toast = theme.make_font(self, "body", 11)
        self._f_big_override = theme.make_font(self, "heading", 34)
        self._f_override_toggle = theme.make_font(self, "heading", 11)
        self._f_override_badge = theme.make_font(self, "heading", 10)
        self._f_micro = theme.make_font(self, "body", 10)
        self._f_micro_bold = theme.make_font(self, "heading", 10)
        self._f_stepper_value = theme.make_font(self, "heading", 14)
        self._f_stepper_btn = theme.make_font(self, "heading", 15)
        self._f_score = theme.make_font(self, "heading", 13)

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=32, pady=(22, 20))

        self._build_topbar(body)

        content = tk.Frame(body, bg=theme.BG)
        content.pack(fill="both", expand=True, pady=(14, 0))
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._build_feed(content)
        self._build_panel(content)
        self._build_toast(body)

    def _build_topbar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=theme.BG)
        bar.pack(fill="x")
        bar.columnconfigure(0, weight=1, uniform="t")
        bar.columnconfigure(2, weight=1, uniform="t")

        self._build_button(bar, "Back", "secondary", "arrow-left", self._handle_back).grid(
            row=0, column=0, sticky="w"
        )

        eyebrow = tk.Frame(bar, bg=theme.BG)
        eyebrow.grid(row=0, column=1)
        tk.Frame(eyebrow, bg=theme.ACCENT, width=8, height=8).pack(side="left")
        tk.Label(
            eyebrow, text="GRIPVT", bg=theme.BG, fg=theme.INK_65, font=self._f_eyebrow
        ).pack(side="left", padx=(8, 0))

        chip_wrap = tk.Frame(bar, bg=theme.BG)
        chip_wrap.grid(row=0, column=2, sticky="e")
        chip = tk.Frame(
            chip_wrap, bg=theme.BG, highlightthickness=1,
            highlightbackground=theme.ACCENT, highlightcolor=theme.ACCENT,
        )
        chip.pack(side="right")
        tk.Label(
            chip, text=self._record.participant_id.replace("participant", "PARTICIPANT "),
            bg=theme.BG, fg=theme.ACCENT, font=self._f_chip,
        ).pack(padx=10, pady=3)

    def _build_feed(self, parent: tk.Frame) -> None:
        holder = tk.Frame(parent, bg=theme.BG, width=FEED_W, height=FEED_H)
        holder.grid(row=0, column=0, sticky="n")
        holder.grid_propagate(False)

        self._slot = tk.Frame(
            holder, bg=theme.BG, highlightthickness=2,
            highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        self._slot.place(x=0, y=0, width=FEED_W, height=FEED_H)

        self._feed_label = tk.Label(self._slot, bg=theme.SURFACE, text="")
        self._feed_label.pack(fill="both", expand=True)

        self._count_badge = tk.Frame(self._slot, bg=theme.INK)
        badge_inner = tk.Frame(self._count_badge, bg=theme.INK)
        badge_inner.pack(padx=12, pady=6)
        tk.Frame(badge_inner, bg=theme.ACCENT, width=7, height=7).pack(side="left")
        self._count_label = tk.Label(
            badge_inner, text="", bg=theme.INK, fg=theme.BG, font=self._f_badge,
        )
        self._count_label.pack(side="left", padx=(8, 0))

        self._view_stack = tk.Frame(self._slot, bg=theme.INK)
        view_box = tk.Frame(self._view_stack, bg=theme.INK)
        view_box.pack(anchor="e")
        self._view_label = tk.Label(
            view_box, text="LIVE", bg=theme.INK, fg=theme.BG, font=self._f_badge,
        )
        self._view_label.pack(padx=12, pady=6)
        self._live_btn = tk.Label(
            self._view_stack, text="BACK TO LIVE", bg=theme.INK, fg=theme.BG,
            font=self._f_link, cursor="hand2",
        )
        self._live_btn.bind("<Button-1>", lambda _e: self._back_to_live())

        self._unavailable = tk.Frame(self._slot, bg=theme.BG)
        inner = tk.Frame(self._unavailable, bg=theme.BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            inner, text="CAMERA UNAVAILABLE", bg=theme.BG, fg=theme.INK,
            font=theme.make_font(self, "heading", 14),
        ).pack()
        tk.Label(
            inner, text="Check the USB sensor connection and try again.",
            bg=theme.BG, fg=theme.INK_55, font=self._f_small, wraplength=300,
        ).pack(pady=(10, 12))
        self._build_button(inner, "Retry", "secondary", None, self.retry_camera).pack()

    def _build_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=theme.BG)
        panel.grid(row=0, column=1, sticky="nsew", padx=(28, 0))

        self._pose_rows = tk.Frame(panel, bg=theme.BG)
        self._pose_rows.pack(fill="x", pady=(0, 14))

        self._instruction = tk.Label(
            panel, text="", bg=theme.BG, fg=theme.INK_60, font=self._f_instruction,
            wraplength=330, justify="left", anchor="w",
        )
        self._instruction.pack(fill="x", pady=(0, 14))

        burst = tk.Frame(panel, bg=theme.BG)
        burst.pack(fill="x", pady=(0, 14))
        head = tk.Frame(burst, bg=theme.BG)
        head.pack(fill="x", pady=(0, 6))
        tk.Label(
            head, text="BURST", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker
        ).pack(side="left")
        self._burst_value = tk.Label(
            head, text=f"0 / {BASELINE_FRAMES}", bg=theme.BG, fg=theme.INK, font=self._f_burst
        )
        self._burst_value.pack(side="right")
        track = tk.Frame(burst, bg=theme.NEUTRAL_200, height=6)
        track.pack(fill="x")
        track.pack_propagate(False)
        self._burst_fill = tk.Frame(track, bg=theme.ACCENT)
        self._burst_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

        tk.Frame(panel, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(0, 0))

        self._result_box = tk.Frame(panel, bg=theme.BG)
        self._result_box.pack(fill="both", expand=True, pady=12)

        self._override_box = tk.Frame(panel, bg=theme.BG)
        self._override_box.pack(fill="x")

        tk.Frame(panel, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(0, 14))

        self._actions = tk.Frame(panel, bg=theme.BG)
        self._actions.pack(fill="x")

        self._render_poses()
        self._render_result()
        self._render_override()
        self._render_actions()

    def _build_toast(self, parent: tk.Frame) -> None:
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

    # ── re-rendered regions ─────────────────────────────────────────────────
    def _render_poses(self) -> None:
        for child in self._pose_rows.winfo_children():
            child.destroy()

        active = self._review or self._selected
        for key, name, _instruction in POSES:
            state = self._pose_state[key]
            selected = key == active
            bg = theme.ACCENT_08 if selected else theme.BG
            row = tk.Frame(self._pose_rows, bg=bg)
            row.pack(fill="x")
            inner = tk.Frame(row, bg=bg)
            inner.pack(fill="x", padx=10, pady=9)

            num = tk.Frame(
                inner, bg=bg, highlightthickness=1,
                highlightbackground=theme.INK, highlightcolor=theme.INK,
            )
            num.pack(side="left")
            num_label = tk.Label(
                num, text=str(POSE_KEYS.index(key) + 1), bg=bg, fg=theme.INK,
                font=self._f_pose_num, width=2,
            )
            num_label.pack(pady=1)
            name_label = tk.Label(inner, text=name, bg=bg, fg=theme.INK, font=self._f_pose)
            name_label.pack(side="left", padx=(12, 0))

            marker = tk.Canvas(inner, width=15, height=15, bg=bg, highlightthickness=0)
            marker.pack(side="right")
            self._draw_state_marker(marker, state)
            count = tk.Label(
                inner,
                text=f"({self._pose_counts[key]})" if key in self._pose_counts else "",
                bg=bg, fg=theme.INK_55, font=self._f_count,
            )
            count.pack(side="right", padx=(0, 10))

            tk.Frame(row, bg=theme.NEUTRAL_200, height=1).pack(fill="x")

            group = (row, inner, num, num_label, name_label, marker, count)

            def paint(colour: str, widgets: tuple = group) -> None:
                for w in widgets:
                    w.configure(bg=colour)

            for widget in group:
                widget.bind("<Button-1>", lambda _e, k=key: self._select_pose(k))
                if not selected:
                    widget.bind("<Enter>", lambda _e, p=paint: p(theme.ACCENT_06))
                    widget.bind("<Leave>", lambda _e, p=paint: p(theme.BG))
                widget.configure(cursor="hand2")

    def _draw_state_marker(self, canvas: tk.Canvas, state: str) -> None:
        if state == "pending":
            canvas.create_oval(1, 1, 14, 14, outline=theme.DIVIDER, width=2)
        elif state == "capturing":
            canvas.create_oval(1, 1, 14, 14, outline=theme.ACCENT, fill=theme.ACCENT)
        else:
            canvas.create_line(2, 8, 6, 12, fill=theme.INK, width=2, capstyle="round")
            canvas.create_line(6, 12, 13, 3, fill=theme.INK, width=2, capstyle="round")

    def _render_result(self) -> None:
        for child in self._result_box.winfo_children():
            child.destroy()

        wrap = tk.Frame(self._result_box, bg=theme.BG)
        # pack (not place) so wrap's real size counts toward _result_box's
        # requested size -- a place()-centered child can silently overflow
        # into the override toggle packed below it if _result_box shrinks
        # to make room for that toggle (found in-rig 2026-08-18: the fail
        # message was getting drawn under the toggle row, not "cut off").
        wrap.pack(expand=True)

        if self._result is not None and self._result.manual_override:
            badge = tk.Frame(wrap, bg=theme.INK)
            badge.pack(pady=(0, 10))
            badge_inner = tk.Frame(badge, bg=theme.INK)
            badge_inner.pack(padx=10, pady=5)
            tk.Frame(badge_inner, bg=theme.BG, width=6, height=6).pack(side="left")
            tk.Label(
                badge_inner, text="MANUAL OVERRIDE ACTIVE", bg=theme.INK, fg=theme.BG,
                font=self._f_override_badge,
            ).pack(side="left", padx=(7, 0))

        tk.Label(
            wrap, text="WORST-POSE MARKERS", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker
        ).pack(pady=(0, 10))

        if self._status == "running":
            tk.Label(
                wrap, text="Searching…", bg=theme.BG, fg=theme.INK, font=self._f_burst
            ).pack()
            tk.Label(
                wrap, text=f"Scoring {TARGET_MARKERS} markers across three poses.",
                bg=theme.BG, fg=theme.INK_55, font=self._f_small, wraplength=300,
            ).pack(pady=(10, 0))
            return

        result = self._result
        if result is None:
            if self._status == "error" and self._error_message:
                tk.Label(
                    wrap, text="—", bg=theme.BG, fg=theme.ACCENT_700, font=self._f_big_idle
                ).pack()
                tk.Label(
                    wrap, text=self._error_message, bg=theme.BG, fg=theme.ACCENT_700,
                    font=self._f_fail, wraplength=320, justify="center",
                ).pack(pady=(8, 0))
                return
            tk.Label(
                wrap, text="—", bg=theme.BG, fg=theme.INK_35, font=self._f_big_idle
            ).pack()
            tk.Label(
                wrap, text="Capture all three poses, then run calibration.",
                bg=theme.BG, fg=theme.INK_55, font=self._f_small, wraplength=300,
            ).pack(pady=(8, 0))
            return

        worst_n = result.poses[result.worst_pose].n_detected
        colour = theme.INK if result.passed else theme.ACCENT_700
        # The override panel shares this fixed-height column, so the number
        # shrinks while it is open to leave the steppers room without
        # overflowing the window (D18 wiring).
        big_font = self._f_big_override if self._override_open else self._f_big
        row = tk.Frame(wrap, bg=theme.BG)
        row.pack()
        tk.Label(
            row, text=str(worst_n), bg=theme.BG, fg=colour, font=big_font
        ).pack(side="left")
        tk.Label(
            row, text=f"/ {TARGET_MARKERS}", bg=theme.BG, fg=colour, font=self._f_unit
        ).pack(side="left", anchor="s", pady=(0, 10))
        if not self._override_open:
            tk.Label(
                wrap, text=self._subline(result), bg=theme.BG, fg=theme.INK_55,
                font=self._f_small, wraplength=320,
            ).pack(pady=(8, 0))
        if not result.passed and not self._override_open:
            if worst_n > TARGET_MARKERS:
                fail_text = (
                    f"{POSE_NAMES[result.worst_pose]} detected extra blobs "
                    f"({worst_n} / {TARGET_MARKERS}), likely reflections. Recapture and try "
                    "again."
                )
            else:
                fail_text = (
                    f"{POSE_NAMES[result.worst_pose]} below threshold "
                    f"({worst_n} / {TARGET_MARKERS}). Reseat the sensor and try again."
                )
            tk.Label(
                wrap, text=fail_text, bg=theme.BG, fg=theme.ACCENT_700, font=self._f_fail,
                wraplength=320, justify="center",
            ).pack(pady=(8, 0))

    def _subline(self, result: SearchResult) -> str:
        counts = " · ".join(
            f"{POSE_SHORT[key]} {result.poses[key].n_detected}" for key in POSE_KEYS
        )
        worst = result.poses[result.worst_pose]
        extras = [
            f"{cell} {worst.corners[cell]}/{result.reference_corners[cell]}"
            for cell in sorted(result.reference_corners)
            if worst.corners[cell] < result.reference_corners[cell]
        ]
        if worst.oversized:
            extras.append(f"oversized {worst.oversized}")
        return counts + ("  |  " + " · ".join(extras) if extras else "")

    # ── manual override panel (Phase 4 UI) ──────────────────────────────────
    def _render_override(self) -> None:
        for child in self._override_box.winfo_children():
            child.destroy()
        if self._status != "fail":
            # D12: never surface the override outside a failed calibration.
            return

        # No divider of its own -- the existing divider between _override_box
        # and _actions already separates this from the buttons below, and a
        # second one here only ate space the result box needed (in-rig
        # 2026-08-18: caused the fail message to be drawn under this row).
        toggle = tk.Frame(self._override_box, bg=theme.BG, cursor="hand2")
        toggle.pack(anchor="w", pady=(6, 0))
        caret = tk.Canvas(toggle, width=12, height=12, bg=theme.BG, highlightthickness=0)
        self._draw_caret(caret, self._override_open)
        caret.pack(side="left", padx=(0, 7))
        label = tk.Label(
            toggle, text="MANUAL OVERRIDE", bg=theme.BG, fg=theme.INK,
            font=self._f_override_toggle,
        )
        label.pack(side="left")
        for w in (toggle, caret, label):
            w.bind("<Button-1>", lambda _e: self._toggle_override())
            w.bind("<Enter>", lambda _e: label.configure(fg=theme.ACCENT_700))
            w.bind("<Leave>", lambda _e: label.configure(fg=theme.INK))

        if not self._override_open:
            return

        diverged = self._manual_diverged()
        border, width = (theme.INK, 2) if diverged else (theme.NEUTRAL_200, 1)
        box = tk.Frame(
            self._override_box, bg=theme.BG, highlightthickness=width,
            highlightbackground=border, highlightcolor=border,
        )
        box.pack(fill="x", pady=(8, 0))
        inner = tk.Frame(box, bg=theme.BG)
        inner.pack(fill="x", padx=11, pady=9)
        self._render_override_panel(inner)

    def _render_override_panel(self, inner: tk.Frame) -> None:
        steppers = tk.Frame(inner, bg=theme.BG)
        steppers.pack(fill="x", pady=(0, 9))
        steppers.columnconfigure(0, weight=1, uniform="s")
        steppers.columnconfigure(1, weight=1, uniform="s")

        thresh_col = self._build_stepper(
            steppers, "THRESHOLD", f"{THRESH_LO}–{THRESH_HI}", str(self._manual_thresh),
            lambda: self._step_manual("thresh", -1), lambda: self._step_manual("thresh", 1),
            self._manual_thresh > THRESH_LO, self._manual_thresh < THRESH_HI,
        )
        thresh_col.grid(row=0, column=0, sticky="ew", padx=(0, 12))

        area_col = self._build_stepper(
            steppers, "MIN AREA", f"{AREA_LO}–{AREA_HI}", str(int(self._manual_min_area)),
            lambda: self._step_manual("min_area", -1), lambda: self._step_manual("min_area", 1),
            self._manual_min_area > AREA_LO, self._manual_min_area < AREA_HI,
        )
        area_col.grid(row=0, column=1, sticky="ew")

        auto = self._auto_params
        auto_score_text = (
            f"{self._auto_worst_count} / {TARGET_MARKERS}"
            if self._auto_worst_count is not None else "—"
        )
        auto_params_text = (
            f"thresh {int(auto['thresh'])} · min area {int(auto['min_area'])}"
            if auto is not None else "—"
        )
        self._build_score_row(
            inner, "AUTO", auto_params_text, auto_score_text, theme.INK_55,
        ).pack(fill="x")

        manual_params_text = (
            f"thresh {self._manual_thresh} · min area {int(self._manual_min_area)}"
        )
        if self._manual_scoring:
            manual_score_text, manual_colour = "scoring…", theme.INK_45
        elif self._result is not None and self._result.manual_override:
            n = self._result.poses[self._result.worst_pose].n_detected
            manual_score_text = f"{n} / {TARGET_MARKERS}"
            manual_colour = theme.INK if self._result.passed else theme.ACCENT_700
        else:
            manual_score_text, manual_colour = auto_score_text, theme.INK_55
        self._build_score_row(
            inner, "MANUAL", manual_params_text, manual_score_text, manual_colour,
        ).pack(fill="x")

    def _build_stepper(
        self, parent: tk.Frame, label: str, bounds_text: str, value_text: str,
        on_dec: Callable[[], None], on_inc: Callable[[], None],
        dec_enabled: bool, inc_enabled: bool,
    ) -> tk.Frame:
        col = tk.Frame(parent, bg=theme.BG)
        head = tk.Frame(col, bg=theme.BG)
        head.pack(fill="x", pady=(0, 5))
        tk.Label(
            head, text=label, bg=theme.BG, fg=theme.INK_60, font=self._f_micro_bold
        ).pack(side="left")
        tk.Label(
            head, text=bounds_text, bg=theme.BG, fg=theme.INK_35, font=self._f_micro
        ).pack(side="right")

        box = tk.Frame(
            col, bg=theme.BG, highlightthickness=1,
            highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        box.pack(fill="x")

        dec = self._build_step_button(box, "−", on_dec, dec_enabled)
        dec.pack(side="left", ipady=5)
        tk.Frame(box, bg=theme.INK, width=1).pack(side="left", fill="y")
        tk.Label(
            box, text=value_text, bg=theme.BG, fg=theme.INK, font=self._f_stepper_value,
        ).pack(side="left", expand=True, fill="both", ipady=5)
        tk.Frame(box, bg=theme.INK, width=1).pack(side="left", fill="y")
        inc = self._build_step_button(box, "+", on_inc, inc_enabled)
        inc.pack(side="right", ipady=5)
        return col

    def _build_step_button(
        self, parent: tk.Misc, symbol: str, command: Callable[[], None], enabled: bool,
    ) -> tk.Label:
        """One press == one step, full stop -- no hold-to-repeat (Guillaume,
        in-rig 2026-08-18: wants full manual control over each step). Also
        sidesteps a real bug a repeat timer had: _step_manual rebuilds this
        very widget on every call, which would destroy the pressed widget
        mid-hold and drop the ButtonRelease that was supposed to cancel it,
        so the repeat never stopped."""
        fg = theme.INK if enabled else theme.INK_35
        btn = tk.Label(
            parent, text=symbol, bg=theme.BG, fg=fg, font=self._f_stepper_btn,
            width=3, cursor="hand2" if enabled else "arrow",
        )
        if enabled:
            def on_release(_e: object) -> None:
                x, y = btn.winfo_pointerxy()
                hovered = btn.winfo_containing(x, y)
                if hovered is btn:
                    command()

            btn.bind("<Enter>", lambda _e: btn.configure(bg=theme.ACCENT_10))
            btn.bind("<Leave>", lambda _e: btn.configure(bg=theme.BG))
            btn.bind("<ButtonPress-1>", lambda _e: btn.configure(bg=theme.ACCENT_18))
            btn.bind("<ButtonRelease-1>", on_release)
        return btn

    def _build_score_row(
        self, parent: tk.Frame, label: str, params_text: str, score_text: str, colour: str,
    ) -> tk.Frame:
        row = tk.Frame(parent, bg=theme.BG)
        tk.Frame(row, bg=theme.NEUTRAL_200, height=1).pack(fill="x")
        inner = tk.Frame(row, bg=theme.BG)
        inner.pack(fill="x", pady=4)
        tk.Label(
            inner, text=label, bg=theme.BG, fg=theme.INK_55, font=self._f_micro_bold,
            width=7, anchor="w",
        ).pack(side="left")
        tk.Label(
            inner, text=score_text, bg=theme.BG, fg=colour, font=self._f_score,
        ).pack(side="right")
        tk.Label(
            inner, text=params_text, bg=theme.BG, fg=theme.INK_55, font=self._f_micro,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        return row

    def _draw_caret(self, canvas: tk.Canvas, open_: bool) -> None:
        if open_:
            canvas.create_line(2, 4, 6, 9, 10, 4, fill=theme.INK, width=2,
                                capstyle="round", joinstyle="round")
        else:
            canvas.create_line(4, 2, 9, 6, 4, 10, fill=theme.INK, width=2,
                                capstyle="round", joinstyle="round")

    def _manual_diverged(self) -> bool:
        if self._auto_params is None:
            return False
        return (
            self._manual_thresh != int(self._auto_params["thresh"])
            or self._manual_min_area != float(self._auto_params["min_area"])
        )

    def _toggle_override(self) -> None:
        if self._override_open:
            self._close_override()
        else:
            self._open_override()
        self._refresh()

    def _open_override(self) -> None:
        if self._auto_params is not None:
            # The winning auto params are not guaranteed to sit inside the
            # sweep's own range (_candidates() always keeps the seed even
            # when it falls outside THRESH_RANGE/MIN_AREA_RANGE), so clamp
            # to whatever the stepper can actually represent (D15).
            self._manual_thresh, self._manual_min_area = clamp_manual_params(
                self._auto_params["thresh"], self._auto_params["min_area"]
            )
        self._manual_scoring = False
        self._override_open = True

    def _close_override(self) -> None:
        self._reset_override_state()

    def _reset_override_state(self) -> None:
        if self._rescore_after is not None:
            self.after_cancel(self._rescore_after)
            self._rescore_after = None
        self._override_open = False
        self._manual_scoring = False
        self._manual_thresh = 0
        self._manual_min_area = 0.0

    def _step_manual(self, key: str, delta: int) -> None:
        if key == "thresh":
            new_thresh, _ = clamp_manual_params(self._manual_thresh + delta, self._manual_min_area)
            if new_thresh == self._manual_thresh:
                return
            self._manual_thresh = new_thresh
        else:
            _, new_area = clamp_manual_params(self._manual_thresh, self._manual_min_area + delta)
            if new_area == self._manual_min_area:
                return
            self._manual_min_area = new_area
        self._render_result()
        self._render_override()
        self._schedule_rescore()

    def _schedule_rescore(self) -> None:
        if self._rescore_after is not None:
            self.after_cancel(self._rescore_after)
        self._manual_scoring = True
        self._render_override()
        self._rescore_after = self.after(_RESCORE_DEBOUNCE_MS, self._trigger_rescore)

    def _trigger_rescore(self) -> None:
        self._rescore_after = None
        self._manual_rescore(self._manual_thresh, self._manual_min_area)

    def _render_actions(self) -> None:
        for child in self._actions.winfo_children():
            child.destroy()

        all_captured = all(self._pose_state[key] == "captured" for key in POSE_KEYS)
        busy = self._status == "running"

        if busy:
            self._build_button(
                self._actions, "Searching…", "primary", None, lambda: None, enabled=False
            ).pack(fill="x", pady=(0, 10))
        elif self._status == "fail":
            row = tk.Frame(self._actions, bg=theme.BG)
            row.pack(fill="x", pady=(0, 10))
            row.columnconfigure(0, weight=1, uniform="a")
            row.columnconfigure(1, weight=1, uniform="a")
            self._build_button(row, "Recapture", "secondary", None, self._recapture).grid(
                row=0, column=0, sticky="ew", padx=(0, 6)
            )
            self._build_button(row, "Run Calibration", "primary", None, self._run).grid(
                row=0, column=1, sticky="ew", padx=(6, 0)
            )
        elif self._status == "pass":
            self._build_button(
                self._actions, "Recalibrate", "ghost", None, self._recapture
            ).pack(fill="x", pady=(0, 10))
        elif self._status == "error":
            # No SearchResult exists to retry against -- search() is a pure
            # function of these exact frames, so "Run Calibration" again
            # would just reproduce the same error. Recapture is the only
            # way forward.
            self._build_button(
                self._actions, "Recapture", "primary", None, self._recapture
            ).pack(fill="x", pady=(0, 10))
        elif all_captured:
            self._build_button(
                self._actions, "Run Calibration", "primary", None, self._run
            ).pack(fill="x", pady=(0, 10))
        else:
            pending = self._next_pending()
            label = (
                f"Capturing — {POSE_NAMES[self._capturing].title()}…"
                if self._capturing
                else f"Capture — {POSE_NAMES[pending].title()}" if pending else "Capture"
            )
            self._build_button(
                self._actions, label, "primary", "dot", self._capture,
                enabled=self._capturing is None and self._cap is not None,
            ).pack(fill="x", pady=(0, 10))

        if self._status == "fail":
            self._build_button(
                self._actions, "Continue anyway", "secondary", "arrow-right", self._continue
            ).pack(fill="x")
        else:
            self._build_button(
                self._actions, "Continue", "primary", "arrow-right", self._continue,
                enabled=self._status == "pass",
            ).pack(fill="x")

    def _refresh(self) -> None:
        self._render_poses()
        self._render_result()
        self._render_override()
        self._render_actions()
        self._update_instruction()
        self._update_burst()

    def _update_instruction(self) -> None:
        key = self._capturing or self._review or self._selected
        text = next(i for k, _n, i in POSES if k == key)
        self._instruction.configure(text=text)

    def _update_burst(self) -> None:
        if self._capturing is not None:
            progress = min(len(self._burst), BASELINE_FRAMES)
        else:
            active = self._review or self._selected
            progress = BASELINE_FRAMES if self._pose_state[active] == "captured" else 0
        self._burst_value.configure(text=f"{progress} / {BASELINE_FRAMES}")
        self._burst_fill.place_configure(relwidth=progress / BASELINE_FRAMES)

    # ── actions ─────────────────────────────────────────────────────────────
    def _next_pending(self) -> str | None:
        if self._pose_state[self._selected] == "pending":
            return self._selected
        return next((k for k in POSE_KEYS if self._pose_state[k] == "pending"), None)

    def _select_pose(self, key: str) -> None:
        if self._capturing is not None:
            return
        self._selected = key
        self._review = key if self._pose_state[key] == "captured" else None
        self._refresh()
        self._paint_review()

    def _back_to_live(self) -> None:
        self._review = None
        # Done here rather than waiting on the next poll, so this still clears
        # when the feed has stopped (camera lost after a capture).
        self._live_btn.pack_forget()
        self._view_label.configure(text="LIVE")
        self._refresh()

    def _capture(self) -> None:
        if self._capturing is not None:
            return
        if self._cap is None:
            self._show_toast("Camera unavailable — reconnect the sensor to capture.")
            return
        target = self._next_pending()
        if target is None:
            return
        self._capturing = target
        self._selected = target
        self._review = None
        self._burst = []
        self._pose_state[target] = "capturing"
        self._refresh()

    def _finish_capture(self) -> None:
        key = self._capturing
        assert key is not None
        median = np.median(np.stack(self._burst[:BASELINE_FRAMES]), axis=0).astype(np.uint8)
        dets = self._detect(median)
        self._pose_frames[key] = median
        self._pose_dets[key] = dets
        self._pose_counts[key] = len(dets)
        self._pose_state[key] = "captured"
        self._capturing = None
        self._burst = []
        self._result = None
        self._status = "idle"
        nxt = self._next_pending()
        if nxt is not None:
            self._selected = nxt
        self._refresh()

    def _run(self) -> None:
        if self._status == "running":
            return
        if any(self._pose_state[k] != "captured" for k in POSE_KEYS):
            return
        self._reset_override_state()
        self._status = "running"
        self._result = None
        self._error_message = None
        self._review = None
        self._refresh()

        frames = dict(self._pose_frames)
        base = load_params()
        mask = self._roi.sensor_mask(frames["relaxed"].shape)
        holder: dict = {}

        def work() -> None:
            try:
                holder["result"] = search(frames, base, mask)
            except (AnchorError, ValueError) as exc:
                holder["error"] = str(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._search_after = self.after(100, lambda: self._poll_search(thread, holder))

    def _poll_search(self, thread: threading.Thread, holder: dict) -> None:
        if thread.is_alive():
            self._search_after = self.after(100, lambda: self._poll_search(thread, holder))
            return
        self._search_after = None
        error = holder.get("error")
        if error is not None:
            # AnchorError/ValueError means search() never produced a
            # SearchResult at all (no grid geometry to score against) --
            # distinct from a completed "fail" where every pose was scored
            # but missed 99. Resetting to "idle" here used to leave all
            # three poses stuck "captured" with only a "Run Calibration"
            # action, which just reproduces the same error on the same
            # frames forever with no way back to Recapture (found in-rig
            # 2026-08-19).
            self._status = "error"
            self._error_message = error
            self._result = None
            self._auto_params = None
            self._auto_worst_count = None
            self._reset_override_state()
            participants.save_calibration_failure(
                self._record, self._pose_frames, self._pose_dets, error
            )
            self._show_toast(error)
            self._refresh()
            return
        result: SearchResult = holder["result"]
        self._result = result
        # Held so a later manual re-score can still report what the automatic
        # sweep found, even after self._result is replaced (D17's auto_params).
        self._auto_params = result.params
        self._auto_worst_count = result.poses[result.worst_pose].n_detected
        self._status = "pass" if result.passed else "fail"
        self._params = result.params
        self._pose_dets = dict(result.detections)
        for key in POSE_KEYS:
            self._pose_counts[key] = result.poses[key].n_detected
        if not result.passed:
            self._selected = result.worst_pose
            self._review = result.worst_pose
        self._refresh()
        self._paint_review()

    def _manual_rescore(self, thresh: int, min_area: float) -> None:
        """Backend-only entry point for a manual detector override (Phase 4).
        No widget calls this yet -- Guillaume is designing the layout
        separately in Claude Design (D18) -- but the method is complete: it
        refuses outside a failed calibration (D12), re-scores the same three
        captured frames (D13) against the geometry the automatic search
        already resolved (D14), and never touches the shared seed file
        (D16)."""
        if self._status != "fail" or self._result is None:
            return
        thresh, min_area = clamp_manual_params(thresh, min_area)
        frames = dict(self._pose_frames)
        base = load_params()
        mask = self._roi.sensor_mask(frames["relaxed"].shape)
        geom = self._result.geometry
        reference_corners = self._result.reference_corners
        reference_bands = self._result.reference_bands
        auto_params = self._auto_params
        holder: dict = {}

        def work() -> None:
            try:
                holder["result"] = rescore(
                    frames, base, mask, geom, reference_corners, thresh, min_area,
                    reference_bands=reference_bands,
                )
            except ValueError as exc:
                holder["error"] = str(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._search_after = self.after(
            100, lambda: self._poll_manual_rescore(thread, holder, auto_params)
        )

    def _poll_manual_rescore(
        self, thread: threading.Thread, holder: dict, auto_params: dict | None,
    ) -> None:
        if thread.is_alive():
            self._search_after = self.after(
                100, lambda: self._poll_manual_rescore(thread, holder, auto_params)
            )
            return
        self._search_after = None
        self._manual_scoring = False
        error = holder.get("error")
        if error is not None:
            self._show_toast(error)
            self._refresh()
            return
        result: SearchResult = replace(
            holder["result"], manual_override=True, auto_params=auto_params
        )
        self._result = result
        self._manual_override = True
        self._status = "pass" if result.passed else "fail"
        self._params = result.params
        self._pose_dets = dict(result.detections)
        for key in POSE_KEYS:
            self._pose_counts[key] = result.poses[key].n_detected
        if not result.passed:
            self._selected = result.worst_pose
            self._review = result.worst_pose
        else:
            # A passing manual override collapses the panel like any other
            # pass (D12) -- the record itself still carries manual_override.
            self._reset_override_state()
        self._refresh()
        self._paint_review()

    def invalidate(self) -> None:
        """F5/D2: force a fresh calibration before Continue re-enables.

        Called from main.py's restore_cal when the wear that just ended
        either recorded a session or was abandoned via Restart -- either
        invalidates a per-wear calibration (D2). Reuses _recapture's reset
        so the two paths can't drift apart; a bare Back with no Start never
        reaches this method at all."""
        self._recapture()

    def _recapture(self) -> None:
        self._pose_state = {key: "pending" for key in POSE_KEYS}
        self._pose_counts = {}
        self._pose_frames = {}
        self._pose_dets = {}
        self._selected = POSE_KEYS[0]
        self._review = None
        self._status = "idle"
        self._result = None
        self._error_message = None
        self._overridden = False
        self._manual_override = False
        self._auto_params = None
        self._auto_worst_count = None
        self._reset_override_state()
        self._burst = []
        self._capturing = None
        self._params = load_params()
        self._refresh()

    def _continue(self) -> None:
        result = self._result
        if result is None or self._status not in ("pass", "fail"):
            return
        self._overridden = not result.passed
        roi_config = self._roi.config
        # F6a (D3): nothing is written here anymore -- calibration is held in
        # memory and only reaches disk once a session directory exists
        # (SessionWindow._confirm_start calls save_calibration with
        # dest_dir=session_dir). This bundle carries everything that call
        # needs; it must stay in sync with save_calibration's signature in
        # output/participants.py.
        pending_calibration = {
            "result": result,
            "pose_frames": dict(self._pose_frames),
            "overridden": self._overridden,
            "roi": (
                {"x": roi_config.x, "y": roi_config.y, "w": roi_config.w,
                 "h": roi_config.h, "alpha": roi_config.alpha}
                if roi_config is not None else None
            ),
            "calibration_image_size": list(self._cal.image_size),
            "manual_override": result.manual_override,
            "auto_params": result.auto_params,
        }
        self._on_continue(result, pending_calibration)

    # ── toast ───────────────────────────────────────────────────────────────
    def show_toast(self, message: str) -> None:
        """Public (E1): landing.restore_cal calls this from outside the class
        after a Session-window Restart hands back a post-navigation message.
        _show_toast alias kept below so every existing internal caller (this
        file's own retry/detect-failure paths) still works unchanged."""
        if self._toast_after is not None:
            self.after_cancel(self._toast_after)
        self._toast_label.configure(text=message)
        self._reset_toast_colors()
        self._toast.place(in_=self._toast_parent, relx=1.0, x=0, y=0, anchor="ne")
        self._toast.lift()
        self._toast_after = self.after(_TOAST_TIMEOUT_MS, lambda: self._fade_toast(1))

    _show_toast = show_toast  # E1: private alias for this file's own callers

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

    # ── camera + detection ──────────────────────────────────────────────────
    def _open_camera(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            cap.release()
            self._show_unavailable()
            return
        cal_w, cal_h = self._cal.image_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self._cap = cap
        self._unavailable.place_forget()
        self._count_badge.place(x=12, y=12)
        self._view_stack.place(relx=1.0, x=-12, y=12, anchor="ne")
        # Capture is gated on a live camera and the actions were first rendered
        # while _cap was still None, so they have to be rebuilt now.
        self._render_actions()
        self._poll()

    def _show_unavailable(self) -> None:
        self._cap = None
        self._count_badge.place_forget()
        self._view_stack.place_forget()
        self._unavailable.place(x=0, y=0, relwidth=1, relheight=1)
        self._unavailable.lift()
        self._render_actions()

    def _set_camera_disconnected(self) -> None:
        """Fires only after CAMERA_FAIL_POLLS consecutive failed reads.
        Never touches self._pose_frames/_pose_dets/_result/_status -- an
        already-captured pose must survive a disconnect untouched (D3). A
        burst in progress is discarded instead, since a median across frames
        from before and after a gap is not trustworthy (D2)."""
        self._after_id = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._disconnect_at = time.monotonic()
        self._read_fail_streak = 0

        if self._capturing is not None:
            self._pose_state[self._capturing] = "pending"
            self._burst = []
            self._capturing = None

        self._show_unavailable()
        self._refresh()

    def retry_camera(self) -> None:
        if self._cap is not None:
            return
        self._open_camera()
        if self._cap is None:
            self._show_toast("Camera still unavailable — check the USB connection.")
            return
        self._render_actions()
        self._read_fail_streak = 0
        if self._disconnect_at is not None:
            gap = round(time.monotonic() - self._disconnect_at)
            self._show_toast(f"Camera reconnected — {gap}s gap")
            self._disconnect_at = None

    def _processed_gray(self, frame: np.ndarray) -> np.ndarray:
        undistorted = self._tracker.undistort(frame)
        return cv2.cvtColor(self._roi.crop(undistorted), cv2.COLOR_BGR2GRAY)

    def _detect(self, gray: np.ndarray) -> list:
        params = dict(self._params, size_method="cc")
        proc = preprocess(gray, params)
        mask = self._roi.sensor_mask(proc.shape)
        if mask is not None:
            proc = cv2.bitwise_and(proc, mask)
        return detect(denoise(gray, params), proc, params)

    def _paint(self, gray: np.ndarray, dets: list, label: str, count: int) -> None:
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, _size in dets:
            cv2.circle(display, (int(round(x)), int(round(y))), 6, _RING_BGR, 2)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(_fit_letterbox(Image.fromarray(rgb), FEED_W, FEED_H))
        self._feed_label.configure(image=photo)
        self._current_img = photo
        self._count_label.configure(text=f"{count} / {TARGET_MARKERS} MARKERS")
        self._view_label.configure(text=label)
        if self._review is not None:
            self._live_btn.pack(anchor="e", pady=(6, 0))
        else:
            self._live_btn.pack_forget()

    def _paint_review(self) -> None:
        key = self._review
        if key is None or key not in self._pose_frames:
            return
        dets = self._pose_dets.get(key, [])
        self._paint(self._pose_frames[key], dets, f"POSE {POSE_KEYS.index(key) + 1}", len(dets))

    def _poll(self) -> None:
        cap = self._cap
        if cap is None:
            return
        ret, frame = cap.read()
        if ret:
            self._read_fail_streak = 0
            gray = self._processed_gray(frame)
            self._live_gray = gray
            if self._capturing is not None:
                self._burst.append(gray)
                self._update_burst()
                if len(self._burst) >= BASELINE_FRAMES:
                    self._finish_capture()
            else:
                self._frame_tick += 1
                if self._frame_tick % DETECT_EVERY == 0:
                    self._live_dets = self._detect(gray)
            if self._review is None:
                self._paint(gray, self._live_dets, "LIVE", len(self._live_dets))
        else:
            self._read_fail_streak += 1
            if self._read_fail_streak >= CAMERA_FAIL_POLLS:
                self._set_camera_disconnected()
                return
        self._after_id = self.after(POLL_MS, self._poll)

    def release_camera(self) -> None:
        for attr in ("_after_id", "_search_after", "_toast_after"):
            pending = getattr(self, attr)
            if pending is not None:
                self.after_cancel(pending)
                setattr(self, attr, None)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _handle_back(self) -> None:
        self.release_camera()
        self._on_back()

    def _on_close(self) -> None:
        self.release_camera()
        self.destroy()

    # ── button/icon idiom (copied from session_window so the windows stay
    #    independent) ────────────────────────────────────────────────────────
    def _build_button(
        self, parent: tk.Misc, label: str, kind: str, icon: str | None,
        command: Callable[[], None], enabled: bool = True,
    ) -> tk.Frame:
        if not enabled:
            fill, fg, border = theme.BG, theme.INK_45, theme.INK_45
        elif kind == "primary":
            fill, fg, border = theme.ACCENT, theme.BG, theme.ACCENT
        elif kind == "ghost":
            fill, fg, border = theme.BG, theme.ACCENT, theme.BG
        else:
            fill, fg, border = theme.BG, theme.INK, theme.DIVIDER

        btn = tk.Frame(
            parent, bg=fill, highlightthickness=1,
            highlightbackground=border, highlightcolor=border,
        )
        inner = tk.Frame(btn, bg=fill)
        inner.pack(padx=14, pady=8)

        widgets: list[tk.Widget] = [btn, inner]
        if icon in ("arrow-left", "dot"):
            ic = self._icon(inner, icon, fg, fill)
            ic.pack(side="left", padx=(0, 6))
            widgets.append(ic)
        text = tk.Label(inner, text=label, bg=fill, fg=fg, font=self._f_button)
        text.pack(side="left")
        widgets.append(text)
        if icon == "arrow-right":
            ic = self._icon(inner, icon, fg, fill)
            ic.pack(side="left", padx=(6, 0))
            widgets.append(ic)

        if enabled:
            if kind == "primary":
                hover, active = theme.ACCENT_600, theme.ACCENT_700
            elif kind == "ghost":
                hover, active = theme.ACCENT_10, theme.ACCENT_08
            else:
                hover, active = theme.HOVER, theme.ACTIVE
            self._wire_button(btn, widgets, fill, hover, active, command)
        return btn

    def _icon(self, parent: tk.Misc, kind: str, color: str, bg: str) -> tk.Canvas:
        c = tk.Canvas(parent, width=14, height=14, bg=bg, highlightthickness=0)
        s = 14 / 24

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        if kind == "arrow-right":
            c.create_line(*p(5, 12, 19, 12), fill=color, width=2, capstyle="round")
            c.create_line(*p(12, 5, 19, 12, 12, 19), fill=color, width=2,
                          capstyle="round", joinstyle="round")
        elif kind == "arrow-left":
            c.create_line(*p(19, 12, 5, 12), fill=color, width=2, capstyle="round")
            c.create_line(*p(12, 19, 5, 12, 12, 5), fill=color, width=2,
                          capstyle="round", joinstyle="round")
        elif kind == "dot":
            c.create_oval(*p(4, 4, 20, 20), fill=color, outline=color)
        return c

    def _wire_button(
        self, btn: tk.Frame, widgets: list[tk.Widget], base: str,
        hover: str, active: str, command: Callable[[], None],
    ) -> None:
        def paint(colour: str) -> None:
            for w in widgets:
                w.configure(bg=colour)

        def on_release(_e: object) -> None:
            x, y = btn.winfo_pointerxy()
            hovered = btn.winfo_containing(x, y)
            if hovered is not None and (hovered is btn or str(hovered).startswith(str(btn))):
                paint(hover)
                command()
            else:
                paint(base)

        for w in widgets:
            w.bind("<Enter>", lambda _e: paint(hover))
            w.bind("<Leave>", lambda _e: paint(base))
            w.bind("<ButtonPress-1>", lambda _e: paint(active))
            w.bind("<ButtonRelease-1>", on_release)
            w.configure(cursor="hand2")


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
