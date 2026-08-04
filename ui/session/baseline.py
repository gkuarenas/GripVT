"""BaselineMixin -- the Session window's Capture Baseline gate state machine.

Part of the ui/session/ package (see ui/session/window.py's module docstring
for the package overview). Split out of the former ui/session_window.py
verbatim (mechanical move, no logic changes) -- see
docs/logs/20260729_141951_codebase_reorg_and_session_split_plan.md, D8.

The right panel is a two-phase gate on Tracker.capture_baseline(): phase 1
(gate box) blocks Start until a real 15-frame burst locks >= MIN_ACCEPTABLE
of the 99 Kalman states; phase 2 (baseline strip) shows the locked count and
unlocks Start. A below-threshold capture can be overridden (flags the saved
record) or recaptured -- but Re-capture is permanently disabled once Start is
pressed, since Kalman states are locked once per session (see CLAUDE.md).

_refresh_transport reads/writes state owned by the progress, recording, and
scaffold groups -- left as plain `self.` calls per D1/D2 of the split plan,
not decoupled.
"""
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable

import numpy as np

from core.detector_search import (
    GRID_SHAPE, MIN_ACCEPTABLE, TARGET_MARKERS, SearchResult, region_counts,
)
from core.roi import ROICrop
from core.tracker import BASELINE_FRAMES, Tracker
from output.participants import ParticipantRecord
from ui import theme

_BASELINE_IDLE: dict = {
    "status": "none", "frame": 0, "count": None,
    "overridden": False, "corners": None, "bands": None, "subline": "",
}


class BaselineMixin:
    """Baseline-capture gate. Composed into SessionWindow alongside the other
    ui/session/ mixins -- see window.py for the single __init__ that owns all
    of the attributes declared below (D1/D3 of the split plan)."""

    # ── Cross-group attributes (D4): owned/assigned in window.py's __init__
    # (or, for a few, written by another mixin) -- declared here only so
    # mypy resolves self.<attr> access within this mixin. No runtime effect.
    _baseline: dict
    _baseline_frames: list[np.ndarray]
    _baseline_clip_frames: list[np.ndarray]
    _pending_baseline: dict | None
    _pending_grid_layout: dict | None
    _phase1_frame: tk.Frame
    _phase2_frame: tk.Frame
    _recording: bool
    _recording_started: bool
    _trial: int
    _trial_max: int
    _grip_marked: bool
    _start_btn: tk.Frame
    _stop_btn: tk.Frame
    _next_btn: tk.Frame
    _grip_btn: tk.Frame
    _restart_btn: tk.Frame
    _session_num: int
    _gate_box: tk.Frame
    _strip: tk.Frame
    _strip_recapture_btn: tk.Frame | None
    _capture_frame_label: tk.Label | None
    _capture_fill: tk.Frame | None
    _calibration_result: SearchResult | None
    _participant: ParticipantRecord | None
    _tracker: Tracker
    _roi: ROICrop
    _f_kicker: tkfont.Font
    _f_gate_heading: tkfont.Font
    _f_gate_body: tkfont.Font
    _f_gate_value: tkfont.Font
    _f_gate_fail_heading: tkfont.Font
    _f_gate_count: tkfont.Font
    _f_gate_count_unit: tkfont.Font
    _f_gate_footnote: tkfont.Font
    _f_strip_label: tkfont.Font
    _f_strip_tag: tkfont.Font
    _f_strip_subline: tkfont.Font

    # Cross-group methods (D2): plain self.method() calls into the scaffold
    # group (window.py), declared the same way for mypy.
    _crosshair_icon: Callable[..., tk.Canvas]
    _warning_icon: Callable[..., tk.Canvas]
    _check_icon_square: Callable[..., tk.Frame]
    _build_button: Callable[..., tk.Frame]

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
        # F2: one grip mark per segment -- greys out once pressed, returns
        # when Next Trial or Undo opens a fresh segment. Deliberately NOT a
        # precondition of anything else (D4).
        grip_disabled = not locked or not self._recording or self._grip_marked
        self._start_btn.set_enabled(not start_disabled)
        self._stop_btn.set_enabled(not stop_disabled)
        self._next_btn.set_enabled(not next_trial_disabled)
        self._grip_btn.set_enabled(not grip_disabled)
        # D7: flips on the instant the first Start of a window is confirmed.
        self._restart_btn.set_enabled(self._session_num >= 1)

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
        self._baseline_clip_frames = []
        self._baseline = {
            "status": "capturing", "frame": 0, "count": None,
            "overridden": False, "corners": None, "bands": None, "subline": "",
        }
        self._render_baseline()

    def _on_recapture_baseline_click(self) -> None:
        if self._recording_started:
            return
        self._baseline_frames = []
        self._baseline_clip_frames = []
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
        self._baseline_clip_frames = self._baseline_frames
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
        """F6b (D3): buffers the save_baseline() kwargs bundle instead of
        writing -- _confirm_start (recording.py) flushes it into the session
        directory once one exists. Called on a pass, or on an operator
        override of a below-threshold capture; either overwrites whatever was
        previously buffered, so a fail -> recapture -> pass sequence (or a
        pass -> recapture -> pass one) only ever leaves the winning attempt
        buffered -- the same guarantee the .mp4 path already gave when this
        wrote straight to disk."""
        if self._participant is None:
            return
        b = self._baseline
        self._pending_baseline = {
            "locked_count": b["count"],
            "passed": b["status"] == "pass",
            "overridden": b["overridden"],
            "corners": b["corners"],
            "bands": b["bands"],
            "params": self._tracker.params,
            "frames": self._baseline_clip_frames,
        }
        # Phase 3 (marker-grid-lookup plan): buffered from the tracker's
        # already-computed grid_layout* attributes (set by capture_baseline())
        # -- same overwrite-on-recapture guarantee as _pending_baseline above,
        # since both are (re)built from the same winning attempt in this one
        # method call. grid_shape is a fixed config constant (GRID_SHAPE),
        # not something capture_baseline() derives per attempt, so it's
        # correct to include even on the fallback_sort path.
        self._pending_grid_layout = {
            "layout": self._tracker.grid_layout,
            "source": self._tracker.grid_layout_source,
            "reason": self._tracker.grid_layout_reason,
            "locked_count": b["count"],
            "grid_shape": {"n_rows": GRID_SHAPE[1], "n_cols": GRID_SHAPE[0]},
        }
