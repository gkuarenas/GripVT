"""CameraFeedMixin -- camera polling, feed display, and the {n}/99 TRACKED
indicator for the Session window.

Part of the ui/session/ package (see ui/session/window.py's module docstring
for the package overview). Split out of the former ui/session_window.py
verbatim (mechanical move, no logic changes) -- see
docs/logs/20260729_141951_codebase_reorg_and_session_split_plan.md, D8.
"""
import logging
import time
import tkinter as tk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.calibration import CameraCalibration
from core.detector_search import TARGET_MARKERS
from core.profiling import PROFILE, mark_period, maybe_report_every, report, stage
from core.roi import ROICrop
from core.tracker import MarkerRecord, Tracker
from output.video import VideoLogWriter
from output.writer import CSVWriter, EventLog, now_ms
from ui import theme

_log = logging.getLogger(__name__)

FEED_W = 450
FEED_H = 600

CAMERA_INDEX = 1
# Target period / rate ceiling (D3, docs/logs/20260731_152249_framerate_spike_
# plan.md) -- NOT a trailing sleep added on top of the work. _poll() computes
# whatever budget is left of this after the frame's real work and schedules
# only that; when the work exceeds POLL_MS, the loop runs as fast as the work
# allows instead of the true period being work + POLL_MS.
POLL_MS = 33
# D2: floor for the remaining-budget computation below -- never schedule
# after(0). The Tk event loop still needs a turn to process button presses,
# the REC timer, and modals even when the frame work eats the whole budget;
# a busy loop that starves the UI is worse than a slow one.
MIN_POLL_MS = 1

# A single dropped read at 30fps is normal jitter, not a disconnection --
# only a sustained streak means the USB device actually went away (E9).
CAMERA_FAIL_POLLS = 15  # ~0.5s at POLL_MS

# D7 (framerate spike plan): the PIL/Tk repaint (cvtColor -> Image.fromarray
# -> letterbox resize -> PhotoImage -> configure) costs real time but a human
# eye can't tell a 30Hz repaint from a 15Hz one -- skipping every other poll
# halves that cost while detection/tracking still runs on every poll (D6).
DISPLAY_EVERY = 2

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

_LETTERBOX_BG = tuple(int(theme.SURFACE.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

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


class CameraFeedMixin:
    """Camera lifecycle + tracked-marker indicator. Composed into
    SessionWindow alongside the other ui/session/ mixins -- see window.py
    for the single __init__ that owns all of the attributes declared below
    (D1/D3)."""

    # ── Cross-group attributes (D4): owned/assigned in window.py's __init__
    # (or, for _last_records, written by this mixin and read elsewhere) --
    # declared here only so mypy resolves self.<attr> access within this
    # mixin. No runtime effect.
    _cal: CameraCalibration
    _cap: cv2.VideoCapture | None
    _read_fail_streak: int
    _baseline: dict
    _feed_label: tk.Label
    _current_img: ImageTk.PhotoImage | None
    _after_id: str | None
    _disconnect_at: float
    _disconnected_panel: tk.Frame
    _badge_container: tk.Frame
    _recording: bool
    _paused_banner: tk.Frame
    _baseline_container: tk.Frame
    _event_log: EventLog | None
    _t0_wall_ms: float
    _t0_monotonic: float
    _open_camera_loss_event: dict | None
    _tracking_disabled: bool
    _frame_tick: int
    _display_tick: int
    _last_records: list[MarkerRecord] | None
    _writer: CSVWriter | None
    _video: VideoLogWriter | None
    _tracked_label: tk.Label
    _tracked_tier: str | None
    _tracked_badge: tk.Frame
    _tracked_inner: tk.Frame
    _tracked_dot: tk.Canvas
    _tracked_dot_id: int
    _tracked_blink_state: bool
    _tracked_blink_after: str | None
    _toast_after: str | None
    _rec_after_id: str | None
    _tracker: Tracker
    _roi: ROICrop

    # Cross-group methods (D2): plain self.method() calls into the baseline,
    # recording, and modal groups, declared the same way for mypy.
    _capture_baseline_frame: Callable[..., None]
    _pause_timer: Callable[[], None]
    _resume_timer: Callable[[], None]
    _show_rec_badge: Callable[[], None]
    _show_rec_paused_badge: Callable[[], None]
    _show_toast: Callable[[str], None]
    _finalize_recording: Callable[..., None]
    _hide_rec_badges: Callable[[], None]
    _stop_timer: Callable[[], None]
    _buffer_recording_frame: Callable[..., None]
    _refresh_transport: Callable[[], None]

    # This mixin is composed onto tk.Toplevel (window.py's class statement),
    # not declared here -- these two are inherited, not cross-group per se,
    # but need the same mypy-only declaration since CameraFeedMixin itself
    # has no base class.
    after: Callable[..., str]
    after_cancel: Callable[..., None]

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
        # Phase 2 (framerate spike plan): the real work's duration is what's
        # left over from POLL_MS's budget, computed below instead of always
        # sleeping the full POLL_MS on top of it (D3).
        t_start = time.perf_counter()
        # Phase 1 (framerate spike, see docs/logs/20260731_152249_
        # framerate_spike_plan.md): mark_period() is the real achieved-fps
        # measure, and the stage() wraps below are pure instrumentation --
        # both are no-ops unless GRIPVT_PROFILE=1, so this changes nothing
        # about the loop's behaviour or timing.
        mark_period()
        with stage("poll_total"):
            with stage("read"):
                ret, frame = self._cap.read()
            if ret:
                self._read_fail_streak = 0
                if self._video is not None:
                    ts = now_ms(self._t0_wall_ms, self._t0_monotonic)
                    with stage("video_write"):
                        self._video.write(frame, ts)  # raw, pre-undistortion (D7)
                if self._baseline["status"] == "capturing":
                    self._capture_baseline_frame(frame)
                # Phase 3 (framerate spike plan, D4/D5): undistort the full
                # frame once and hand the same array to both the display crop
                # and the tracker's math -- previously Tracker.process_frame
                # undistorted this identical frame a second time internally.
                with stage("undistort_display"):
                    undistorted = self._tracker.undistort(frame)
                    display = self._roi.crop(undistorted)
                # Phase 4 (framerate spike plan, D6/D7): before the baseline
                # is locked the operator needs a smooth feed to position the
                # sleeve (and the loop is cheap then anyway), so repaint every
                # poll. Once locked, throttle to every DISPLAY_EVERY polls --
                # a count badge/feed doesn't need to redraw faster than a
                # human eye can tell.
                self._display_tick += 1
                should_repaint = (
                    not self._tracker.baseline_set
                    or self._display_tick % DISPLAY_EVERY == 0
                )
                if should_repaint:
                    with stage("display_paint"):
                        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                        img = _fit_letterbox(Image.fromarray(rgb), FEED_W, FEED_H)
                        photo = ImageTk.PhotoImage(img)
                        self._feed_label.configure(image=photo)
                        self._current_img = photo  # plain tk.Label drops the image without this ref
                with stage("track"):
                    self._update_tracked_indicator(frame, undistorted)
                maybe_report_every(100)
            else:
                # E9: a lone dropped read is normal 30fps jitter -- only a
                # sustained streak means the camera actually disconnected.
                self._read_fail_streak += 1
                if self._read_fail_streak >= CAMERA_FAIL_POLLS:
                    self._set_camera_disconnected()
                    return  # disconnected state owns rescheduling from here (none, until retry)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            remaining_ms = max(MIN_POLL_MS, POLL_MS - elapsed_ms)
            self._after_id = self.after(int(remaining_ms), self._poll)

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
    def _update_tracked_indicator(
        self, raw_frame: np.ndarray, undistorted: np.ndarray | None = None,
    ) -> None:
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

        undistorted (Phase 3, framerate spike plan D4): _poll() already
        undistorted raw_frame once for the display crop -- pass that same
        array through to Tracker.process_frame() so it isn't undistorted a
        second time. None (the default) preserves today's behaviour, e.g.
        for any other caller that hasn't already undistorted the frame.
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
            records = self._tracker.process_frame(raw_frame, undistorted)
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
        # Phase 1 (framerate spike): print the final table even if the
        # session ended before hitting a 100-frame maybe_report_every boundary.
        if PROFILE:
            print(report())


def _fit_letterbox(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (box_w, box_h), _LETTERBOX_BG)
    canvas.paste(resized, ((box_w - new_w) // 2, (box_h - new_h) // 2))
    return canvas
