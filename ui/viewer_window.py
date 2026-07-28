"""Main window for the GripVT viewer app.

Responsibilities:
  - Open camera, undistort, ROI crop, display live feed with marker overlay
  - Capture Baseline button — initialises Kalman states for all detected markers
  - Start/Stop Session — records marker displacement CSV + overlay PNG to output/sessions/
"""
import time

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from core.calibration import CameraCalibration
from core.roi import DEFAULT_ALPHA, ROICrop
from core.tracker import BASELINE_FRAMES, MarkerRecord, Tracker
from output.writer import CSVWriter, make_session_dir
from ui.overlay import draw_overlay

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps
# Camera captures natively at 4:3 (1024x768); the live feed is rotated 90 degrees
# clockwise for display, so the displayed window is 3:4 portrait — swapped, not stretched.
VIDEO_W = 768
VIDEO_H = 1024

_HUD_ORIGIN = (8, 20)
_HUD_COLOR = (0, 255, 200)


class ViewerWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GripVT Viewer")
        self.geometry(f"{VIDEO_W}x{VIDEO_H + 48}")
        self.resizable(True, True)
        # Without this, the label's requested size (driven by the displayed
        # image) grows the window, which fires another <Configure> event with
        # a bigger box, which grows the image again -- runaway feedback loop.
        self.pack_propagate(False)
        self._display_size = (VIDEO_W, VIDEO_H)

        self._roi = ROICrop()
        # Undistort with whatever alpha the ROI was selected under (see
        # tools/select_roi.py) -- a mismatch here remaps pixel positions and
        # misaligns the crop with the marker grid.
        roi_config = self._roi.config
        self._alpha = roi_config.alpha if roi_config is not None else DEFAULT_ALPHA
        self._cal = CameraCalibration(alpha=self._alpha)
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")
        # Capture resolution must match the calibration's image_size, or the
        # undistort maps (built for that resolution) misalign with the frame.
        cal_w, cal_h = self._cal.image_size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self._after_id: str | None = None

        # Tracker shares this window's CameraCalibration instance so its
        # internal undistort (which also rotates -- see Tracker.undistort())
        # and the pre-baseline raw preview always agree on the same pixel
        # mapping -- see _on_alpha_change for why the tracker is rebuilt
        # (not just self._cal) whenever alpha changes.
        self._tracker = Tracker(self._cal, self._roi)

        self._baseline_capturing = False
        self._baseline_frames: list[np.ndarray] = []

        self._session_active = False
        self._session_writer: CSVWriter | None = None
        self._session_t0: float | None = None
        self._last_overlay_frame: np.ndarray | None = None

        # Control bar must be packed before the label -- with pack_propagate
        # disabled, the packer allocates a fixed cavity to slaves in pack
        # order, and a fill="both"/expand=True slave packed first claims the
        # entire cavity, leaving nothing for a fixed-height slave packed after.
        self._build_controls()

        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True)
        self._label.bind("<Configure>", self._on_label_resize)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    def _build_controls(self) -> None:
        bar = ctk.CTkFrame(self, height=48, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="Undistort alpha:").pack(side="left", padx=(12, 6), pady=8)
        self._alpha_seg = ctk.CTkSegmentedButton(
            bar,
            values=["0  (crop)", "1  (full)"],
            command=self._on_alpha_change,
        )
        self._alpha_seg.set("0  (crop)" if self._alpha == 0.0 else "1  (full)")
        self._alpha_seg.pack(side="left", pady=8)

        self._capture_baseline_btn = ctk.CTkButton(
            bar, text="Capture Baseline", command=self._on_capture_baseline_click,
        )
        self._capture_baseline_btn.pack(side="left", padx=(12, 6), pady=8)

        self._session_btn = ctk.CTkButton(
            bar, text="Start Session", state="disabled", command=self._on_session_toggle_click,
        )
        self._session_btn.pack(side="left", padx=(6, 12), pady=8)

    def _on_alpha_change(self, value: str) -> None:
        self._alpha = 0.0 if value.startswith("0") else 1.0
        self._cal = CameraCalibration(alpha=self._alpha)
        # Baseline is never active while the alpha control is enabled (it is
        # disabled the moment a capture starts -- see
        # _on_capture_baseline_click), so rebuilding the tracker here never
        # discards locked Kalman states.
        self._tracker = Tracker(self._cal, self._roi)
        roi_config = self._roi.config
        if roi_config is not None and self._alpha != roi_config.alpha:
            print(
                f"Warning: viewer alpha={self._alpha} no longer matches "
                f"roi.json alpha={roi_config.alpha} -- the ROI crop will "
                "be misaligned until you re-run select_roi.py at this alpha."
            )

    def _on_capture_baseline_click(self) -> None:
        if self._baseline_capturing or self._session_active:
            return
        self._baseline_frames = []
        self._baseline_capturing = True
        self._capture_baseline_btn.configure(state="disabled")
        # Locking alpha here (not just after the burst completes) keeps the
        # tracker's undistort mapping fixed for every frame in the burst --
        # changing it mid-capture would mix pixel spaces within one baseline.
        self._alpha_seg.configure(state="disabled")

    def _finish_baseline_capture(self) -> None:
        count = self._tracker.capture_baseline(self._baseline_frames)
        self._baseline_frames = []
        self._baseline_capturing = False
        self._capture_baseline_btn.configure(text="Re-capture Baseline", state="normal")
        self._session_btn.configure(state="normal")
        print(f"Baseline captured: {count} markers locked")

    def _on_session_toggle_click(self) -> None:
        if self._baseline_capturing:
            return
        if self._session_active:
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self) -> None:
        session_dir = make_session_dir("recordings/_viewer_scratch")
        self._session_writer = CSVWriter(session_dir)
        self._session_t0 = time.monotonic()
        self._session_active = True
        self._session_btn.configure(text="Stop Session")
        self._capture_baseline_btn.configure(state="disabled")

    def _stop_session(self) -> None:
        writer = self._session_writer
        assert writer is not None
        writer.write_window()
        if self._last_overlay_frame is not None:
            cv2.imwrite(writer.png_path, self._last_overlay_frame)
        writer.close()
        self._session_writer = None
        self._session_t0 = None
        self._session_active = False
        self._session_btn.configure(text="Start Session")
        self._capture_baseline_btn.configure(state="normal")

    def _draw_status(self, frame: np.ndarray, text: str) -> None:
        cv2.putText(frame, text, _HUD_ORIGIN, cv2.FONT_HERSHEY_PLAIN, 1.0, _HUD_COLOR, 1)

    def _render_raw(self, raw_frame: np.ndarray) -> np.ndarray:
        # Routed through self._tracker.undistort() (not self._cal.undistort())
        # so rotation is applied consistently in one place -- see
        # Tracker.undistort().
        frame = self._tracker.undistort(raw_frame)
        frame = self._roi.crop(frame)
        self._draw_status(frame, "No baseline captured -- press Capture Baseline")
        return frame

    def _render_baseline_capture(self, raw_frame: np.ndarray) -> np.ndarray:
        self._baseline_frames.append(raw_frame)
        progress = min(len(self._baseline_frames), BASELINE_FRAMES)
        if len(self._baseline_frames) >= BASELINE_FRAMES:
            self._finish_baseline_capture()

        frame = self._tracker.undistort(raw_frame)
        frame = self._roi.crop(frame)
        self._draw_status(frame, f"Capturing baseline... {progress}/{BASELINE_FRAMES}")
        return frame

    def _render_tracked(self, raw_frame: np.ndarray) -> np.ndarray:
        records: list[MarkerRecord] = self._tracker.process_frame(raw_frame)
        full = self._tracker.last_undistorted
        assert full is not None
        annotated = draw_overlay(full, records, self._session_active, self._tracker.frame_index)
        frame = self._roi.crop(annotated)

        state_str = "RECORDING" if self._session_active else "TRACKING"
        self._draw_status(frame, f"markers {len(records)}  [{state_str}]")

        if self._session_active:
            writer = self._session_writer
            t0 = self._session_t0
            assert writer is not None and t0 is not None
            timestamp_ms = (time.monotonic() - t0) * 1000.0
            writer.buffer_frame(records, self._tracker.frame_index, timestamp_ms)
            self._last_overlay_frame = frame

        return frame

    def _on_label_resize(self, event) -> None:
        # Fit the 3:4 feed inside the label's current box without stretching it —
        # the box itself isn't locked to 3:4 since the control bar eats a fixed
        # 48px, so whichever dimension is the tighter constraint wins.
        w, h = event.width, event.height
        if w <= 0 or h <= 0:
            return
        if w / h > VIDEO_W / VIDEO_H:
            new_h = h
            new_w = int(h * VIDEO_W / VIDEO_H)
        else:
            new_w = w
            new_h = int(w * VIDEO_H / VIDEO_W)
        self._display_size = (max(new_w, 1), max(new_h, 1))

    def _poll(self) -> None:
        ret, frame = self._cap.read()
        if ret:
            if self._baseline_capturing:
                display = self._render_baseline_capture(frame)
            elif self._tracker.baseline_set:
                display = self._render_tracked(frame)
            else:
                display = self._render_raw(frame)

            frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=self._display_size)
            self._label.configure(image=ctk_img)
            self._label.image = ctk_img  # keep reference alive
        self._after_id = self.after(POLL_MS, self._poll)

    def _on_close(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        if self._session_active:
            self._stop_session()
        self._cap.release()
        self.destroy()
