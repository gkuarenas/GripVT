"""Main window for the GripVT viewer app.

Responsibilities:
  - Open camera, undistort, ROI crop, display live feed with marker overlay
  - Capture Baseline button — initialises Kalman states for all detected markers
  - Start/Stop Session — records marker displacement CSV + overlay PNG to output/sessions/
"""
import cv2
import customtkinter as ctk
from PIL import Image

from core.calibration import CameraCalibration
from core.roi import ROICrop

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps
VIDEO_W = 1280
VIDEO_H = 720


class ViewerWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GripVT Viewer")
        self.geometry(f"{VIDEO_W}x{VIDEO_H + 48}")
        self.resizable(False, False)

        self._alpha = 1.0
        self._cal = CameraCalibration(alpha=self._alpha)
        self._roi = ROICrop()
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")
        # Capture resolution must match the calibration's image_size, or the
        # undistort maps (built for that resolution) misalign with the frame.
        cal_w, cal_h = self._cal.image_size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)
        self._after_id: str | None = None

        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True)

        self._build_controls()

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
        self._alpha_seg.set("1  (full)")
        self._alpha_seg.pack(side="left", pady=8)

    def _on_alpha_change(self, value: str) -> None:
        self._alpha = 0.0 if value.startswith("0") else 1.0
        self._cal = CameraCalibration(alpha=self._alpha)

    def _poll(self) -> None:
        ret, frame = self._cap.read()
        if ret:
            frame = self._cal.undistort(frame)
            frame = self._roi.crop(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(VIDEO_W, VIDEO_H))
            self._label.configure(image=ctk_img)
            self._label.image = ctk_img  # keep reference alive
        self._after_id = self.after(POLL_MS, self._poll)

    def _on_close(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._cap.release()
        self.destroy()
