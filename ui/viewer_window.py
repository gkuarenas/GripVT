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
from core.roi import DEFAULT_ALPHA, ROICrop

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps
# Camera captures natively at 4:3 (1024x768); the live feed is rotated 90 degrees
# clockwise for display, so the displayed window is 3:4 portrait — swapped, not stretched.
VIDEO_W = 768
VIDEO_H = 1024


class ViewerWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GripVT Viewer")
        self.geometry(f"{VIDEO_W}x{VIDEO_H + 48}")
        self.resizable(True, True)
        self._display_size = (VIDEO_W, VIDEO_H)

        self._roi = ROICrop()
        # Undistort with whatever alpha the ROI was selected under (see
        # tools/select_roi.py) -- a mismatch here remaps pixel positions and
        # misaligns the crop with the marker grid.
        self._alpha = self._roi.config.alpha if self._roi.is_set else DEFAULT_ALPHA
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

        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True)
        self._label.bind("<Configure>", self._on_label_resize)

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
        self._alpha_seg.set("0  (crop)" if self._alpha == 0.0 else "1  (full)")
        self._alpha_seg.pack(side="left", pady=8)

    def _on_alpha_change(self, value: str) -> None:
        self._alpha = 0.0 if value.startswith("0") else 1.0
        self._cal = CameraCalibration(alpha=self._alpha)
        if self._roi.is_set and self._alpha != self._roi.config.alpha:
            print(
                f"Warning: viewer alpha={self._alpha} no longer matches "
                f"roi.json alpha={self._roi.config.alpha} -- the ROI crop will "
                "be misaligned until you re-run select_roi.py at this alpha."
            )

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
            frame = self._cal.undistort(frame)
            frame = self._roi.crop(frame)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=self._display_size)
            self._label.configure(image=ctk_img)
            self._label.image = ctk_img  # keep reference alive
        self._after_id = self.after(POLL_MS, self._poll)

    def _on_close(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._cap.release()
        self.destroy()
