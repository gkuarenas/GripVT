"""Main window for the GripVT viewer app.

Responsibilities:
  - Open camera (16:9, raw, no undistortion)
  - Display live feed with marker overlay
  - Capture Baseline button — initialises Kalman states for all detected markers
  - Start/Stop Session — records marker displacement CSV + overlay PNG to output/sessions/
"""
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from core.roi import ROICrop

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps


class ViewerWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GripVT Viewer")
        self.geometry("1280x720")
        self.resizable(False, False)

        self._roi = ROICrop()
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._after_id: str | None = None

        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    def _poll(self) -> None:
        ret, frame = self._cap.read()
        if ret:
            frame = self._roi.crop(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((1280, 720), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._label.configure(image=photo)
            self._label.image = photo  # keep reference alive
        self._after_id = self.after(POLL_MS, self._poll)

    def _on_close(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._cap.release()
        self.destroy()
