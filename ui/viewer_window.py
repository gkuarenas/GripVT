"""Main window for the GripVT viewer app.

Responsibilities:
  - Open camera (16:9, raw, no undistortion)
  - Display live feed with marker overlay
  - Capture Baseline button — initialises Kalman states for all detected markers
  - Start/Stop Session — records marker displacement CSV + overlay PNG to output/sessions/
"""
import customtkinter as ctk

# TODO: implement viewer window


class ViewerWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GripVT Viewer")
        self.geometry("1280x720")
        # TODO: add camera feed, controls, and overlay canvas
