"""Session Recording window for the GripVT desktop app.

Opened from the New Participant window's "Save & Continue". Shows the live
USB feed (undistorted + ROI-cropped, same pipeline as the pre-baseline
viewer preview) alongside session/trial progress and target-force readouts.
Visual design ported from the Modernist design system (Claude Design:
"GripVT Session Recording").

Baseline capture, Kalman tracking, and CSV recording are not wired here --
only Tracker.undistort() + ROICrop.crop() are reused to render the live
preview. REC badge, progress bars, target force, and the Start/Stop/Next
Trial handlers are static placeholders for this pass.
"""
import tkinter as tk
from typing import Callable

import cv2
from PIL import Image, ImageTk

from core.calibration import CameraCalibration
from core.roi import DEFAULT_ALPHA, ROICrop
from core.tracker import Tracker
from ui import theme

WINDOW_W = 880
WINDOW_H = 740
FEED_W = 450
FEED_H = 600

CAMERA_INDEX = 1
POLL_MS = 33  # ~30 fps, matching viewer_window

_LETTERBOX_BG = tuple(int(theme.SURFACE.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


class SessionWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._on_back = on_back or (lambda: print("Back clicked"))

        self.title("GripVT — Session Recording")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._current_img: ImageTk.PhotoImage | None = None
        self._after_id: str | None = None
        self._cap: cv2.VideoCapture | None = None

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

        badge = tk.Frame(slot, bg=theme.INK)
        badge.place(x=12, y=12)
        badge_inner = tk.Frame(badge, bg=theme.INK)
        badge_inner.pack(padx=12, pady=6)
        dot = tk.Canvas(badge_inner, width=9, height=9, bg=theme.INK, highlightthickness=0)
        dot.create_oval(0, 0, 9, 9, fill=theme.ACCENT, outline=theme.ACCENT)
        dot.pack(side="left")
        tk.Label(
            badge_inner, text="REC 00:00", bg=theme.INK, fg=theme.BG, font=self._f_rec,
        ).pack(side="left", padx=(6, 0))

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
        self._progress_block(panel, "TRIAL", "12 / 30", 0.40, pady_top=16)

        tk.Frame(panel, bg=theme.DIVIDER, height=2).pack(fill="x", pady=18)

        target = tk.Frame(panel, bg=theme.BG)
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

        tk.Frame(panel, bg=theme.DIVIDER, height=2).pack(fill="x", pady=18)

        row = tk.Frame(panel, bg=theme.BG)
        row.pack(fill="x")
        row.columnconfigure(0, weight=1, uniform="ss")
        row.columnconfigure(1, weight=1, uniform="ss")
        start = self._build_button(row, "Start", "primary", "play", self._on_start)
        start.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        stop = self._build_button(row, "Stop", "secondary", "stop", self._on_stop, enabled=False)
        stop.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        next_btn = self._build_button(
            panel, "Next Trial", "secondary", "arrow-right", self._on_next_trial, enabled=False,
        )
        next_btn.pack(fill="x", pady=(12, 0))

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

    # ── button/icon idiom (copied from new_participant_window, not imported,
    #    so the windows stay independent) ──────────────────────────────────
    def _build_button(
        self, parent: tk.Frame, label: str, kind: str, icon: str,
        command: Callable[[], None], enabled: bool = True,
    ) -> tk.Frame:
        primary = kind == "primary"
        if not enabled:
            fill, fg, border = theme.BG, theme.INK_45, theme.INK_45
        elif primary:
            fill, fg, border = theme.ACCENT, theme.BG, theme.ACCENT
        else:
            fill, fg, border = theme.BG, theme.INK, theme.DIVIDER

        btn = tk.Frame(
            parent, bg=fill, highlightthickness=1,
            highlightbackground=border, highlightcolor=border,
        )
        inner = tk.Frame(btn, bg=fill)
        inner.pack(padx=14, pady=8)

        widgets: list[tk.Widget] = [btn, inner]
        if icon in ("arrow-left", "play", "stop"):
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
            hover, active = (theme.ACCENT_600, theme.ACCENT_700) if primary else (theme.HOVER, theme.ACTIVE)
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
        elif kind == "play":
            c.create_polygon(*p(6, 3, 20, 12, 6, 21), fill=color, outline=color)
        elif kind == "stop":
            c.create_rectangle(*p(5, 5, 19, 19), fill=color, outline=color)
        return c

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

    # ── stub handlers ───────────────────────────────────────────────────────
    def _on_start(self) -> None:
        print("Start clicked")

    def _on_stop(self) -> None:
        print("Stop clicked")

    def _on_next_trial(self) -> None:
        print("Next Trial clicked")

    # ── camera lifecycle ────────────────────────────────────────────────────
    def _poll(self) -> None:
        ret, frame = self._cap.read()
        if ret:
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
