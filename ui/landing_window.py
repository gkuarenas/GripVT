"""Landing window for the GripVT desktop app.

First screen shown when the app is launched. Asks whether the operator is
setting up a **new** participant or resuming an **existing** one, then routes
to the appropriate flow (wired later -- for now the two buttons call the
``on_new_participant`` / ``on_existing_participant`` hooks, defaulting to a
print stub).

Visual design ported from the "Modernist" design system (Claude Design:
"GripVT Landing Window"). It is a flat aesthetic -- 0px radius, Archivo type,
warm off-white surface, single hot red-orange accent. Because plain tkinter
themes native buttons, the two choices are built as composite frames
(border + canvas icon + label) so the flat look is fully under our control.
"""
import ctypes
import sys
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from typing import Callable

# ── Modernist tokens ────────────────────────────────────────────────────────
# tkinter has no alpha, so every colour that the design expresses as a
# color-mix over a known background is pre-composited to a solid hex here.
INK = "#201e1d"          # --color-text  (body text)
BG = "#f3f2f2"           # --color-bg    (window body)
SUBTITLE = "#5f5e5d"     # text @ 70% over bg (subheading)
DIVIDER = "#9f9d9d"      # text @ 40% over bg (hr + button border)
HOVER = "#e4e3e3"        # text @ 7%  over bg (secondary btn hover)
ACTIVE = "#d5d4d4"       # text @ 14% over bg (secondary btn active)

WINDOW_W = 600
WINDOW_H = 300

# Archivo is the design's typeface, shipped with the app (not assumed installed)
# as the static weights under ui/fonts/Archivo and registered at startup.
_FONT_DIR = Path(__file__).resolve().parent / "fonts" / "Archivo"
_fonts_registered = False


def _register_archivo() -> None:
    """Load the bundled Archivo TTFs into this process so Tk can use them.

    Windows-only: AddFontResourceEx with FR_PRIVATE makes the fonts visible to
    the process without a system install. Idempotent across LandingWindow
    instances via the module-level guard; a no-op on macOS/Linux, where the
    code falls back to the platform default sans.
    """
    global _fonts_registered
    if _fonts_registered or sys.platform != "win32":
        return
    try:
        FR_PRIVATE = 0x10
        for ttf in _FONT_DIR.glob("*.ttf"):
            ctypes.windll.gdi32.AddFontResourceExW(str(ttf), FR_PRIVATE, 0)
        _fonts_registered = True
    except Exception:
        pass  # fall back to the default sans in _init_fonts


class LandingWindow(tk.Tk):
    def __init__(
        self,
        on_new_participant: Callable[[], None] | None = None,
        on_existing_participant: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_new = on_new_participant or (lambda: print("New Participant clicked"))
        self._on_existing = on_existing_participant or (
            lambda: print("Existing Participant clicked")
        )

        self.title("GripVT")
        self.configure(bg=BG)
        # Fixed-size, non-resizable: the design is a precise 600x300 canvas and
        # the space-between layout is tuned to that height, not a fluid box.
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._init_fonts()
        self._build_body()
        self._apply_dark_titlebar()

    # ── window helpers ──────────────────────────────────────────────────────
    def _center(self, w: int, h: int) -> None:
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3  # a touch above dead-centre
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _apply_dark_titlebar(self) -> None:
        # Paint the native OS title bar dark to match the flat dark-ink design.
        # tkinter exposes no option for this; on Windows it's a DWM attribute
        # set through ctypes. No-op (and harmless) on macOS/Linux.
        if sys.platform != "win32":
            return
        try:
            # The title bar belongs to the OS frame *around* Tk's drawing
            # surface, so step up from winfo_id() to its parent HWND.
            self.update_idletasks()  # ensure the native window exists first
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            enabled = ctypes.c_int(1)
            # Attribute is 20 on Windows 10 20H1+ / Windows 11, 19 on older
            # builds; try the modern value first, fall back to the legacy one.
            for attr in (20, 19):
                hresult = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(enabled), ctypes.sizeof(enabled)
                )
                if hresult == 0:  # S_OK
                    break
            # The frame won't repaint with the new colour until it's redrawn;
            # a withdraw/deiconify cycle forces that without moving the window.
            self.withdraw()
            self.deiconify()
        except Exception:
            pass  # DWM unavailable: fall back to the default light bar

    def _init_fonts(self) -> None:
        _register_archivo()
        families = set(tkfont.families(self))
        # The static Archivo build registers each cut as its own family; Tk sees
        # "Archivo" (Regular 400 / Bold 700), "Archivo Medium", "Archivo
        # SemiBold", "Archivo ExtraBold". If Archivo is missing (non-Windows /
        # load failed), fall back to the platform default sans so the window
        # still reads correctly.
        default = tkfont.nametofont("TkDefaultFont").actual("family")

        def pick(*names: str) -> str:
            # First bundled family that Tk sees, else the platform default.
            return next((n for n in names if n in families), default)

        # Design weights: headings 800 (Archivo ExtraBold, falling back to Bold),
        # subhead 600 (Archivo SemiBold). Each static cut is a distinct family,
        # so weight stays "normal" and the family name carries the weight; the
        # lone exception is plain "Archivo", where Bold needs weight="bold".
        heading = pick("Archivo ExtraBold", "Archivo")
        heading_weight = "bold" if heading == "Archivo" else "normal"
        semibold = pick("Archivo SemiBold")
        # Negative sizes force pixels (px), not points.
        self._font_title = tkfont.Font(family=heading, size=-45, weight=heading_weight)
        self._font_subtitle = tkfont.Font(family=semibold, size=-22, weight="normal")
        self._font_button = tkfont.Font(family=heading, size=-16, weight=heading_weight)

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        # Design padding: 26px top, 40px sides, 30px bottom.
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=40, pady=(15, 30))

        # Header block (packed top) and button row (packed bottom) reproduce the
        # design's justify-content: space-between within the body.
        header = tk.Frame(body, bg=BG)
        header.pack(side="top", fill="x")

        tk.Label(
            header, text="GripVT", bg=BG, fg=INK, font=self._font_title, anchor="w"
        ).pack(fill="x")
        tk.Label(
            header,
            text="Are you a new or existing participant?",
            bg=BG,
            fg=SUBTITLE,
            font=self._font_subtitle,
            anchor="w",
        ).pack(fill="x", pady=(1, 0))
        # 2px divider rule.
        tk.Frame(header, bg=DIVIDER, height=2).pack(fill="x", pady=(16, 0))

        row = tk.Frame(body, bg=BG)
        row.pack(side="bottom", fill="x")
        row.columnconfigure(0, weight=1, uniform="btn")
        row.columnconfigure(1, weight=1, uniform="btn")
        self._build_choice(row, 0, "new", "New Participant", self._on_new)
        self._build_choice(row, 1, "existing", "Existing Participant", self._on_existing)

    def _build_choice(
        self, parent: tk.Frame, col: int, icon: str, label: str, command: Callable[[], None]
    ) -> None:
        # Composite "button": a bordered frame holding an icon canvas + label,
        # left-aligned, 76px tall. Built by hand because native tk/ttk buttons
        # can't be flattened to the Modernist look.
        btn = tk.Frame(
            parent, bg=BG, height=76,
            highlightbackground=DIVIDER, highlightcolor=DIVIDER, highlightthickness=1,
        )
        btn.grid(row=0, column=col, sticky="ew", padx=(0, 8) if col == 0 else (8, 0))
        btn.grid_propagate(False)

        inner = tk.Frame(btn, bg=BG)
        inner.place(relx=0.0, rely=0.5, anchor="w", x=20)  # padding-left 20, v-centre

        icon_canvas = tk.Canvas(inner, width=25, height=25, bg=BG, highlightthickness=0)
        icon_canvas.pack(side="left")
        self._draw_icon(icon_canvas, icon)
        text = tk.Label(inner, text=label, bg=BG, fg=INK, font=self._font_button)
        text.pack(side="left", padx=(14, 0))  # 14px icon-to-text gap

        # Every widget that shows BG must recolour together on hover, or the
        # canvas/label would punch opaque holes in the fill (no tk alpha).
        tinted = [btn, inner, icon_canvas, text]
        self._wire_hover(btn, tinted, command)

    def _wire_hover(
        self, btn: tk.Frame, tinted: list[tk.Widget], command: Callable[[], None]
    ) -> None:
        def paint(colour: str) -> None:
            for w in tinted:
                w.configure(bg=colour)

        def on_enter(_e: object) -> None:
            paint(HOVER)

        def on_leave(_e: object) -> None:
            paint(BG)

        def on_press(_e: object) -> None:
            paint(ACTIVE)

        def on_release(_e: object) -> None:
            # Only fire if the pointer is still over the button (click, not drag-off).
            x, y = btn.winfo_pointerxy()
            hovered = btn.winfo_containing(x, y)
            if hovered is not None and (hovered is btn or str(hovered).startswith(str(btn))):
                paint(HOVER)
                command()
            else:
                paint(BG)

        # Bind to the button and every child so the hover/click region is the
        # whole tile, not just the strip of frame not covered by children.
        for w in (btn, *tinted):
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<ButtonPress-1>", on_press)
            w.bind("<ButtonRelease-1>", on_release)
            w.configure(cursor="hand2")

    # ── icons (lucide user-plus / user-check, stroked) ──────────────────────
    def _draw_icon(self, c: tk.Canvas, kind: str) -> None:
        s = 25 / 24  # design icons are authored on a 24x24 grid, drawn at 25px

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        # Head: circle cx9 cy7 r4.
        c.create_oval(*p(5, 3, 13, 11), outline=INK, width=2)
        # Shoulders: rounded path, approximated by a smooth open polyline.
        c.create_line(
            *p(16, 21, 16, 19, 12, 15, 6, 15, 2, 19, 2, 21),
            fill=INK, width=2, smooth=True, capstyle="round", joinstyle="round",
        )
        if kind == "new":
            # user-plus: vertical + horizontal stroke.
            c.create_line(*p(19, 8, 19, 14), fill=INK, width=2, capstyle="round")
            c.create_line(*p(16, 11, 22, 11), fill=INK, width=2, capstyle="round")
        else:
            # user-check: a tick.
            c.create_line(
                *p(16, 11, 18, 13, 22, 9),
                fill=INK, width=2, capstyle="round", joinstyle="round",
            )
