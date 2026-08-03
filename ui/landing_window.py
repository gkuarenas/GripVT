"""Landing window for the GripVT desktop app.

First screen shown when the app is launched. Asks whether the operator is
setting up a **new** participant or resuming a session after an app crash,
then routes to the appropriate flow (wired later -- for now the two buttons
call the ``on_new_participant`` / ``on_resume_session`` hooks, defaulting to
a print stub).

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

from output.participants import list_resumable_participants
from ui import theme
from ui.session.progress import SESSION_COUNT

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


def _has_resumable_participant() -> bool:
    """Whether any `participantNNN_*` folder under RECORDINGS_DIR is resumable.

    Delegates to `output.participants.list_resumable_participants`, which owns
    the scan (folder under RECORDINGS_DIR, `participant.json` exists, fewer
    than SESSION_COUNT session dirs on disk). Never raises: the landing
    window must open even if `recordings/` is missing, unreadable, or
    malformed -- `list_resumable_participants` already guarantees this, but
    the try/except is kept here too so a future change to that contract can't
    silently break landing-window startup.
    """
    try:
        return any(list_resumable_participants(SESSION_COUNT))
    except Exception:
        return False


class LandingWindow(tk.Tk):
    def __init__(
        self,
        on_new_participant: Callable[[], None] | None = None,
        on_resume_session: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_new = on_new_participant or (lambda: print("New Participant clicked"))
        self._on_resume = on_resume_session or (
            lambda: print("Resume Session clicked")
        )

        self.title("GripVT")
        self.configure(bg=BG)
        # Fixed-size, non-resizable: the design is a precise 600x300 canvas and
        # the space-between layout is tuned to that height, not a fluid box.
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._init_fonts()
        self._has_resumable = _has_resumable_participant()
        self._build_body()
        self.update()
        theme.apply_dark_titlebar(self)

    # ── window helpers ──────────────────────────────────────────────────────
    def _center(self, w: int, h: int) -> None:
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3  # a touch above dead-centre
        self.geometry(f"{w}x{h}+{x}+{y}")

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
        self._font_caption = tkfont.Font(family=pick("Archivo"), size=-11, weight="normal")

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
            text="Start a new participant or resume a session?",
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
        self._build_choice(
            row, 1, "resume", "Resume Session", self._on_resume,
            enabled=self._has_resumable,
        )

    def _build_choice(
        self,
        parent: tk.Frame,
        col: int,
        icon: str,
        label: str,
        command: Callable[[], None],
        *,
        enabled: bool = True,
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
        # Disabled tiles (D8) swap ACCENT/INK-ish colours for DIVIDER as the
        # opacity stand-in -- tkinter has no real alpha.
        fg = INK if enabled else DIVIDER
        self._draw_icon(icon_canvas, icon, color=fg)

        # Wrapped in its own column so the disabled "No incomplete sessions"
        # caption can pack underneath the label without disturbing the icon.
        text_col = tk.Frame(inner, bg=BG)
        text_col.pack(side="left", padx=(14, 0))  # 14px icon-to-text gap
        text = tk.Label(text_col, text=label, bg=BG, fg=fg, font=self._font_button)
        text.pack(anchor="w")

        if not enabled:
            # No hover/press bindings, no command, default cursor (D8): the
            # tile is inert when there is nothing to resume.
            tk.Label(
                text_col, text="No incomplete sessions", bg=BG, fg=DIVIDER,
                font=self._font_caption,
            ).pack(anchor="w")
            btn.configure(cursor="arrow")
            return

        # Every widget that shows BG must recolour together on hover, or the
        # canvas/label would punch opaque holes in the fill (no tk alpha).
        tinted = [btn, inner, icon_canvas, text_col, text]
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

    # ── icons (lucide user-plus / rotate-ccw, stroked) ──────────────────────
    def _draw_icon(self, c: tk.Canvas, kind: str, color: str = INK) -> None:
        s = 25 / 24  # design icons are authored on a 24x24 grid, drawn at 25px

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        if kind == "new":
            # user-plus: head + shoulders + plus, unchanged from before (D6).
            c.create_oval(*p(5, 3, 13, 11), outline=color, width=2)
            c.create_line(
                *p(16, 21, 16, 19, 12, 15, 6, 15, 2, 19, 2, 21),
                fill=color, width=2, smooth=True, capstyle="round", joinstyle="round",
            )
            c.create_line(*p(19, 8, 19, 14), fill=color, width=2, capstyle="round")
            c.create_line(*p(16, 11, 22, 11), fill=color, width=2, capstyle="round")
            return

        # resume: lucide rotate-ccw -- an open arc plus a hook at the gap,
        # replacing the former user-check tick (D6). No head/shoulders here.
        c.create_arc(
            *p(3, 3, 21, 21), start=100, extent=320,
            style=tk.ARC, outline=color, width=2,
        )
        c.create_line(
            *p(3, 3, 3, 8, 8, 8),
            fill=color, width=2, capstyle="round", joinstyle="round",
        )
