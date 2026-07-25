"""Shared Modernist design tokens and Tk helpers for the GripVT windows.

The design system expresses every muted tone as a `color-mix` over `--color-bg`.
tkinter has no alpha, so each such tone is pre-composited to a solid hex against
BG here. Values were verified against the landing window's committed tokens.
"""
import ctypes
import sys
import tkinter.font as tkfont
from pathlib import Path

# Base roles
INK = "#201e1d"        # --color-text
BG = "#f3f2f2"         # --color-bg (window body)
SURFACE = "#eae9e9"    # --color-surface (input fill)
ACCENT = "#ec3013"     # --color-accent
ACCENT_600 = "#dd2b0f"
ACCENT_700 = "#ae1800"
NEUTRAL_200 = "#eae7e7"  # --color-neutral-200 (distinct from SURFACE #eae9e9)

# Ink over BG at design opacities
INK_70 = "#5f5e5d"     # subtitles / field labels
INK_65 = "#6a6868"     # eyebrow
INK_60 = "#747372"     # body subtitle
INK_55 = "#7f7d7d"     # radio descriptions
INK_45 = "#949392"     # input hover border
INK_35 = "#a9a8a7"     # idle placeholder numeral
DIVIDER = "#9f9d9d"    # 40% ink: hr + borders
HOVER = "#e4e3e3"      # 7% ink: secondary button hover
ACTIVE = "#d5d4d4"     # 14% ink: secondary button active

# Accent over BG at design opacities (tk has no alpha)
ACCENT_10 = "#f2dfdc"  # toast / alert fill
ACCENT_08 = "#f2e2e0"  # selected list row
ACCENT_06 = "#f3e6e5"  # list row hover

_FONT_DIR = Path(__file__).resolve().parent / "fonts" / "Archivo"
_fonts_registered = False


def register_archivo() -> None:
    """Load the bundled Archivo TTFs into this process (Windows, FR_PRIVATE).

    Idempotent; a no-op on macOS/Linux, where the windows fall back to the
    platform default sans.
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
        pass


def make_font(win, role: str, px: int) -> tkfont.Font:
    """Build a Tk font for a design role ("heading" 800, "semibold" 600, "body" 400).

    Static Archivo ships each cut as its own family, so the family name carries
    the weight; the sole exception is plain "Archivo", where bold needs a weight.
    Negative size forces pixels, not points.
    """
    register_archivo()
    families = set(tkfont.families(win))
    default = tkfont.nametofont("TkDefaultFont").actual("family")

    def pick(*names: str) -> str:
        return next((n for n in names if n in families), default)

    if role == "heading":
        fam = pick("Archivo ExtraBold", "Archivo")
        weight = "bold" if fam == "Archivo" else "normal"
    elif role == "semibold":
        fam, weight = pick("Archivo SemiBold", "Archivo"), "normal"
    else:
        fam, weight = pick("Archivo"), "normal"
    return tkfont.Font(family=fam, size=-px, weight=weight)


def apply_dark_titlebar(win) -> None:
    """Paint the native OS title bar dark to match the flat ink design (Windows).

    A DWM attribute set through ctypes; no-op and harmless on macOS/Linux.
    """
    if sys.platform != "win32":
        return
    try:
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        enabled = ctypes.c_int(1)
        for attr in (20, 19):  # 20 on Win10 20H1+/Win11, 19 on older builds
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(enabled), ctypes.sizeof(enabled)
            ) == 0:
                break
        win.withdraw()
        win.deiconify()
    except Exception:
        pass
