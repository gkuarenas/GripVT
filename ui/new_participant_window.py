"""New Participant window for the GripVT desktop app.

Opened from the landing window's "New Participant" choice. Collects the
participant's anthropometrics and activity level before a session. Visual
design ported from the Modernist design system (Claude Design: "GripVT New
Participant") -- flat, 0px radius, Archivo type, single hot red-orange accent.

On Save & Continue the entered data is validated and persisted: a de-identified
``participant.json`` is written under ``recordings/participantNNN_<ts>/`` and the
name is routed to an external key file (see ``output/participants.py``). The
resulting ``ParticipantRecord`` is handed to ``on_save`` to open the session.
"""
import tkinter as tk
from typing import Callable

from output import participants
from output.participants import ParticipantRecord
from ui import theme

WINDOW_W = 800
WINDOW_H = 500

# Error banner fill: design's color-mix(accent 10%, bg) pre-composited (no tk alpha).
_BANNER_BG = "#f2dfdc"

ACTIVITY_LEVELS = [
    ("Sedentary", "Little to no exercise, desk-based"),
    ("Lightly Active", "Light exercise 1–3 days/week"),
    ("Moderately Active", "Moderate exercise 3–5 days/week"),
    ("Very Active", "Hard exercise 6–7 days/week"),
    ("Extremely Active", "Very hard exercise, physical job"),
]

MEASUREMENTS = [
    ("Age (years)", "e.g. 34", "int"),
    ("Weight (kg)", "e.g. 72.5", "decimal"),
    ("Height (cm)", "e.g. 172", "decimal"),
    ("Forearm Circ. (cm)", "e.g. 26.4", "decimal"),
    ("MVC (kg)", "e.g. 32.1", "decimal"),
]


class NewParticipantWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        on_save: Callable[[ParticipantRecord], None] | None = None,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._on_save = on_save or (lambda record: print(f"Saved {record.participant_id}"))
        self._on_back = on_back or (lambda: print("Back clicked"))

        self.title("GripVT — New Participant")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        self._activity = tk.StringVar(value=ACTIVITY_LEVELS[0][0])
        self._dots: dict[str, tk.Canvas] = {}
        self._ph_state: dict[tk.Entry, dict] = {}
        self._field_errors: dict[tk.Entry, tk.Label] = {}

        self._init_fonts()
        self._build_body()
        theme.apply_dark_titlebar(self)

    def _center(self, w: int, h: int) -> None:
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _init_fonts(self) -> None:
        self._f_eyebrow = theme.make_font(self, "semibold", 11)
        self._f_heading = theme.make_font(self, "heading", 25)
        self._f_subtitle = theme.make_font(self, "body", 13)
        self._f_label = theme.make_font(self, "body", 12)
        self._f_input = theme.make_font(self, "body", 14)
        self._f_radio = theme.make_font(self, "semibold", 13)
        self._f_desc = theme.make_font(self, "body", 11)
        self._f_button = theme.make_font(self, "heading", 14)

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=36, pady=(26, 22))

        self._build_footer(body)   # packed bottom first so activity can flex
        self._build_header(body)
        self._build_activity(body)  # fills the space between header and footer

    def _build_header(self, parent: tk.Frame) -> None:
        head = tk.Frame(parent, bg=theme.BG)
        head.pack(side="top", fill="x")

        eyebrow = tk.Frame(head, bg=theme.BG)
        eyebrow.pack(fill="x")
        tk.Frame(eyebrow, bg=theme.ACCENT, width=8, height=8).pack(side="left")
        tk.Label(
            eyebrow, text="GRIPVT", bg=theme.BG, fg=theme.INK_65, font=self._f_eyebrow
        ).pack(side="left", padx=(8, 0))

        tk.Label(
            head, text="New Participant", bg=theme.BG, fg=theme.INK,
            font=self._f_heading, anchor="w",
        ).pack(fill="x", pady=(8, 0))
        tk.Label(
            head, text="Enter the participant's details to begin a session.",
            bg=theme.BG, fg=theme.INK_60, font=self._f_subtitle, anchor="w",
        ).pack(fill="x", pady=(5, 0))
        tk.Frame(head, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(13, 18))

        self._name = self._build_field(head, "Full Name", "e.g. Juan Dela Cruz")
        self._name.master.pack(fill="x", pady=(0, 15))

        grid = tk.Frame(head, bg=theme.BG)
        grid.pack(fill="x")
        self._measures: list[tk.Entry] = []
        for col, (label, placeholder, numeric) in enumerate(MEASUREMENTS):
            grid.columnconfigure(col, weight=1, uniform="m")
            field = self._build_field(grid, label, placeholder, numeric)
            field.master.grid(
                row=0, column=col, sticky="ew",
                padx=(0, 14) if col < len(MEASUREMENTS) - 1 else 0,
            )
            self._measures.append(field)

    def _build_activity(self, parent: tk.Frame) -> None:
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(side="top", fill="both", expand=True, pady=(16, 0))

        tk.Label(
            wrap, text="Activity Level", bg=theme.BG, fg=theme.INK_70,
            font=self._f_label, anchor="w",
        ).pack(fill="x", pady=(0, 8))

        options = tk.Frame(wrap, bg=theme.BG)
        options.pack(fill="x")
        for col, (value, desc) in enumerate(ACTIVITY_LEVELS):
            options.columnconfigure(col, weight=1, uniform="a")
            self._build_radio(options, col, value, desc)

    def _build_footer(self, parent: tk.Frame) -> None:
        footer = tk.Frame(parent, bg=theme.BG)
        footer.pack(side="bottom", fill="x")
        self._error_banner = tk.Frame(
            footer, bg=_BANNER_BG, highlightthickness=1,
            highlightbackground=theme.ACCENT, highlightcolor=theme.ACCENT,
        )
        banner_inner = tk.Frame(self._error_banner, bg=_BANNER_BG)
        banner_inner.pack(fill="x", padx=10, pady=5)
        self._alert_icon(banner_inner, _BANNER_BG).pack(side="left", padx=(0, 7))
        tk.Label(
            banner_inner, text="Please complete all required fields.",
            bg=_BANNER_BG, fg=theme.ACCENT_700, font=self._f_desc,
        ).pack(side="left")

        self._divider = tk.Frame(footer, bg=theme.DIVIDER, height=2)
        self._divider.pack(fill="x", pady=(0, 14))

        row = tk.Frame(footer, bg=theme.BG)
        row.pack(fill="x")
        self._build_button(row, "Back", "secondary", "left", self._on_back).pack(side="left")
        self._build_button(row, "Save & Continue", "primary", "right", self._handle_save).pack(side="right")

    # ── collect + validate + persist ────────────────────────────────────────
    def _labeled_entries(self) -> list[tuple[str, tk.Entry]]:
        return [("Full Name", self._name)] + [
            (label, entry) for (label, _p, _n), entry in zip(MEASUREMENTS, self._measures)
        ]

    def _field_value(self, entry: tk.Entry) -> str:
        state = self._ph_state.get(entry)
        if state is not None and state["placeholder"]:
            return ""
        return entry.get().strip()

    def _collect(self) -> tuple[dict | None, list[str]]:
        missing: list[str] = []
        name = self._field_value(self._name)
        if not name:
            missing.append("Full Name")

        values: dict = {}
        keys = ["age", "weight_kg", "height_cm", "forearm_circ_cm", "mvc_kg"]
        for (label, _ph, numeric), entry, key in zip(MEASUREMENTS, self._measures, keys):
            raw = self._field_value(entry)
            parsed = None
            if raw not in ("", "."):
                try:
                    parsed = int(raw) if numeric == "int" else float(raw)
                except ValueError:
                    parsed = None
            if parsed is None:
                missing.append(label)
            else:
                values[key] = parsed

        if missing:
            return None, missing
        values["name"] = name
        values["activity_level"] = self._activity.get()
        return values, []

    def _handle_save(self) -> None:
        values, missing = self._collect()
        if values is None:
            self._show_field_errors(missing)
            return
        self._clear_field_errors()
        record = participants.create_participant(**values)
        self._on_save(record)

    def _show_field_errors(self, missing: list[str]) -> None:
        for label, entry in self._labeled_entries():
            bad = label in missing
            entry.configure(highlightbackground=theme.ACCENT if bad else theme.DIVIDER)
            err = self._field_errors[entry]
            if bad:
                err.pack(fill="x", anchor="w", pady=(4, 0))
            else:
                err.pack_forget()
        self._error_banner.pack(fill="x", pady=(0, 10), before=self._divider)

    def _clear_field_errors(self) -> None:
        for _label, entry in self._labeled_entries():
            entry.configure(highlightbackground=theme.DIVIDER)
            self._field_errors[entry].pack_forget()
        self._error_banner.pack_forget()

    def _clear_one(self, entry: tk.Entry) -> None:
        """Clear a single field's error once it holds input, matching the design."""
        err = self._field_errors[entry]
        if err.winfo_manager() != "pack" or not self._field_value(entry):
            return
        entry.configure(highlightbackground=theme.DIVIDER)
        err.pack_forget()
        if not any(e.winfo_manager() == "pack" for e in self._field_errors.values()):
            self._error_banner.pack_forget()

    def _alert_icon(self, parent: tk.Misc, bg: str) -> tk.Canvas:
        c = tk.Canvas(parent, width=13, height=13, bg=bg, highlightthickness=0)
        s = 13 / 24
        col = theme.ACCENT_700
        c.create_oval(*(v * s for v in (2, 2, 22, 22)), outline=col, width=1)
        c.create_line(*(v * s for v in (12, 7, 12, 13)), fill=col, width=1)
        c.create_line(*(v * s for v in (12, 16, 12, 16.6)), fill=col, width=1)
        return c

    # ── fields ──────────────────────────────────────────────────────────────
    def _build_field(
        self, parent: tk.Misc, label: str, placeholder: str, numeric: str | None = None
    ) -> tk.Entry:
        field = tk.Frame(parent, bg=theme.BG)

        head = tk.Frame(field, bg=theme.BG)
        head.pack(fill="x", pady=(0, 5))
        tk.Label(head, text=label, bg=theme.BG, fg=theme.INK_70, font=self._f_label).pack(side="left")
        tk.Label(head, text=" *", bg=theme.BG, fg=theme.ACCENT_700, font=self._f_label).pack(side="left")

        entry = tk.Entry(
            field, bg=theme.SURFACE, fg=theme.INK, font=self._f_input,
            insertbackground=theme.ACCENT, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=theme.DIVIDER,
            highlightcolor=theme.ACCENT,
        )
        if numeric is not None:
            allow_decimal = numeric == "decimal"
            vcmd = (entry.register(lambda P, d=allow_decimal: _is_number(P, d)), "%P")
            entry.configure(validate="key", validatecommand=vcmd)
        entry.pack(fill="x", ipady=6)
        self._ph_state[entry] = _add_placeholder(entry, placeholder)

        msg = "Full name is required." if label == "Full Name" else "Required."
        self._field_errors[entry] = tk.Label(
            field, text=msg, bg=theme.BG, fg=theme.ACCENT_700, font=self._f_desc, anchor="w",
        )
        entry.bind("<KeyRelease>", lambda _e, en=entry: self._clear_one(en), add="+")
        return entry

    # ── radios (custom flat dot, matching .radio) ───────────────────────────
    def _build_radio(self, parent: tk.Frame, col: int, value: str, desc: str) -> None:
        cell = tk.Frame(parent, bg=theme.BG)
        cell.grid(row=0, column=col, sticky="new", padx=(0, 14) if col < len(ACTIVITY_LEVELS) - 1 else 0)

        top = tk.Frame(cell, bg=theme.BG)
        top.pack(fill="x")
        dot = tk.Canvas(top, width=16, height=16, bg=theme.BG, highlightthickness=0)
        dot.pack(side="left")
        tk.Label(top, text=value, bg=theme.BG, fg=theme.INK, font=self._f_radio).pack(
            side="left", padx=(8, 0)
        )
        tk.Label(
            cell, text=desc, bg=theme.BG, fg=theme.INK_55, font=self._f_desc,
            wraplength=104, justify="left", anchor="w",
        ).pack(fill="x", padx=(24, 0), pady=(6, 0))

        self._dots[value] = dot
        self._draw_dot(value)
        for w in (cell, top, dot, *top.winfo_children(), *cell.winfo_children()):
            w.bind("<Button-1>", lambda _e, v=value: self._select_activity(v))
            w.configure(cursor="hand2")

    def _select_activity(self, value: str) -> None:
        self._activity.set(value)
        for v in self._dots:
            self._draw_dot(v)

    def _draw_dot(self, value: str) -> None:
        c = self._dots[value]
        c.delete("all")
        selected = self._activity.get() == value
        outline = theme.ACCENT if selected else theme.DIVIDER
        c.create_oval(1, 1, 15, 15, outline=outline, width=2, fill=theme.BG)
        if selected:
            c.create_oval(5, 5, 11, 11, outline=theme.ACCENT, fill=theme.ACCENT)

    # ── buttons (composite flat, matching .btn) ─────────────────────────────
    def _build_button(
        self, parent: tk.Frame, label: str, kind: str, arrow: str, command: Callable[[], None]
    ) -> tk.Frame:
        primary = kind == "primary"
        fill = theme.ACCENT if primary else theme.BG
        fg = theme.BG if primary else theme.INK
        border = fill if primary else theme.DIVIDER

        btn = tk.Frame(
            parent, bg=fill, highlightthickness=1,
            highlightbackground=border, highlightcolor=border,
        )
        inner = tk.Frame(btn, bg=fill)
        inner.pack(padx=14, pady=8)

        widgets: list[tk.Widget] = [btn, inner]
        if arrow == "left":
            icon = self._arrow(inner, "left", fg, fill)
            icon.pack(side="left", padx=(0, 6))
            widgets.append(icon)
        text = tk.Label(inner, text=label, bg=fill, fg=fg, font=self._f_button)
        text.pack(side="left")
        widgets.append(text)
        if arrow == "right":
            icon = self._arrow(inner, "right", fg, fill)
            icon.pack(side="left", padx=(6, 0))
            widgets.append(icon)

        hover, active = (theme.ACCENT_600, theme.ACCENT_700) if primary else (theme.HOVER, theme.ACTIVE)
        self._wire_button(btn, widgets, fill, hover, active, command)
        return btn

    def _arrow(self, parent: tk.Misc, direction: str, color: str, bg: str) -> tk.Canvas:
        c = tk.Canvas(parent, width=14, height=14, bg=bg, highlightthickness=0)
        s = 14 / 24

        def p(*coords: float) -> list[float]:
            return [v * s for v in coords]

        if direction == "right":
            c.create_line(*p(5, 12, 19, 12), fill=color, width=2, capstyle="round")
            c.create_line(*p(12, 5, 19, 12, 12, 19), fill=color, width=2,
                          capstyle="round", joinstyle="round")
        else:
            c.create_line(*p(19, 12, 5, 12), fill=color, width=2, capstyle="round")
            c.create_line(*p(12, 19, 5, 12, 12, 5), fill=color, width=2,
                          capstyle="round", joinstyle="round")
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


def _is_number(proposed: str, allow_decimal: bool) -> bool:
    """True if `proposed` is a non-negative number so far (empty is fine).

    No minus sign is ever allowed, so a value below zero is untypeable; letters
    are rejected. For decimals, at most one dot.
    """
    if proposed == "":
        return True
    allowed = set("0123456789") | ({"."} if allow_decimal else set())
    if any(ch not in allowed for ch in proposed):
        return False
    return proposed.count(".") <= 1


def _add_placeholder(entry: tk.Entry, text: str) -> dict:
    """Grey placeholder that clears on focus and restores while empty.

    Returns the mutable state dict so callers can tell whether the field is
    still showing placeholder text (i.e. effectively empty) rather than input.
    """
    state = {"placeholder": True}

    def show() -> None:
        # Placeholder text isn't a valid number; write it with key validation
        # off so it can't trip and disable the field's validatecommand.
        mode = entry.cget("validate")
        entry.configure(validate="none")
        entry.delete(0, "end")
        entry.insert(0, text)
        entry.configure(validate=mode)
        entry.configure(fg=theme.INK_55)
        state["placeholder"] = True

    def on_focus_in(_e: object) -> None:
        if state["placeholder"]:
            entry.delete(0, "end")
            entry.configure(fg=theme.INK)
            state["placeholder"] = False

    def on_focus_out(_e: object) -> None:
        if not entry.get():
            show()

    show()
    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)
    return state
