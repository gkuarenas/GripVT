"""Resume Session picker for the GripVT desktop app.

Opened from the landing window's "Resume Session" choice (gated on
``output.participants.list_resumable_participants`` returning at least one
row -- see ``ui/landing_window.py::_has_resumable_participant``). This window
lets the operator pick a participant whose visit was interrupted -- an app
crash, a power loss, a forced quit -- mid-way through their up-to-SESSION_COUNT
sessions, and continue that visit rather than starting a fresh one.

Scope is crash recovery *within one visit*: picking a row skips straight to
``CalibrationWindow`` (anthropometrics entry is skipped, because
``participant.json`` already exists on disk for that participant) and a new
``sessionN_<ts>`` folder is written under the chosen participant on Continue.
There is no partial-trial state restoration -- that is out of scope (see
``docs/logs/20260803_191346_resume_session_window_plan.md``).

Visual design ported from the Modernist design system (Claude Design: "GripVT
Resume Session") -- flat, 0px radius, Archivo type, single hot red-orange
accent, same hand-built composite-widget approach as ``NewParticipantWindow``
(plain tk/ttk buttons can't be flattened to this look).
"""
import tkinter as tk
from typing import Callable

from output.participants import ParticipantRecord, ResumableParticipant, list_resumable_participants, load_participant
from ui import theme
from ui.session.progress import SESSION_COUNT

WINDOW_W = 800
WINDOW_H = 560

# Disabled "Continue to Calibration" fill/border: theme.ACCENT pre-composited
# at 45% over theme.BG (tk has no real alpha) -- D18.
_ACCENT_45 = "#f09b8e"

_ROW_H = 52


class ResumeSessionWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        on_continue: Callable[[ParticipantRecord], None] | None = None,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._on_continue = on_continue or (
            lambda record: print(f"Continue clicked for {record.participant_id}")
        )
        self._on_back = on_back or (lambda: print("Back clicked"))

        self.title("GripVT — Resume Session")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self._center(WINDOW_W, WINDOW_H)

        # Captured once at open -- the list does not live-refresh while this
        # window is open (D11: it is only reachable when at least one row
        # exists, and nothing else in this flow can change recordings/ while
        # this window has focus).
        self._items: list[ResumableParticipant] = list_resumable_participants(SESSION_COUNT)
        self._selected: ResumableParticipant | None = None
        # Per-row bookkeeping for repainting on hover/select (D17): keyed by
        # id() of the item since ResumableParticipant is a frozen dataclass
        # (hashable, but identity is simplest and matches list membership).
        self._row_tinted: dict[int, list[tk.Widget]] = {}
        self._row_check: dict[int, tk.Canvas] = {}

        self._init_fonts()
        self._build_body()
        self.update()
        theme.apply_dark_titlebar(self)

    # ── window helpers ──────────────────────────────────────────────────────
    def _center(self, w: int, h: int) -> None:
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _init_fonts(self) -> None:
        self._f_eyebrow = theme.make_font(self, "semibold", 11)
        self._f_heading = theme.make_font(self, "heading", 25)
        self._f_subtitle = theme.make_font(self, "body", 13)
        self._f_th = theme.make_font(self, "semibold", 11)
        self._f_pid = theme.make_font(self, "heading", 14)
        self._f_cell = theme.make_font(self, "body", 12)
        self._f_button = theme.make_font(self, "heading", 14)

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=36, pady=(26, 22))

        self._build_footer(body)  # packed bottom first so the list can flex
        self._build_header(body)
        self._build_list(body)

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
            head, text="Resume Session", bg=theme.BG, fg=theme.INK,
            font=self._f_heading, anchor="w",
        ).pack(fill="x", pady=(8, 0))
        tk.Label(
            head, text="Select a participant to continue their incomplete visit.",
            bg=theme.BG, fg=theme.INK_60, font=self._f_subtitle, anchor="w",
        ).pack(fill="x", pady=(5, 0))
        tk.Frame(head, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(13, 18))

    def _build_footer(self, parent: tk.Frame) -> None:
        footer = tk.Frame(parent, bg=theme.BG)
        footer.pack(side="bottom", fill="x")
        tk.Frame(footer, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(0, 14))

        row = tk.Frame(footer, bg=theme.BG)
        row.pack(fill="x")
        self._build_button(row, "Back", "secondary", "left", self._on_back)[0].pack(side="left")
        self._continue_btn, self._continue_widgets = self._build_button(
            row, "Continue to Calibration", "primary", "right", self._handle_continue,
        )
        self._continue_btn.pack(side="right")

        # D16: single inline error label, not packed until Continue hits a
        # corrupt/half-written participant.json -- "packed under the footer
        # row" means appended below `row` within this same footer frame.
        self._error_label = tk.Label(
            footer, text="Could not read participant.json — pick another participant.",
            bg=theme.BG, fg=theme.ACCENT_700, font=self._f_cell, anchor="w",
        )

        # Continue starts disabled -- D10, no row is selected on open.
        self._set_continue_enabled(False)

    # ── column layout shared by the header row and every data row (D22) ────
    def _configure_columns(self, frame: tk.Frame) -> None:
        frame.columnconfigure(0, weight=12)
        frame.columnconfigure(1, weight=14)
        frame.columnconfigure(2, weight=10)
        frame.columnconfigure(3, weight=0, minsize=24)

    # ── scrollable list (D15) ────────────────────────────────────────────────
    def _build_list(self, parent: tk.Frame) -> None:
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(side="top", fill="both", expand=True)

        self._build_list_header(wrap)

        list_area = tk.Frame(wrap, bg=theme.BG)
        list_area.pack(fill="both", expand=True)

        canvas = tk.Canvas(list_area, bg=theme.BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        rows = tk.Frame(canvas, bg=theme.BG)
        window_id = canvas.create_window((0, 0), window=rows, anchor="nw")

        def on_rows_configure(_e: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(e: object) -> None:
            canvas.itemconfigure(window_id, width=e.width)

        rows.bind("<Configure>", on_rows_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_wheel(e: object) -> None:
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def bind_wheel(_e: object) -> None:
            canvas.bind_all("<MouseWheel>", on_wheel)

        def unbind_wheel(_e: object) -> None:
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)

        for item in self._items:
            self._build_row(rows, item)

    def _build_list_header(self, parent: tk.Frame) -> None:
        head = tk.Frame(parent, bg=theme.BG)
        head.pack(fill="x")
        self._configure_columns(head)

        labels = ("PARTICIPANT", "PROGRESS", "LAST SESSION", "")
        for col, text in enumerate(labels):
            tk.Label(
                head, text=text, bg=theme.BG, fg=theme.INK_55, font=self._f_th,
                anchor="w",
            ).grid(row=0, column=col, sticky="w", padx=(14 if col == 0 else 8, 0), pady=(0, 8))

        tk.Frame(parent, bg=theme.DIVIDER, height=1).pack(fill="x")

    def _build_row(self, parent: tk.Frame, item: ResumableParticipant) -> None:
        row = tk.Frame(parent, bg=theme.BG, height=_ROW_H)
        row.pack(fill="x")
        row.grid_propagate(False)
        self._configure_columns(row)

        # Participant
        cell0 = tk.Frame(row, bg=theme.BG)
        cell0.grid(row=0, column=0, sticky="nsew", padx=(14, 8))
        pid_label = tk.Label(
            cell0, text=item.participant_id, bg=theme.BG, fg=theme.INK,
            font=self._f_pid, anchor="w",
        )
        pid_label.pack(anchor="w")

        # Progress
        cell1 = tk.Frame(row, bg=theme.BG)
        cell1.grid(row=0, column=1, sticky="nsew", padx=8)
        prog = tk.Frame(cell1, bg=theme.BG)
        prog.pack(anchor="w")
        track = tk.Frame(prog, bg=theme.NEUTRAL_200, height=6, width=90)
        track.pack_propagate(False)
        track.pack(side="left")
        frac = item.complete_session_count / SESSION_COUNT if SESSION_COUNT else 0.0
        bar_fill = tk.Frame(track, bg=theme.ACCENT)
        bar_fill.place(relx=0, rely=0, relwidth=frac, relheight=1)
        count_label = tk.Label(
            prog, text=f"{item.complete_session_count}/{SESSION_COUNT}", bg=theme.BG,
            fg=theme.INK, font=self._f_cell,
        )
        count_label.pack(side="left", padx=(10, 0))

        # Last Session
        cell2 = tk.Frame(row, bg=theme.BG)
        cell2.grid(row=0, column=2, sticky="nsew", padx=8)
        dt = item.last_session_at
        if dt is not None:
            hour_12 = dt.hour % 12 or 12
            ampm = "AM" if dt.hour < 12 else "PM"
            text = f"{dt:%b} {dt.day} · {hour_12}:{dt:%M} {ampm}"
            color = theme.INK
        else:
            text = "No sessions yet"
            color = theme.INK_55
        last_label = tk.Label(
            cell2, text=text, bg=theme.BG, fg=color, font=self._f_cell, anchor="w",
        )
        last_label.pack(anchor="w")

        # Trailing checkmark column (D20) -- canvas bg follows the row fill.
        cell3 = tk.Frame(row, bg=theme.BG, width=24)
        cell3.grid(row=0, column=3, sticky="nsew")
        cell3.grid_propagate(False)
        check = tk.Canvas(cell3, width=16, height=16, bg=theme.BG, highlightthickness=0)
        check.place(relx=0.5, rely=0.5, anchor="center")

        # tinted: widgets that show the row's plain BG and must be recoloured
        # together on hover/select (D17). track/bar_fill keep their own
        # semantic colours (NEUTRAL_200 track, ACCENT fill) always -- but per
        # D17 the hover/click region is still "the row frame and every
        # descendant", so they're bound (below) without being recoloured.
        tinted = [row, cell0, pid_label, cell1, prog, count_label, cell2, last_label, cell3, check]
        bindable = tinted + [track, bar_fill]
        self._row_tinted[id(item)] = tinted
        self._row_check[id(item)] = check
        self._wire_row(item, bindable)

    # ── selection (D17, D20) ─────────────────────────────────────────────────
    def _wire_row(self, item: ResumableParticipant, bindable: list[tk.Widget]) -> None:
        def on_enter(_e: object) -> None:
            if self._selected is not item:
                self._paint_row(item, theme.ACCENT_06)

        def on_leave(_e: object) -> None:
            if self._selected is not item:
                self._paint_row(item, theme.BG)

        def on_click(_e: object) -> None:
            self._select(item)

        for w in bindable:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", on_click)
            w.configure(cursor="hand2")

    def _paint_row(self, item: ResumableParticipant, colour: str) -> None:
        for w in self._row_tinted[id(item)]:
            w.configure(bg=colour)

    def _select(self, item: ResumableParticipant) -> None:
        if self._selected is item:
            return
        was_none = self._selected is None
        if self._selected is not None:
            self._paint_row(self._selected, theme.BG)
            self._clear_check(self._selected)
        self._selected = item
        self._paint_row(item, theme.ACCENT_08)
        self._draw_check(item)
        if was_none:
            self._set_continue_enabled(True)

    def _draw_check(self, item: ResumableParticipant) -> None:
        c = self._row_check[id(item)]
        c.delete("all")
        s = 16 / 24
        c.create_line(
            *(v * s for v in (20, 6, 9, 17, 4, 12)),
            fill=theme.ACCENT, width=2, capstyle="round", joinstyle="round",
        )

    def _clear_check(self, item: ResumableParticipant) -> None:
        self._row_check[id(item)].delete("all")

    # ── buttons (ported from ui/new_participant_window.py, D19) ─────────────
    def _build_button(
        self, parent: tk.Frame, label: str, kind: str, arrow: str, command: Callable[[], None]
    ) -> tuple[tk.Frame, list[tk.Widget]]:
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
        return btn, widgets

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

    # ── Continue enable/disable (D18) ────────────────────────────────────────
    def _set_continue_enabled(self, enabled: bool) -> None:
        """Repaints AND (un)binds Continue in one place so the two states
        cannot drift apart (D18). fg stays theme.BG in both states -- this
        button is always "primary", and only its fill/border/cursor/bindings
        change between disabled and enabled."""
        fill = theme.ACCENT if enabled else _ACCENT_45
        for w in self._continue_widgets:
            w.configure(bg=fill)
        self._continue_btn.configure(highlightbackground=fill, highlightcolor=fill)

        if enabled:
            self._continue_btn.configure(cursor="hand2")
            for w in self._continue_widgets:
                w.configure(cursor="hand2")
            self._wire_button(
                self._continue_btn, self._continue_widgets, theme.ACCENT,
                theme.ACCENT_600, theme.ACCENT_700, self._handle_continue,
            )
        else:
            self._continue_btn.configure(cursor="arrow")
            for w in self._continue_widgets:
                w.configure(cursor="arrow")
                for ev in ("<Enter>", "<Leave>", "<ButtonPress-1>", "<ButtonRelease-1>"):
                    w.unbind(ev)

    # ── navigation ────────────────────────────────────────────────────────────
    def _handle_continue(self) -> None:
        if self._selected is None:
            return  # Continue is disabled with no selection, but guard anyway
        try:
            record = load_participant(self._selected.folder)
        except (OSError, ValueError, KeyError, TypeError):
            self._error_label.pack(fill="x", pady=(8, 0))
            return
        self._on_continue(record)
