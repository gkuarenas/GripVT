"""ProgressMixin -- SESSION / TRIAL progress blocks and the TARGET FORCE
readout for the Session window.

Part of the ui/session/ package (see ui/session/window.py's module docstring
for the package overview). Split out of the former ui/session_window.py
verbatim (mechanical move, no logic changes) -- see
docs/logs/20260729_141951_codebase_reorg_and_session_split_plan.md, D8.
"""
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable

from core.trial_sequence import TRIAL_COUNT, TrialSequence, accept_band_kg, target_kg
from output.participants import ParticipantRecord
from ui import theme

# D1 (session-indicator plan): pure UI/visit bookkeeping, not a sequence-
# driving module like TRIAL_COUNT -- lives here next to self._trial_max.
SESSION_COUNT = 10


class ProgressMixin:
    """SESSION/TRIAL progress + TARGET FORCE. Composed into SessionWindow
    alongside the other ui/session/ mixins -- see window.py for the single
    __init__ that owns all of the attributes declared below (D1/D3)."""

    # ── Cross-group attributes (D4): owned/assigned in window.py's __init__
    # -- declared here only so mypy resolves self.<attr> access within this
    # mixin. No runtime effect.
    _session_num: int
    _undo_available: bool
    _trial: int
    _trial_max: int
    _sequence: TrialSequence | None
    _participant: ParticipantRecord | None
    _target_num_label: tk.Label
    _target_sub_label: tk.Label
    _target_band_label: tk.Label
    _f_kicker: tkfont.Font
    _f_progress_value: tkfont.Font
    _f_strip_tag: tkfont.Font
    _f_small: tkfont.Font

    # Cross-group methods (D2): plain self.method() calls into the scaffold
    # group (window.py) and the recording group, declared the same way.
    _build_button: Callable[..., tk.Frame]
    _on_undo_trial: Callable[[], None]

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

    def _build_session_block(self, parent: tk.Frame) -> None:
        """D5 (session-indicator plan): SESSION needs the same live-refresh
        treatment as _build_trial_block below rather than _progress_block's
        throwaway static build -- _confirm_start's self._session_num += 1
        has to repaint this in place, not just at construction."""
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(fill="x")

        row = tk.Frame(wrap, bg=theme.BG)
        row.pack(fill="x")
        tk.Label(
            row, text="SESSION", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker,
        ).pack(side="left")
        self._session_value_label = tk.Label(
            row, text=self._session_value_text(), bg=theme.BG, fg=theme.INK,
            font=self._f_progress_value,
        )
        self._session_value_label.pack(side="right")

        track = tk.Frame(wrap, bg=theme.NEUTRAL_200, height=6)
        track.pack(fill="x", pady=(8, 0))
        self._session_fill = tk.Frame(track, bg=theme.ACCENT)
        self._session_fill.place(relx=0, rely=0, relwidth=self._session_frac(), relheight=1)

    def _session_value_text(self) -> str:
        return f"{self._session_num} / {SESSION_COUNT}"

    def _session_frac(self) -> float:
        return self._session_num / SESSION_COUNT if SESSION_COUNT else 0.0

    def _refresh_session_block(self) -> None:
        """D4 (session-indicator plan): called once, immediately after
        _confirm_start's self._session_num += 1 -- the only place SESSION's
        value changes within a window's lifetime (see D4 for why TRIAL needs
        three refresh call sites and SESSION needs exactly one)."""
        self._session_value_label.configure(text=self._session_value_text())
        self._session_fill.place_configure(relwidth=self._session_frac())

    def _kept_sessions_phrase(self) -> str:
        """D6: single source of truth for "how many prior sessions in this
        visit are still on disk," read at the moment Restart Session would
        fire. D7 keeps Restart disabled below self._session_num == 1, so 0
        is unreachable here -- only the == 1 (no priors yet) vs. > 1 split
        matters."""
        if self._session_num <= 1:
            return "No prior sessions yet."
        return f"Sessions 1-{self._session_num - 1} are kept."

    def _build_trial_block(self, parent: tk.Frame) -> None:
        """TRIAL gets its own rendering path rather than _progress_block
        (Phase 8, E7): it carries a live value/bar plus an inline Undo
        button, so the value label and bar fill are stored for
        _refresh_trial_block to update without a rebuild. SESSION above
        (_build_session_block) now follows the same pattern; _progress_block
        itself is unused but left in place (D5, session-indicator plan)."""
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(fill="x")

        row = tk.Frame(wrap, bg=theme.BG)
        row.pack(fill="x")
        kicker_row = tk.Frame(row, bg=theme.BG)
        kicker_row.pack(side="left")
        tk.Label(
            kicker_row, text="TRIAL", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker,
        ).pack(side="left")
        # Compact ghost Undo, inline after the kicker (design: margin-left
        # 10px, 10.5px label -- self._f_strip_tag is this file's existing
        # 10px heading font, close enough to reuse rather than add a new one).
        self._undo_btn = self._build_button(
            kicker_row, "Undo", "ghost", "undo", self._on_undo_trial,
            enabled=self._undo_available, font=self._f_strip_tag,
        )
        self._undo_btn.pack(side="left", padx=(10, 0))

        self._trial_value_label = tk.Label(
            row, text=self._trial_value_text(), bg=theme.BG, fg=theme.INK,
            font=self._f_progress_value,
        )
        self._trial_value_label.pack(side="right")

        track = tk.Frame(wrap, bg=theme.NEUTRAL_200, height=6)
        track.pack(fill="x", pady=(8, 0))
        self._trial_fill = tk.Frame(track, bg=theme.ACCENT)
        self._trial_fill.place(relx=0, rely=0, relwidth=self._trial_frac(), relheight=1)

    def _trial_value_text(self) -> str:
        return f"{self._trial} / {self._trial_max}"

    def _trial_frac(self) -> float:
        return self._trial / self._trial_max if self._trial_max else 0.0

    def _target_pct_for_trial(self) -> int | None:
        if self._sequence is None:
            return None
        assert 0 <= self._trial - 1 < TRIAL_COUNT
        return self._sequence.levels_pct[self._trial - 1]

    def _target_kg_for_trial(self) -> float | None:
        pct = self._target_pct_for_trial()
        if pct is None or self._participant is None:
            return None
        return target_kg(pct, self._participant.mvc_kg)

    def _refresh_trial_block(self) -> None:
        """Repaints the TRIAL value/bar/Undo in place -- called after every
        Next Trial / Undo (Phase 8), never rebuilds _build_trial_block's
        widgets."""
        self._trial_value_label.configure(text=self._trial_value_text())
        self._trial_fill.place_configure(relwidth=self._trial_frac())
        self._undo_btn.set_enabled(self._undo_available)

    def _target_num_text(self) -> str:
        kg = self._target_kg_for_trial()
        return f"{kg:.1f}" if kg is not None else "—"

    def _target_sub_text(self) -> str:
        pct = self._target_pct_for_trial()
        return f"{pct}% MVC" if pct is not None else "AWAITING START"

    def _target_band_text(self) -> str:
        """The protocol's advisory accept range, for the operator to eyeball
        against the Biopac readout. Display only -- nothing reads this back."""
        pct = self._target_pct_for_trial()
        if pct is None or self._participant is None:
            return "—"
        lo, hi = accept_band_kg(pct, self._participant.mvc_kg)
        return f"OK {lo:.1f} – {hi:.1f} kg"

    def _target_color(self) -> str:
        return theme.INK if self._target_pct_for_trial() is not None else theme.INK_55

    def _refresh_target_block(self) -> None:
        """Repaints the TARGET FORCE value/subline in place -- called after
        every Next Trial / Undo (Phase 8), mirroring _refresh_trial_block."""
        color = self._target_color()
        self._target_num_label.configure(text=self._target_num_text(), fg=color)
        self._target_sub_label.configure(text=self._target_sub_text(), fg=color)
        self._target_band_label.configure(text=self._target_band_text())
