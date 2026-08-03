"""ModalMixin -- generic modal/toast plumbing and the Restart confirmation
modal for the Session window.

Part of the ui/session/ package (see ui/session/window.py's module docstring
for the package overview). Split out of the former ui/session_window.py
verbatim (mechanical move, no logic changes) -- see
docs/logs/20260729_141951_codebase_reorg_and_session_split_plan.md, D8.
"""
import json
import logging
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable

from output.participants import ParticipantRecord, move_sessions_to_aborted, record_abort
from output.writer import EventLog, now_ms
from ui import theme

from .progress import SESSION_COUNT

_TOAST_TIMEOUT_MS = 5000
_FADE_STEPS = 8
_FADE_INTERVAL_MS = 50


class ModalMixin:
    """Modal/toast plumbing + Restart modal. Composed into SessionWindow
    alongside the other ui/session/ mixins -- see window.py for the single
    __init__ that owns all of the attributes declared below (D1/D3)."""

    # ── Cross-group attributes (D4): owned/assigned in window.py's __init__
    # -- declared here only so mypy resolves self.<attr> access within this
    # mixin. No runtime effect.
    _event_log: EventLog | None
    _participant: ParticipantRecord | None
    _t0_wall_ms: float
    _t0_monotonic: float
    _session_num: int
    _session_num_at_open: int
    _on_back: Callable[[str | None], None]
    _toast_after: str | None
    _modal_scrim: tk.Toplevel | None
    _modal_panel: tk.Toplevel | None
    _modal_configure_id: str | None
    _modal_drag_offset: tuple[int, int]
    _f_gate_heading: tkfont.Font
    _f_gate_body: tkfont.Font
    _f_restart_row_heading: tkfont.Font
    _f_restart_danger_heading: tkfont.Font
    _f_small: tkfont.Font
    _f_toast: tkfont.Font
    _f_kicker: tkfont.Font

    # Cross-group methods (D2): plain self.method() calls into the recording,
    # progress, scaffold, and camera groups, declared the same way for mypy.
    _finalize_recording: Callable[..., None]
    release_camera: Callable[[], None]
    _kept_sessions_phrase: Callable[[], str]
    _build_button: Callable[..., tk.Frame]

    # This mixin is composed onto tk.Toplevel (window.py's class statement),
    # not declared here -- inherited, not cross-group per se, but needs the
    # same mypy-only declaration since ModalMixin itself has no base class.
    after: Callable[..., str]
    after_cancel: Callable[..., None]
    after_idle: Callable[..., str]
    bind: Callable[..., str]
    unbind: Callable[..., None]
    update_idletasks: Callable[[], None]
    winfo_rootx: Callable[[], int]
    winfo_rooty: Callable[[], int]
    winfo_width: Callable[[], int]
    winfo_height: Callable[[], int]

    # ── scaffold-adjacent build (built once in _build_body) ────────────────
    def _build_toast(self, parent: tk.Frame) -> None:
        # Top-right of the content frame -- content is already inset by
        # body's own (22, 20)/32 padding, so flush-to-corner here reproduces
        # the design's top:22px; right:32px without a second offset.
        self._toast_parent = parent
        self._toast = tk.Frame(
            parent, bg=theme.ACCENT_10, highlightthickness=1,
            highlightbackground=theme.ACCENT, highlightcolor=theme.ACCENT,
        )
        self._toast_inner = tk.Frame(self._toast, bg=theme.ACCENT_10)
        self._toast_inner.pack(padx=10, pady=7)
        self._toast_label = tk.Label(
            self._toast_inner, text="", bg=theme.ACCENT_10, fg=theme.ACCENT_700,
            font=self._f_toast, wraplength=220, justify="left",
        )
        self._toast_label.pack()

    # ── restart modal (Phase 4, E1/E8) ──────────────────────────────────────
    def _open_restart_modal(self) -> None:
        """Two asymmetric options -- Restart Session costs one session,
        Restart Entire Testing costs the whole visit. Both close this window
        via self._on_back(message); the toast lands on Calibration (E1)."""
        state: dict = {"reason": None, "chips": {}, "note_entry": None, "note_state": None}

        def repaint_chip(key: str) -> None:
            chip, label = state["chips"][key]
            selected = state["reason"] == key
            fill = theme.ACCENT_08 if selected else theme.BG
            fg = theme.ACCENT_700 if selected else theme.INK
            border = theme.ACCENT_600 if selected else theme.DIVIDER
            chip.configure(bg=fill, highlightbackground=border, highlightcolor=border)
            label.configure(bg=fill, fg=fg)

        def on_chip_click(key: str) -> None:
            # Single-select toggle -- clicking the selected chip deselects it.
            state["reason"] = None if state["reason"] == key else key
            for k in state["chips"]:
                repaint_chip(k)

        def note_text() -> str:
            entry = state["note_entry"]
            note_state = state["note_state"]
            if entry is None or note_state is None or note_state["placeholder"]:
                return ""
            return entry.get()

        def record_abandonment(scope: str) -> None:
            """F4: persist the abandonment at the *participant* level, then
            F7: relocate exactly the session directories that record just
            named into `aborted/`.

            Deliberately outside the `_event_log is not None` guard below.
            `_event_log` exists only between a confirmed Start and
            finalization, so the in-session `session_aborted` event misses the
            common case entirely -- Restart pressed between sessions, with
            nothing recording, previously left no trace anywhere and made an
            abandoned visit indistinguishable from a real multi-session one.

            Nothing is deleted; this only records which session directories
            were walked away from, then moves those same directories (D8).
            Failures are logged and swallowed: Restart is the recovery path,
            and being stuck in a broken session with a participant waiting is
            worse than losing the record or the move (D8/D9).

            The move uses the *exact* `abandoned_sessions` list just written
            to `aborted.json` (read back from disk, not recomputed) so the
            record and the moved directories can never disagree -- naming a
            directory in aborted.json that wasn't actually moved would be
            worse than either failing alone.
            """
            if self._participant is None:
                return
            try:
                path = record_abort(
                    self._participant,
                    scope=scope,
                    reason=state["reason"],
                    note=note_text(),
                    session_num_at_abort=self._session_num,
                )
            except Exception:
                logging.exception("failed to write aborted.json; continuing with Restart")
                return

            try:
                with open(path) as f:
                    payload = json.load(f)
                session_names = payload["aborts"][-1]["abandoned_sessions"]
            except Exception:
                logging.exception(
                    "failed to read back aborted.json; skipping session move"
                )
                return

            try:
                move_sessions_to_aborted(self._participant, session_names)
            except Exception:
                logging.exception(
                    "failed to move abandoned sessions into aborted/; continuing with Restart"
                )

        def confirm_session() -> None:
            # D6: Restart aborts the in-progress segment -- log the reason/
            # note as a session_aborted event, then discard the uncommitted
            # buffer (already-committed trials stay on disk untouched). No
            # event log exists if Restart was pressed before any Start, in
            # which case there is nothing to record.
            if self._event_log is not None:
                self._event_log.append({
                    "type": "session_aborted",
                    "scope": "session",
                    "reason": state["reason"],
                    "note": note_text(),
                    "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
                })
                self._finalize_recording("restart", discard_open_segment=True)
            # After finalize, so the session dir this abandons is on disk and
            # gets named in aborted.json.
            record_abandonment("session")
            self._close_modal()
            self.release_camera()
            message = (
                f"Session {self._session_num} abandoned — returned to Sensor "
                f"Calibration. {self._kept_sessions_phrase()}"
            )
            # D3: hand the next window the pre-Start snapshot, not the
            # incremented live value, so the retried session reuses the
            # same number -- built the toast off the live value above first.
            self._session_num = self._session_num_at_open
            self._on_back(message)

        def confirm_all() -> None:
            # D6: same abort handling as confirm_session, scope="all".
            if self._event_log is not None:
                self._event_log.append({
                    "type": "session_aborted",
                    "scope": "all",
                    "reason": state["reason"],
                    "note": note_text(),
                    "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
                })
                self._finalize_recording("restart", discard_open_segment=True)
            record_abandonment("all")
            self._close_modal()
            self.release_camera()
            message = (
                "All sessions abandoned — returned to Sensor Calibration. "
                f"The next session recorded will be session 1 of {SESSION_COUNT}."
            )
            self._session_num = 0  # D3: full reset, per Q1
            self._on_back(message)

        def build(parent: tk.Frame) -> None:
            wrap = tk.Frame(parent, bg=theme.BG)
            wrap.pack(padx=28, pady=28)
            # 560px panel, same fixed-width spacer idiom as the bookend modal.
            tk.Frame(wrap, bg=theme.BG, width=504, height=1).pack()

            tk.Label(
                wrap, text="Restart?", bg=theme.BG, fg=theme.INK, font=self._f_gate_heading,
                anchor="w", justify="left",
            ).pack(fill="x")
            tk.Label(
                wrap,
                text="Choose what to abandon. Neither option deletes data — this "
                     "can't be undone once confirmed.",
                bg=theme.BG, fg=theme.INK_70, font=self._f_gate_body, wraplength=504,
                justify="left", anchor="w",
            ).pack(fill="x", pady=(6, 20))

            # Restart Session -- costs one session, kept.
            row1 = tk.Frame(
                wrap, bg=theme.BG, highlightthickness=1,
                highlightbackground=theme.DIVIDER, highlightcolor=theme.DIVIDER,
            )
            row1.pack(fill="x", pady=(0, 12))
            row1_in = tk.Frame(row1, bg=theme.BG)
            row1_in.pack(fill="x", padx=16, pady=16)
            row1_text = tk.Frame(row1_in, bg=theme.BG)
            row1_text.pack(side="left", fill="x", expand=True, padx=(0, 12))
            tk.Label(
                row1_text, text="Restart Session", bg=theme.BG, fg=theme.INK,
                font=self._f_restart_row_heading, anchor="w",
            ).pack(fill="x")
            tk.Label(
                row1_text,
                text=(
                    f"Abandons session {self._session_num} and returns to Sensor "
                    f"Calibration. {self._kept_sessions_phrase()} The next attempt "
                    f"is still session {self._session_num} of {SESSION_COUNT}."
                ),
                bg=theme.BG, fg=theme.INK_70, font=self._f_small, wraplength=320,
                justify="left", anchor="w",
            ).pack(fill="x", pady=(4, 0))
            self._build_button(
                row1_in, "Restart Session", "secondary", None, confirm_session,
            ).pack(side="right")

            # Restart Entire Testing -- costs the whole visit, flagged accent
            # treatment so the asymmetry with the row above reads at a glance.
            row2 = tk.Frame(
                wrap, bg=theme.ACCENT_10, highlightthickness=2,
                highlightbackground=theme.ACCENT_600, highlightcolor=theme.ACCENT_600,
            )
            row2.pack(fill="x")
            row2_in = tk.Frame(row2, bg=theme.ACCENT_10)
            row2_in.pack(fill="x", padx=18, pady=18)
            row2_text = tk.Frame(row2_in, bg=theme.ACCENT_10)
            row2_text.pack(side="left", fill="x", expand=True, padx=(0, 12))
            tk.Label(
                row2_text, text="Restart Entire Testing", bg=theme.ACCENT_10,
                fg=theme.ACCENT_700, font=self._f_restart_danger_heading, anchor="w",
            ).pack(fill="x")
            tk.Label(
                row2_text,
                text=(
                    "Abandons all sessions for this participant's visit and "
                    "returns to Sensor Calibration. Nothing is deleted; every "
                    "session is kept and flagged. The next session recorded "
                    f"will be session 1 of {SESSION_COUNT}."
                ),
                bg=theme.ACCENT_10, fg=theme.ACCENT_700, font=self._f_small,
                wraplength=320, justify="left", anchor="w",
            ).pack(fill="x", pady=(4, 0))
            self._build_button(
                row2_in, "Restart Entire Testing", "danger", None, confirm_all,
            ).pack(side="right")

            tk.Frame(wrap, bg=theme.DIVIDER, height=2).pack(fill="x", pady=(20, 16))

            tk.Label(
                wrap, text="REASON", bg=theme.BG, fg=theme.INK_60, font=self._f_kicker,
                anchor="w",
            ).pack(fill="x", pady=(0, 8))
            chips_row = tk.Frame(wrap, bg=theme.BG)
            chips_row.pack(fill="x", pady=(0, 16))
            for key in ("Sensor slipped", "Camera fault", "Participant request", "Other"):
                chip = tk.Frame(
                    chips_row, bg=theme.BG, highlightthickness=1,
                    highlightbackground=theme.DIVIDER, highlightcolor=theme.DIVIDER,
                )
                chip.pack(side="left", padx=(0, 8))
                label = tk.Label(chip, text=key, bg=theme.BG, fg=theme.INK, font=self._f_small)
                label.pack(padx=12, pady=7)
                state["chips"][key] = (chip, label)
                for w in (chip, label):
                    w.bind("<Button-1>", lambda _e, k=key: on_chip_click(k))
                    w.configure(cursor="hand2")

            note_entry = tk.Entry(
                wrap, bg=theme.BG, fg=theme.INK, font=self._f_small, relief="flat",
                highlightthickness=1, highlightbackground=theme.DIVIDER,
                highlightcolor=theme.DIVIDER,
            )
            note_entry.pack(fill="x", ipady=6, pady=(0, 20))
            state["note_entry"] = note_entry
            state["note_state"] = _add_placeholder(note_entry, "Add a note (optional)")

            cancel_row = tk.Frame(wrap, bg=theme.BG)
            cancel_row.pack(fill="x")
            self._build_button(
                cancel_row, "Cancel", "ghost", None, self._close_modal,
            ).pack(side="right")

        self._open_modal(build)

    # ── toast (copied from calibration_window, not imported, so the windows
    #    stay independent) ────────────────────────────────────────────────────
    def _show_toast(self, message: str) -> None:
        if self._toast_after is not None:
            self.after_cancel(self._toast_after)
        self._toast_label.configure(text=message)
        self._reset_toast_colors()
        self._toast.place(in_=self._toast_parent, relx=1.0, x=0, y=0, anchor="ne")
        self._toast.lift()
        self._toast_after = self.after(_TOAST_TIMEOUT_MS, lambda: self._fade_toast(1))

    def _fade_toast(self, step: int) -> None:
        t = step / _FADE_STEPS
        bg = _lerp(theme.ACCENT_10, theme.BG, t)
        edge = _lerp(theme.ACCENT, theme.BG, t)
        fg = _lerp(theme.ACCENT_700, theme.BG, t)
        self._toast.configure(bg=bg, highlightbackground=edge, highlightcolor=edge)
        self._toast_inner.configure(bg=bg)
        self._toast_label.configure(bg=bg, fg=fg)
        if step < _FADE_STEPS:
            self._toast_after = self.after(_FADE_INTERVAL_MS, lambda: self._fade_toast(step + 1))
        else:
            self._toast_after = None
            self._toast.place_forget()
            self._reset_toast_colors()

    def _reset_toast_colors(self) -> None:
        self._toast.configure(
            bg=theme.ACCENT_10, highlightbackground=theme.ACCENT, highlightcolor=theme.ACCENT
        )
        self._toast_inner.configure(bg=theme.ACCENT_10)
        self._toast_label.configure(bg=theme.ACCENT_10, fg=theme.ACCENT_700)

    # ── modal (Phase 3, E4/E5) ──────────────────────────────────────────────
    def _open_modal(self, build_content: Callable[[tk.Frame], None]) -> None:
        """Blocking modal as two borderless Toplevels -- tkinter frames have
        no alpha, so the design's rgba(32,30,29,0.55) scrim can't be a single
        widget (E4). Only one modal may be open at a time; opening a second
        closes the first."""
        self._close_modal()

        # self is a mixin here at the class-definition level, but at runtime
        # it's always the composed SessionWindow (a tk.Toplevel) -- mypy
        # can't see that through ModalMixin alone, so these four Tk calls
        # need a narrow ignore (no runtime effect, same pattern the split
        # plan's D4 uses for cross-group attribute access).
        scrim = tk.Toplevel(self)  # type: ignore[arg-type]
        scrim.overrideredirect(True)
        scrim.transient(self)  # type: ignore[call-overload]
        scrim.attributes("-alpha", 0.55)
        scrim.configure(bg=theme.INK)

        panel = tk.Toplevel(self)  # type: ignore[arg-type]
        panel.overrideredirect(True)
        panel.transient(self)  # type: ignore[call-overload]
        panel.configure(
            bg=theme.BG, highlightthickness=2,
            highlightbackground=theme.INK, highlightcolor=theme.INK,
        )
        inner = tk.Frame(panel, bg=theme.BG)
        inner.pack()
        build_content(inner)

        self._modal_scrim = scrim
        self._modal_panel = panel
        self._reposition_modal()
        # Rig-feedback correction (2026-07-27): on Windows a freshly-created
        # overrideredirect Toplevel's initial geometry doesn't reliably
        # "stick" before the next event-loop tick -- on the rig this landed
        # the modal pinned at the desktop's top-left instead of centered.
        # Re-applying once more once the loop goes idle fixes it. Safe even
        # if the modal has since closed: _reposition_modal no-ops when
        # _modal_scrim/_modal_panel are None.
        self.after_idle(self._reposition_modal)
        scrim.lift()
        panel.lift()

        # The window is resizable(False, False) but can still be dragged, so
        # both Toplevels have to follow it to stay aligned over the session.
        self._modal_configure_id = self.bind(
            "<Configure>", lambda _e: self._reposition_modal(),
        )

        # Manual drag-to-move (rig-feedback correction, 2026-07-27): an
        # overrideredirect Toplevel has no OS titlebar to grab, so dragging
        # the panel is the permanent fallback regardless of whether the idle
        # re-position above also fixes the underlying race. Bound directly
        # to the panel Toplevel itself, not its content children, so
        # grabbing empty panel background moves it while clicks on the
        # buttons/entries/chips built by build_content keep their own
        # bindings untouched. Only the panel moves -- the scrim is left
        # alone here and keeps tracking the session window via the
        # <Configure> binding above, so it still covers the whole window.
        # A fresh bind per _open_modal call (panel is a new Toplevel every
        # time); destroying the Toplevel in _close_modal removes these
        # bindings with it, so there is nothing to unbind explicitly.
        panel.bind("<ButtonPress-1>", self._on_modal_drag_start)
        panel.bind("<B1-Motion>", self._on_modal_drag_motion)

        panel.grab_set()

    def _reposition_modal(self) -> None:
        if self._modal_scrim is None or self._modal_panel is None:
            return
        self.update_idletasks()
        x, y = self.winfo_rootx(), self.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        self._modal_scrim.geometry(f"{w}x{h}+{x}+{y}")

        panel = self._modal_panel
        panel.update_idletasks()
        pw, ph = panel.winfo_reqwidth(), panel.winfo_reqheight()
        px, py = x + (w - pw) // 2, y + (h - ph) // 2
        panel.geometry(f"{pw}x{ph}+{px}+{py}")

    def _on_modal_drag_start(self, event: tk.Event) -> None:
        """<ButtonPress-1> on the panel Toplevel -- event.x/y are already
        panel-relative because the binding is on the panel itself, so this
        is directly the pointer's offset from the panel's top-left."""
        self._modal_drag_offset = (event.x, event.y)

    def _on_modal_drag_motion(self, event: tk.Event) -> None:
        """<B1-Motion> on the panel Toplevel. Moves only the panel -- the
        scrim is meant to keep covering the whole session window and stays
        aligned to it via the <Configure> binding in _open_modal, not via
        drag. Guarded the same way _reposition_modal is, in case a queued
        motion event still lands after the modal has closed mid-drag."""
        if self._modal_panel is None:
            return
        offset_x, offset_y = self._modal_drag_offset
        x = event.x_root - offset_x
        y = event.y_root - offset_y
        self._modal_panel.geometry(f"+{x}+{y}")

    def _close_modal(self) -> None:
        """Safe to call when no modal is open. Called unconditionally from
        both _on_close and _handle_back (E5) so a modal held open at window
        teardown can never leave a dangling grab."""
        if self._modal_configure_id is not None:
            self.unbind("<Configure>", self._modal_configure_id)
            self._modal_configure_id = None
        if self._modal_panel is not None:
            self._modal_panel.grab_release()
            self._modal_panel.destroy()
            self._modal_panel = None
        if self._modal_scrim is not None:
            self._modal_scrim.destroy()
            self._modal_scrim = None


def _add_placeholder(entry: tk.Entry, text: str) -> dict:
    """Grey placeholder that clears on focus and restores while empty.

    Copied from new_participant_window.py, not imported -- see the
    _build_button section's comment on this file's independence convention.
    Returns the mutable state dict so callers can tell whether the field is
    still showing placeholder text (i.e. effectively empty) rather than input.
    """
    state = {"placeholder": True}

    def show() -> None:
        entry.delete(0, "end")
        entry.insert(0, text)
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


def _lerp(from_hex: str, to_hex: str, t: float) -> str:
    a = tuple(int(from_hex[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(to_hex[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
