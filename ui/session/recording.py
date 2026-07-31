"""RecordingMixin -- trial start/stop/undo, bookend confirm modals, CSV
finalize, and the REC timer for the Session window.

Part of the ui/session/ package (see ui/session/window.py's module docstring
for the package overview). Split out of the former ui/session_window.py
verbatim (mechanical move, no logic changes) -- see
docs/logs/20260729_141951_codebase_reorg_and_session_split_plan.md, D8.

CSV recording, trial sequencing, and target-force values are real. A
confirmed Start generates and locks a 30-trial randomized %MVC sequence
(core/trial_sequence.py), persists it to sequence.json, and drives the
TRIAL kicker and TARGET FORCE block off sequence[trial_index - 1] --
self._trial is the sole index into that sequence, so Next Trial / Undo
(E7) can never desync from it (see D2 in the trial-sequencing plan). Both
also open/close/void real segments in trials.csv via TrialLog, alongside
markers.csv (per-frame) and events.json (bookend/annotation) writers.

D7 (the codebase-reorg-session-split plan): draw_overlay is imported here,
not referenced through another module, because _finalize_recording (the
rig-found data-loss regression guard's subject) lives in this file --
tests must patch ui.session.recording.draw_overlay, not the old
ui.session_window.draw_overlay.
"""
import logging
import os
import time
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from typing import Callable

import cv2

from core.calibration import CameraCalibration
from core.tracker import MarkerRecord, Tracker
from core.trial_sequence import (
    SequenceGenerationError, TrialSequence, generate_sequence, target_kg,
)
from output.participants import ParticipantRecord
from output.video import VIDEO_EXT, VIDEO_FPS, VideoLogWriter
from output.writer import (
    CSVWriter, EventLog, TrialLog, anchor_wall_clock, make_session_dir, now_ms, save_sequence,
)
from ui import theme
from ui.overlay import draw_overlay

from .camera_feed import RECORDING_TRACK_EVERY

_log = logging.getLogger(__name__)

# D7: force-commit buffered markers.csv rows before they grow past this many
# buffered frames (~60s at 30fps), so an unbroken recording with no Next
# Trial press can't hold an unbounded amount of uncommitted data in RAM.
MAX_BUFFER_FRAMES = 1800


class RecordingMixin:
    """Trial start/stop/undo, bookend modals, CSV finalize, REC timer.
    Composed into SessionWindow alongside the other ui/session/ mixins --
    see window.py for the single __init__ that owns all of the attributes
    declared below (D1/D3)."""

    # ── Cross-group attributes (D4): owned/assigned in window.py's __init__
    # -- declared here only so mypy resolves self.<attr> access within this
    # mixin. No runtime effect.
    _recording: bool
    _recording_started: bool
    _trial: int
    _trial_max: int
    _trial_log: TrialLog | None
    _writer: CSVWriter | None
    _video: VideoLogWriter | None
    _cal: CameraCalibration
    _segment_id: int
    _t0_wall_ms: float
    _t0_monotonic: float
    _undo_available: bool
    _sequence: TrialSequence | None
    _participant: ParticipantRecord | None
    _session_num: int
    _strip_recapture_btn: tk.Frame | None
    _event_log: EventLog | None
    _tracker: Tracker
    _open_camera_loss_event: dict | None
    _last_records: list[MarkerRecord] | None
    _rec_paused_badge: tk.Frame
    _tracked_badge: tk.Frame
    _rec_badge: tk.Frame
    _rec_label: tk.Label
    _rec_after_id: str | None
    _rec_segment_start: float | None
    _rec_elapsed: float
    _timer_paused: bool
    _paused_banner: tk.Frame
    _f_bookend_heading: tkfont.Font
    _f_gate_body: tkfont.Font
    _f_small: tkfont.Font

    # Cross-group methods (D2): plain self.method() calls into the progress,
    # baseline, and modal groups, declared the same way for mypy.
    _target_pct_for_trial: Callable[[], int | None]
    _target_kg_for_trial: Callable[[], float | None]
    _refresh_trial_block: Callable[[], None]
    _refresh_target_block: Callable[[], None]
    _refresh_session_block: Callable[[], None]
    _refresh_transport: Callable[[], None]
    _locked: Callable[[], bool]
    _open_modal: Callable[..., None]
    _close_modal: Callable[[], None]
    _show_toast: Callable[[str], None]
    _build_button: Callable[..., tk.Frame]

    # This mixin is composed onto tk.Toplevel (window.py's class statement),
    # not declared here -- inherited, not cross-group per se, but needs the
    # same mypy-only declaration since RecordingMixin itself has no base
    # class.
    after: Callable[..., str]
    after_cancel: Callable[..., None]

    # ── stub handlers ───────────────────────────────────────────────────────
    def _on_start(self) -> None:
        # Mirrors _refresh_transport's startDisabled rule -- guards against a
        # stray call (e.g. a key binding) reaching a disabled button.
        start_disabled = not self._locked() or self._recording
        if start_disabled:
            return
        self._open_start_bookend_modal()

    def _on_stop(self) -> None:
        stop_disabled = not self._recording
        if stop_disabled:
            return
        self._open_stop_bookend_modal()

    def _on_next_trial(self) -> None:
        # Mirrors _refresh_transport's nextTrialDisabled rule (Phase 8, E7,
        # recording-independent per the rig-feedback correction) -- guards a
        # stray call the same way _on_start/_on_stop already do. D9: also
        # gated on self._recording (see _refresh_transport's 2026-07-28
        # correction) -- a pre-Start call would advance _trial against a
        # sequence that doesn't exist yet (self._sequence is None).
        next_trial_disabled = (
            not self._locked() or not self._recording or self._trial >= self._trial_max
        )
        if next_trial_disabled:
            return
        if self._trial_log is not None:
            # D9: persist the trial boundary only while a recording is open
            # -- guard on _trial_log, not self._recording, so the two can
            # never disagree. Commits the segment that's ending, then opens
            # the next one under a fresh, never-reused segment_id (D2).
            assert self._writer is not None
            self._writer.write_window()
            self._trial_log.close_trial(
                self._segment_id, now_ms(self._t0_wall_ms, self._t0_monotonic),
            )
            self._segment_id += 1
        self._trial = min(self._trial_max, self._trial + 1)
        if self._trial_log is not None:
            self._trial_log.open_trial(
                trial_id=self._segment_id, trial_index=self._trial,
                start_ts=now_ms(self._t0_wall_ms, self._t0_monotonic),
                target_pct=self._target_pct_for_trial(),
                target_kg=self._target_kg_for_trial(),
            )
        self._undo_available = True
        self._refresh_trial_block()
        self._refresh_target_block()
        self._refresh_transport()

    def _on_undo_trial(self) -> None:
        if not self._undo_available:
            return
        if self._trial_log is not None:
            # D2 tombstone: void the segment Next Trial just closed and
            # committed (self._segment_id - 1) -- its markers.csv rows stay
            # on disk untouched, only trials.csv's voided flag flips.
            assert self._writer is not None
            self._trial_log.void_trial(self._segment_id - 1)
            # The just-opened segment (self._segment_id) is still in
            # progress -- discard whatever it has buffered so far (could be
            # zero or a few frames) and re-open it fresh for the redo.
            # open_trial() only stages a row in memory (trials.csv is only
            # written by flush()), so overwriting that entry here leaves no
            # "completed" row behind for a segment that never happened.
            # self._segment_id itself is never decremented (D2) -- only
            # self._trial, the displayed counter, moves back. The voided
            # segment's trial_index was set to self._trial - 1 when the
            # Next Trial that just closed it opened it; redo_trial_index
            # below is computed the same way from the same (still
            # undecremented) self._trial. Both therefore carry the
            # identical trial_index and resolve to the same
            # sequence[trial_index - 1] target (D2) -- Undo re-shows
            # exactly what it just left.
            self._writer.discard_window()
            redo_trial_index = max(1, self._trial - 1)
            # self._trial is not decremented until below, so the bare
            # _target_pct_for_trial()/_target_kg_for_trial() helpers would
            # read the pre-Undo trial here, not redo_trial_index -- index
            # the sequence directly for redo_trial_index instead.
            redo_target_pct = (
                self._sequence.levels_pct[redo_trial_index - 1]
                if self._sequence is not None else None
            )
            redo_target_kg = (
                target_kg(redo_target_pct, self._participant.mvc_kg)
                if redo_target_pct is not None and self._participant is not None
                else None
            )
            self._trial_log.open_trial(
                trial_id=self._segment_id, trial_index=redo_trial_index,
                start_ts=now_ms(self._t0_wall_ms, self._t0_monotonic),
                target_pct=redo_target_pct,
                target_kg=redo_target_kg,
            )
        self._trial = max(1, self._trial - 1)
        self._undo_available = False
        self._refresh_trial_block()
        self._refresh_target_block()
        self._refresh_transport()

    # ── bookend confirm modals (Phase 5) ────────────────────────────────────
    # The two modals instruct OPPOSITE tap orders, and that asymmetry is real:
    # _confirm_start opens the writers (nothing is recorded before Confirm), while
    # _confirm_stop closes them (recording is still live while the modal is up).
    # The original Start copy said "before recording begins", which is literally
    # backwards -- following it put the taps outside the camera span entirely.
    # That is the confirmed root cause of participant001/session1 (2026-07-30)
    # having no usable start bookend: Biopac saw taps at t~3.5-5.5s, the camera's
    # first frame was t~9.65s. Keep both bodies explicit about which side of
    # Confirm the taps belong on.
    def _open_start_bookend_modal(self) -> None:
        def build(parent: tk.Frame) -> None:
            self._build_bookend_content(
                parent,
                title="Confirm Start Bookend",
                body="Recording starts when you press Confirm. Perform the "
                     "triple-tap bookend immediately AFTER confirming — taps "
                     "before Confirm are not recorded.",
                on_confirm=self._confirm_start,
            )

        self._open_modal(build)

    def _open_stop_bookend_modal(self) -> None:
        def build(parent: tk.Frame) -> None:
            self._build_bookend_content(
                parent,
                title="Confirm End Bookend",
                body="Perform the closing triple-tap bookend NOW, while recording "
                     "is still running. Press Confirm only after the taps — "
                     "confirming ends the recording.",
                on_confirm=self._confirm_stop,
            )

        self._open_modal(build)

    def _build_bookend_content(
        self, parent: tk.Frame, title: str, body: str, on_confirm: Callable[[], None],
    ) -> None:
        wrap = tk.Frame(parent, bg=theme.BG)
        wrap.pack(padx=28, pady=24)
        # 420px panel -- forced via a zero-height spacer, the same idiom the
        # topbar uses for its fixed-width Restart column.
        tk.Frame(wrap, bg=theme.BG, width=364, height=1).pack()

        tk.Label(
            wrap, text=title, bg=theme.BG, fg=theme.INK, font=self._f_bookend_heading,
            anchor="w", justify="left",
        ).pack(fill="x", pady=(4, 12))
        tk.Label(
            wrap, text=body, bg=theme.BG, fg=theme.INK_70, font=self._f_gate_body,
            wraplength=364, justify="left", anchor="w",
        ).pack(fill="x")
        tk.Label(
            wrap,
            text="A missed bookend makes this session's force labels unrecoverable.",
            bg=theme.BG, fg=theme.INK_70, font=self._f_small, wraplength=364,
            justify="left", anchor="w",
        ).pack(fill="x", pady=(8, 20))

        btn_row = tk.Frame(wrap, bg=theme.BG)
        btn_row.pack(fill="x")
        self._build_button(btn_row, "Confirm", "primary", None, on_confirm).pack(side="right")

    def _confirm_start(self) -> None:
        self._close_modal()
        if self._participant is not None and self._participant.mvc_kg <= 0:
            self._show_toast("Participant MVC is 0 kg — re-enter MVC before recording")
            return
        try:
            seq = generate_sequence()
        except SequenceGenerationError:
            self._show_toast("Could not generate a trial sequence — Start again")
            return
        self._sequence = seq
        self._trial = 1
        self._session_num += 1
        self._refresh_session_block()
        self._undo_available = False
        self._recording = True
        self._recording_started = True
        self._show_rec_badge()
        if self._strip_recapture_btn is not None:
            self._strip_recapture_btn.set_enabled(False)
        self._start_timer()
        self._refresh_transport()
        self._refresh_trial_block()
        self._refresh_target_block()
        print("Start clicked")

        # D5/D8: degrade to a no-op, mirroring _save_baseline_record's guard
        # (self._participant is None is the headless/test-fixture path) --
        # never crash, never write to the project root.
        if self._participant is None:
            return
        session_dir = make_session_dir(self._participant.folder)
        self._writer = CSVWriter(session_dir)
        self._trial_log = TrialLog(session_dir)
        self._event_log = EventLog(
            session_dir,
            participant_id=self._participant.participant_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
            frame_index_at_start=self._tracker.frame_index,
        )
        self._t0_wall_ms, self._t0_monotonic = anchor_wall_clock()
        try:
            self._video = VideoLogWriter(
                os.path.join(session_dir, f"raw_video{VIDEO_EXT}"),
                size=self._cal.image_size, fps=VIDEO_FPS,
            )
        except Exception as e:
            self._video = None
            self._event_log.append({
                "type": "video_log_failed",
                "error": str(e),
                "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
            })
            self._show_toast("Video log failed to start — recording without raw video.")
        self._segment_id = 0
        save_sequence(
            session_dir, participant_id=self._participant.participant_id,
            sequence=seq, mvc_kg=self._participant.mvc_kg,
        )
        self._trial_log.open_trial(
            trial_id=self._segment_id, trial_index=self._trial,
            start_ts=now_ms(self._t0_wall_ms, self._t0_monotonic),
            target_pct=self._target_pct_for_trial(),
            target_kg=self._target_kg_for_trial(),
        )
        if RECORDING_TRACK_EVERY != 1:
            # D3 approved addition: a static config check, not a dynamic
            # heuristic -- fires once, here, if RECORDING_TRACK_EVERY is ever
            # manually bumped after a rig test shows 30Hz drops frames.
            self._event_log.append({
                "type": "rate_fallback",
                "target_hz": 30 / RECORDING_TRACK_EVERY,
                "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
            })

    def _confirm_stop(self) -> None:
        self._close_modal()
        self._recording = False
        self._hide_rec_badges()
        # Defensive: a Stop while disconnected is not a designed flow, but
        # leaving the paused banner up with no recording in progress would
        # be a stale, misleading state.
        self._paused_banner.pack_forget()
        self._stop_timer()
        self._refresh_transport()
        print("Stop clicked")
        self._finalize_recording("stopped")

    def _buffer_recording_frame(self, records: list[MarkerRecord]) -> None:
        """D7: appends one frame to the CSVWriter's in-RAM buffer and force-
        commits early if it has grown past MAX_BUFFER_FRAMES, so a long
        unbroken recording can't hold unbounded uncommitted data."""
        assert self._writer is not None and self._event_log is not None
        ts = now_ms(self._t0_wall_ms, self._t0_monotonic)
        self._writer.buffer_frame(records, self._tracker.frame_index, ts, trial_id=self._segment_id)
        if self._writer.pending_count >= MAX_BUFFER_FRAMES:
            self._writer.write_window()
            self._event_log.append({
                "type": "buffer_flushed",
                "frame_count": MAX_BUFFER_FRAMES,
                "ts": now_ms(self._t0_wall_ms, self._t0_monotonic),
            })

    def _finalize_recording(self, reason: str, discard_open_segment: bool = False) -> None:
        """General-purpose recording finalizer (D8) -- Phase 3's only caller
        is _confirm_stop, always with the default (commit, not discard).
        Later phases (camera-loss hooks, exit paths, Restart) call this too,
        some with discard_open_segment=True; `reason` is accepted now so
        those callers don't need a second finalize signature -- Phase 3 does
        not itself log it anywhere.
        """
        if self._writer is None:
            return
        assert self._trial_log is not None and self._event_log is not None
        if self._open_camera_loss_event is not None:
            self._open_camera_loss_event["end_ts"] = now_ms(self._t0_wall_ms, self._t0_monotonic)
            self._open_camera_loss_event = None
        self._trial_log.close_trial(self._segment_id, now_ms(self._t0_wall_ms, self._t0_monotonic))
        if discard_open_segment:
            self._writer.discard_window()
            # Rig-found 2026-07-28: without this, a Restart-discarded segment
            # still showed up in trials.csv as a real, non-voided, completed
            # trial (valid start_ts/end_ts) despite having zero markers.csv
            # rows -- indistinguishable from genuine (empty) data.
            self._trial_log.void_trial(self._segment_id)
        else:
            self._writer.write_window()

        if self._last_records is not None and self._tracker.last_undistorted is not None:
            # Best-effort visualization -- must never block the data-integrity
            # flushes below it. A failure here previously left trials.csv and
            # events.json unwritten even though markers.csv had already
            # committed (found via rig verification 2026-07-28: closing the
            # window mid-recording produced markers.csv alone).
            try:
                overlay = draw_overlay(
                    self._tracker.last_undistorted, self._last_records, True,
                    self._tracker.frame_index,
                )
                cv2.imwrite(self._writer.png_path, overlay)
            except Exception:
                _log.exception("overlay.png write failed during _finalize_recording")

        if self._video is not None:
            event = {
                "type": "video_log",
                "path": os.path.basename(self._video.path),
                "frame_count": self._video.frame_count,
            }
            if self._video.first_ts_ms is not None and self._video.last_ts_ms is not None:
                real_duration_ms = self._video.last_ts_ms - self._video.first_ts_ms
                event["real_duration_ms"] = real_duration_ms
                if real_duration_ms > 0:
                    event["achieved_fps"] = self._video.frame_count / (real_duration_ms / 1000)
            self._event_log.append(event)
            self._video.close()
            self._video = None

        self._trial_log.flush()
        self._event_log.flush()
        self._writer.close()
        self._writer = None
        self._trial_log = None
        self._event_log = None

    # ── REC / REC PAUSED / TRACKED badge helpers (Phase 6/7) ────────────────
    def _show_rec_badge(self) -> None:
        """Packs the live REC timer badge, swapping out REC PAUSED if it was
        showing. `before=` keeps REC ahead of the TRACKED badge in the
        container's left-to-right order even if TRACKED was packed first
        (it is packed once, on first tier computation, and never unpacked)."""
        self._rec_paused_badge.pack_forget()
        kwargs = {"side": "left"}
        if self._tracked_badge.winfo_manager() == "pack":
            kwargs["before"] = self._tracked_badge
        self._rec_badge.pack(**kwargs)

    def _show_rec_paused_badge(self) -> None:
        self._rec_badge.pack_forget()
        kwargs = {"side": "left"}
        if self._tracked_badge.winfo_manager() == "pack":
            kwargs["before"] = self._tracked_badge
        self._rec_paused_badge.pack(**kwargs)

    def _hide_rec_badges(self) -> None:
        self._rec_badge.pack_forget()
        self._rec_paused_badge.pack_forget()

    # ── REC timer (Phase 5, E6) ──────────────────────────────────────────────
    def _start_timer(self) -> None:
        """Begin the 1s after-tick from the Start confirm. Fresh state every
        Start -- this is a wall-clock display, not tied to CSV frame
        timestamps (see the module docstring)."""
        self._rec_elapsed = 0.0
        self._rec_segment_start = time.monotonic()
        self._timer_paused = False
        self._tick_timer()

    def _tick_timer(self) -> None:
        elapsed = self._rec_elapsed
        if self._rec_segment_start is not None:
            elapsed += time.monotonic() - self._rec_segment_start
        minutes, seconds = divmod(int(elapsed), 60)
        self._rec_label.configure(text=f"REC {minutes:02d}:{seconds:02d}")
        self._rec_after_id = self.after(1000, self._tick_timer)

    def _pause_timer(self) -> None:
        """Freezes the displayed elapsed time. Phase 6 calls this the moment
        camera loss is confirmed, so the dead gap is never counted."""
        if self._timer_paused or self._rec_segment_start is None:
            return
        self._rec_elapsed += time.monotonic() - self._rec_segment_start
        self._rec_segment_start = None
        self._timer_paused = True

    def _resume_timer(self) -> None:
        """Re-anchors the running segment. Phase 6 calls this on reconnect."""
        if not self._timer_paused:
            return
        self._rec_segment_start = time.monotonic()
        self._timer_paused = False

    def _stop_timer(self) -> None:
        if self._rec_after_id is not None:
            self.after_cancel(self._rec_after_id)
            self._rec_after_id = None
        self._rec_segment_start = None
        self._timer_paused = False
        self._rec_elapsed = 0.0
        self._rec_label.configure(text="REC 00:00")
