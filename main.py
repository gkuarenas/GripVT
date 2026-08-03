"""Entry point for the GripVT landing window."""
from output import participants
from ui.calibration_window import CalibrationWindow
from ui.landing_window import LandingWindow
from ui.new_participant_window import NewParticipantWindow
from ui.resume_session_window import ResumeSessionWindow
from ui.session import SessionWindow

RECAL_TOAST = "Sensor remounted — recalibrate before the next session."


def _should_invalidate(message: str | None, recording_started: bool) -> bool:
    """F5/D2: one wear = one calibration.

    Either Restart scope always invalidates (message is not None -- both
    confirm_session and confirm_all in ui/session/modals.py always build a
    message before calling on_back). A bare Back carries no message, so it
    only invalidates if Start was ever confirmed for that session -- a bare
    Back before Start ended no wear, and forcing a redo would be pointless
    friction (D2's most regression-prone case)."""
    return message is not None or recording_started


def main():
    holder = {}

    def open_session(cal_win, record, result, pending_calibration):
        app = holder["app"]
        holder["participant"] = record  # consumed by the session recording pass
        # The camera at CAMERA_INDEX is exclusive, so calibration must let go of
        # it before the session window tries to open it.
        cal_win.release_camera()
        cal_win.withdraw()

        def restore_cal(message: str | None = None):
            # D3: read before destroy() -- session_win is still alive here.
            holder["session_num"] = session_win._session_num
            # D9: propagate forward whichever way the round trip goes -- if
            # Start consumed the supersede, the attribute is already None on
            # session_win; if the operator backed out before Start, the
            # untouched name flows back to the holder and remains
            # supersedable on the next SessionWindow.
            holder["pending_supersede"] = session_win._pending_supersede_session
            recording_started = session_win._recording_started
            session_win.destroy()
            cal_win.deiconify()
            cal_win.retry_camera()
            if _should_invalidate(message, recording_started):
                # F5/D2: this wear ended (a recorded session or either
                # Restart) -- force a fresh calibration before Continue.
                # message is not None on both Restart paths (D2), so this
                # always fires there; recording_started covers the ordinary
                # Back after a confirmed Start, which carries no message.
                cal_win.invalidate()
                cal_win.show_toast(f"{message} {RECAL_TOAST}" if message else RECAL_TOAST)

        session_num = holder["session_num"]
        pending_supersede = holder["pending_supersede"]
        session_win = SessionWindow(
            app, on_back=restore_cal, detector_params=result.params,
            calibration_result=result, participant=record,
            session_num=session_num, pending_calibration=pending_calibration,
            pending_supersede_session=pending_supersede,
        )
        # Titlebar X quits the whole program, same as every other window in
        # this flow -- but unlike New Participant/Calibration, Session may
        # have an open recording. This overrides SessionWindow's own
        # _on_close (session_window.py) since Tk keeps only one handler per
        # protocol name, so finalize has to happen here explicitly or an
        # in-progress recording's trials.csv/events.json/overlay.png are
        # lost silently (rig-found 2026-07-28: closing via X left only
        # markers.csv on disk).
        session_win.protocol(
            "WM_DELETE_WINDOW",
            lambda: (
                session_win._finalize_recording("closed"),
                session_win.release_camera(),
                app.destroy(),
            ),
        )

    def open_calibration(np_win, record):
        app = holder["app"]
        # D8: computed once here for every caller (New Participant and
        # Resume alike) -- a fresh participant folder naturally returns
        # 0 / None, so New Participant is unaffected while gaining the
        # correct general computation.
        holder["session_num"] = participants.count_complete_sessions(record.folder)
        holder["pending_supersede"] = participants.trailing_incomplete_session(record.folder)
        np_win.withdraw()

        def restore_np():
            cal_win.destroy()
            np_win.deiconify()

        cal_win = CalibrationWindow(
            app, record,
            on_back=restore_np,
            on_continue=lambda result, pending: open_session(
                cal_win, record, result, pending
            ),
        )
        cal_win.protocol(
            "WM_DELETE_WINDOW", lambda: (cal_win.release_camera(), app.destroy())
        )

    def open_new_participant():
        app = holder["app"]
        app.withdraw()

        def restore():
            win.destroy()
            app.deiconify()

        # Back returns to landing; the titlebar X quits the whole program.
        win = NewParticipantWindow(
            app, on_back=restore, on_save=lambda rec: open_calibration(win, rec)
        )
        win.protocol("WM_DELETE_WINDOW", app.destroy)

    def open_resume_session():
        app = holder["app"]
        app.withdraw()

        def restore():
            win.destroy()
            app.deiconify()

        # Back returns to landing; the titlebar X quits the whole program.
        win = ResumeSessionWindow(
            app, on_back=restore, on_continue=lambda rec: open_calibration(win, rec)
        )
        win.protocol("WM_DELETE_WINDOW", app.destroy)

    app = LandingWindow(
        on_new_participant=open_new_participant, on_resume_session=open_resume_session
    )
    holder["app"] = app
    app.mainloop()


if __name__ == "__main__":
    main()
