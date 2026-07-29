"""Entry point for the GripVT landing window."""
from ui.calibration_window import CalibrationWindow
from ui.landing_window import LandingWindow
from ui.new_participant_window import NewParticipantWindow
from ui.session import SessionWindow


def main():
    holder = {}

    def open_session(cal_win, record, result):
        app = holder["app"]
        holder["participant"] = record  # consumed by the session recording pass
        # The camera at CAMERA_INDEX is exclusive, so calibration must let go of
        # it before the session window tries to open it.
        cal_win.release_camera()
        cal_win.withdraw()

        def restore_cal(message: str | None = None):
            # D3: read before destroy() -- session_win is still alive here.
            holder["session_num"] = session_win._session_num
            session_win.destroy()
            cal_win.deiconify()
            cal_win.retry_camera()
            if message is not None:
                cal_win.show_toast(message)

        session_num = holder["session_num"]
        session_win = SessionWindow(
            app, on_back=restore_cal, detector_params=result.params,
            calibration_result=result, participant=record,
            session_num=session_num,
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
        holder["session_num"] = 0  # per-visit counter, seeded once (D3)
        np_win.withdraw()

        def restore_np():
            cal_win.destroy()
            np_win.deiconify()

        cal_win = CalibrationWindow(
            app, record,
            on_back=restore_np,
            on_continue=lambda result: open_session(cal_win, record, result),
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

    app = LandingWindow(on_new_participant=open_new_participant)
    holder["app"] = app
    app.mainloop()


if __name__ == "__main__":
    main()
