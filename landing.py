"""Entry point for the GripVT landing window."""
from ui.calibration_window import CalibrationWindow
from ui.landing_window import LandingWindow
from ui.new_participant_window import NewParticipantWindow
from ui.session_window import SessionWindow


def main():
    holder = {}

    def open_session(cal_win, record, params):
        app = holder["app"]
        holder["participant"] = record  # consumed by the session recording pass
        # The camera at CAMERA_INDEX is exclusive, so calibration must let go of
        # it before the session window tries to open it.
        cal_win.release_camera()
        cal_win.withdraw()

        def restore_cal():
            session_win.destroy()
            cal_win.deiconify()
            cal_win.retry_camera()

        session_win = SessionWindow(app, on_back=restore_cal, detector_params=params)
        session_win.protocol(
            "WM_DELETE_WINDOW", lambda: (session_win.release_camera(), app.destroy())
        )

    def open_calibration(np_win, record):
        app = holder["app"]
        np_win.withdraw()

        def restore_np():
            cal_win.destroy()
            np_win.deiconify()

        cal_win = CalibrationWindow(
            app, record,
            on_back=restore_np,
            on_continue=lambda params: open_session(cal_win, record, params),
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
