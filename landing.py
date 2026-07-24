"""Entry point for the GripVT landing window."""
from ui.landing_window import LandingWindow
from ui.new_participant_window import NewParticipantWindow
from ui.session_window import SessionWindow


def main():
    holder = {}

    def open_session(np_win):
        app = holder["app"]
        np_win.withdraw()

        def restore_np():
            session_win.destroy()
            np_win.deiconify()

        # Back returns to New Participant; the titlebar X quits the whole
        # program, but must release the camera first -- see SessionWindow's
        # own default protocol handler for the same ordering.
        session_win = SessionWindow(app, on_back=restore_np)
        session_win.protocol(
            "WM_DELETE_WINDOW", lambda: (session_win.release_camera(), app.destroy())
        )

    def open_new_participant():
        app = holder["app"]
        app.withdraw()

        def restore():
            win.destroy()
            app.deiconify()

        # Back returns to landing; the titlebar X quits the whole program.
        win = NewParticipantWindow(app, on_back=restore, on_save=lambda: open_session(win))
        win.protocol("WM_DELETE_WINDOW", app.destroy)

    app = LandingWindow(on_new_participant=open_new_participant)
    holder["app"] = app
    app.mainloop()


if __name__ == "__main__":
    main()
