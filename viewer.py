"""Entry point for the GripVT viewer app."""
from ui.viewer_window import ViewerWindow


def main():
    app = ViewerWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
