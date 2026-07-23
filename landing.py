"""Entry point for the GripVT landing window."""
from ui.landing_window import LandingWindow


def main():
    app = LandingWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
