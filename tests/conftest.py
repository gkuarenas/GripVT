"""Shared Tk root for the GUI-window test modules.

Session-scoped: exactly one tk.Tk() is created for the whole pytest run.
test_session_baseline_gate.py and test_session_error_handling.py each used to
create their own module-scoped tk.Tk() -- fine in isolation, but running both
in one process intermittently made the second module's tk.Tk() call raise
TclError, which each file's fixture swallowed into pytest.skip(), silently
skipping close to half the suite on an unlucky run. A single process-wide
root removes the destroy-then-create sequence between modules entirely.
"""
import tkinter as tk

import pytest


@pytest.fixture(scope="session")
def tk_root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no Tk display available")
    r.withdraw()
    yield r
    r.destroy()
