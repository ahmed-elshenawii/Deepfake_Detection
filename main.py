"""
DeepGuard AI — Entry Point
============================
Launch the DeepGuard AI desktop application.

Usage:
    python main.py
"""

import os
import sys

# ── Ensure project root is on sys.path ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Dependency check ──────────────────────────────────────
def _check_deps():
    missing = []
    for pkg in ["customtkinter", "torch", "PIL", "transformers"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌  Missing dependencies:", ", ".join(missing))
        print("   Install with:  pip install customtkinter torch torchvision "
              "transformers Pillow")
        sys.exit(1)

_check_deps()

# ── Set appearance before importing the app ────────────────
import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Launch ─────────────────────────────────────────────────
from gui.app import DeepGuardApp


def main():
    os.chdir(PROJECT_ROOT)
    app = DeepGuardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
