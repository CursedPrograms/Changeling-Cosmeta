"""Changeling Cosmetics: makeup, hair colour and length, beards, on two layers.

    python app.py              the window
    python app.py photo.jpg    the window, with this photo open
"""
import sys


def main():
    from cosmetics.parse import Parser
    from window import Window

    Window(Parser).run(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
