"""Inline ui/data.json into ui/template.html -> ui/fly_bar.html (one self-contained page).

usage: python ui/build_ui.py [data.json]
"""
import sys
from pathlib import Path

UI = Path(__file__).parent


def main():
    data = (UI / (sys.argv[1] if len(sys.argv) > 1 else "data.json")).read_text(encoding="utf-8")
    page = (UI / "template.html").read_text(encoding="utf-8").replace("__DATA__", data.replace("</", "<\\/"))
    (UI / "fly_bar.html").write_text(page, encoding="utf-8")
    print(f"wrote {UI / 'fly_bar.html'} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
