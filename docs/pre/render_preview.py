"""Render the local deck with Chrome or Edge and optionally print the same HTML to PDF."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image


def browser_executable():
    candidates = [
        os.environ.get("CHROME"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for location in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        if location in os.environ:
            base = Path(os.environ[location])
            candidates.extend([base / "Google/Chrome/Application/chrome.exe",
                               base / "Microsoft/Edge/Application/msedge.exe"])
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise SystemExit("Set CHROME to the path of Chrome or Edge.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path)
    parser.add_argument("pages", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("scale", type=int, nargs="?", default=1)
    parser.add_argument("--pdf", type=Path, help="Also export a PDF with one slide per page.")
    args = parser.parse_args()
    if not args.deck.is_file() or args.pages < 1 or args.scale < 1:
        parser.error("Use an existing HTML file and positive page count and scale.")
    browser = browser_executable()
    args.output.mkdir(parents=True, exist_ok=True)
    columns = min(4, args.pages)
    rows = (args.pages + columns - 1) // columns
    width, height, gap = 480, 270, 10
    sheet = Image.new("RGB", (columns * (width + gap) + gap, rows * (height + gap) + gap), "#cbd0d1")
    with tempfile.TemporaryDirectory(prefix="delelstm_slides_", ignore_cleanup_errors=True) as profile:
        # A separate profile keeps rendering out of the user's browser session.
        chrome = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                  "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
                  f"--user-data-dir={profile}", "--window-size=1920,1080", "--virtual-time-budget=5000"]
        deck_url = args.deck.resolve().as_uri()
        for number in range(1, args.pages + 1):
            fresh = Path(profile) / f"page-{number:02d}.png"
            command = [*chrome, f"--force-device-scale-factor={args.scale}", f"--screenshot={fresh}",
                       deck_url + f"#{number}"]
            result = subprocess.run(command, capture_output=True, timeout=60)
            if result.returncode or not fresh.is_file():
                raise RuntimeError(f"Slide {number} did not render. " + result.stderr.decode(errors="replace"))
            with Image.open(fresh) as image:
                if image.size != (1920 * args.scale, 1080 * args.scale):
                    raise ValueError(f"Unexpected screenshot dimensions for slide {number}: {image.size}")
                tile = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            row, column = divmod(number - 1, columns)
            sheet.paste(tile, (gap + column * (width + gap), gap + row * (height + gap)))
            shutil.copyfile(fresh, args.output / f"slide-{number:02d}.png")
        if args.pdf:
            from pypdf import PdfWriter

            printed = Path(profile) / "slides.pdf"
            command = [*chrome, "--no-pdf-header-footer", f"--print-to-pdf={printed}", deck_url]
            result = subprocess.run(command, capture_output=True, timeout=60)
            if result.returncode or not printed.is_file() or not printed.stat().st_size:
                raise RuntimeError("PDF export failed. " + result.stderr.decode(errors="replace"))
            # Keep the browser's tagged pages and omit the export dates from PDF properties.
            without_dates = Path(profile) / "presentation.pdf"
            writer = PdfWriter(clone_from=printed)
            writer.metadata = {k: v for k, v in (writer.metadata or {}).items()
                               if k not in ("/CreationDate", "/ModDate")}
            writer.root_object.pop("/Metadata", None)
            writer.write(without_dates)
            writer.close()
            args.pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(without_dates, args.pdf)
            print(f"Exported PDF to {args.pdf}")
    sheet.save(args.output / "contact.png")
    print(f"Rendered {args.pages} slides to {args.output}")


if __name__ == "__main__":
    main()
