"""Fetch the pinned upstream code/data and apply the course-project patch."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches" / "upstream.json"
PATCH = ROOT / "patches" / "reproduction.patch"


def verify(directory, manifest):
    for name, expected in manifest["patched_sha256_lf"].items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"Missing upstream file: {path}")
        actual = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source differs from the recorded patched version: {path}")
    for dataset in ("Electricity", "PM2.5", "Exchange"):
        for name in ("newX_train.csv", "second_y.csv"):
            if not (directory / "DATA" / dataset / name).is_file():
                raise RuntimeError(f"Missing processed data: {dataset}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing ZIP from the pinned upstream commit")
    parser.add_argument("--destination", type=Path, default=ROOT / "vendor" / "DeLELSTM")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    if destination.exists():
        verify(destination, manifest)
        print(f"Already prepared and verified: {destination}")
        return
    if not shutil.which("git"):
        parser.error("Git is required to apply patches/reproduction.patch")

    with tempfile.TemporaryDirectory(prefix="delelstm-upstream-") as temporary:
        temporary = Path(temporary)
        archive = args.archive.resolve() if args.archive else temporary / "upstream.zip"
        if not args.archive:
            url = f"https://codeload.github.com/wangcq01/DeLELSTM/zip/{manifest['commit']}"
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        source = temporary / "source"
        source.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            if zipped.comment.decode("ascii") != manifest["commit"]:
                raise RuntimeError("The archive does not identify the pinned upstream commit")
            for member in zipped.infolist():
                if member.is_dir():
                    continue
                relative = Path(*Path(member.filename).parts[1:])
                target = (source / relative).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise RuntimeError(f"Invalid archive path: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read(member))
        # Use LF for both patch application and source hash verification.
        for path in source.glob("*.py"):
            path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="")
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=source, check=True)
        verify(source, manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(f"Prepared and verified: {destination}")


if __name__ == "__main__":
    main()
