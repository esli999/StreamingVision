#!/usr/bin/env python3
"""
Download URLs from a plain-text list (one per line), mirroring ``wget -x -nH -P``.

Shows a single tqdm progress bar over files. Streams to disk (suitable for large npz).
Uses concurrent HTTP by default (see ``-j``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

USER_AGENT = "GenMatter-download_url_list/1.0"

# tqdm label when -i matches scripts/*_urls.txt in this repo
_DATASET_DESC = {
    "gestalt_stimuli_urls": "Gestalt stimuli",
    "tapvid_davis_urls": "DAVIS TAP-Vid",
    "rdk_urls": "RDK psychophysics",
}


def _progress_desc(input_path: Path, override: str | None) -> str:
    if override is not None:
        return override
    stem = input_path.stem  # e.g. gestalt_stimuli_urls
    if stem in _DATASET_DESC:
        return _DATASET_DESC[stem]
    return stem.replace("_", " ").title()


def _rel_path(url: str) -> str | None:
    parts = urlparse(url)
    rel = parts.path.lstrip("/")
    if not rel or rel.endswith("/"):
        return None
    return rel


def _download_one(url: str, dest: Path) -> str | None:
    """Return error message string on failure, else None."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1024 * 1024)
                tmp.replace(dest)
            except Exception:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                raise
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return str(e)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True, help="URL list (one URL per line)")
    p.add_argument("-P", "--prefix", type=Path, required=True, help="Destination root (like wget -P)")
    p.add_argument(
        "-c",
        dest="skip_existing",
        action="store_true",
        help="Skip files that already exist (resume / avoid re-downloading)",
    )
    p.add_argument(
        "--desc",
        default=None,
        metavar="LABEL",
        help="Progress bar title (default: inferred from URL list filename)",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=16,
        metavar="N",
        help="Concurrent downloads (default: 16). Set to 1 for sequential. Lower if S3 throttles.",
    )
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be at least 1")

    text = args.input.read_text(encoding="utf-8", errors="replace")
    urls = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    prefix: Path = args.prefix
    failed: list[tuple[str, str]] = []
    bar_desc = _progress_desc(args.input, args.desc)

    def worker(url: str) -> tuple[str, str | None]:
        rel = _rel_path(url)
        if rel is None:
            return url, "empty or directory URL"
        out = prefix / rel
        if args.skip_existing and out.is_file():
            return url, None
        err = _download_one(url, out)
        return url, err

    if args.jobs == 1:
        for url in tqdm(urls, desc=bar_desc, unit="file", file=sys.stderr):
            u, err = worker(url)
            if err is not None:
                failed.append((u, err))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            future_to_url = {ex.submit(worker, url): url for url in urls}
            for fut in tqdm(
                as_completed(future_to_url),
                total=len(urls),
                desc=bar_desc,
                unit="file",
                file=sys.stderr,
            ):
                url = future_to_url[fut]
                try:
                    u, err = fut.result()
                except Exception as e:
                    failed.append((url, str(e)))
                else:
                    if err is not None:
                        failed.append((u, err))

    if failed:
        print(f"\n{len(failed)} download(s) failed:", file=sys.stderr)
        for url, err in failed[:30]:
            print(f"  {url}\n    {err}", file=sys.stderr)
        if len(failed) > 30:
            print(f"  ... and {len(failed) - 30} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
