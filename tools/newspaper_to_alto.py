#!/usr/bin/env python3
"""
Generate ALTO XML from a single newspaper page (or batch of pages).

Uses PP-StructureV3 which provides layout detection, multi-column
reading-order recovery, table recognition and text line extraction.

Usage
-----
    # Single image from URL
    python tools/newspaper_to_alto.py https://example.com/page.jpg

    # Single image (local file), custom output
    python tools/newspaper_to_alto.py page.jpg -o page.alto.xml

    # Multiple images
    python tools/newspaper_to_alto.py *.jpg -o output/

    # Directory of images
    python tools/newspaper_to_alto.py /path/to/scans/ -o output/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# PaddlePaddle disables oneDNN via this env var, which must be set before
# any Paddlepaddle module is imported.  PaddleOCR / PaddleX import paddlepaddle
# internally, so we must do this before any other import that might trigger it.
if "PADDLE_USE_DNNL" not in os.environ:
    os.environ["PADDLE_USE_DNNL"] = "0"

# Ensure the tool directory and the project root are in sys.path so that
# sibling imports (e.g. ``from paddleocr_alto import ...``) work regardless
# of the working directory.
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)                          # for paddleocr_alto.py
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, os.pardir)))  # for tools/paddleocr_alto

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> str:
    """Download ``url`` to ``dest``. Returns the destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(8192):
            fh.write(chunk)
    return str(dest)


def _collect_images(path: Path) -> list[Path]:
    """Return image files under ``path`` (file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        imgs = [p for e in IMAGE_EXTS for p in path.rglob(f"*{e}")]
        return sorted(imgs)
    return []


# ---------------------------------------------------------------------------
# Pipeline: download ->  convert
# ---------------------------------------------------------------------------

def _run_ocr_alto(image_path: str, output_path: str, **opts) -> None:
    """Run PP-StructureV3 (layout-aware OCR) then convert result to ALTO XML."""
    from paddleocr import PPStructureV3
    from paddleocr_alto import convert_to_alto

    print(f"[OCR]    processing  {image_path}", flush=True)

    # PP-StructureV3 provides layout detection + text line recognition,
    # which is essential for multi-column newspaper pages.
    v3 = PPStructureV3(**opts)

    t0 = time.time()
    result = v3.predict(image_path)
    elapsed = time.time() - t0

    # ``result`` is a list of pages; take the first one.
    page = result[0] if isinstance(result, list) else result

    print(f"[OCR]    done      {image_path:>40s}  {elapsed:6.1f}s", flush=True)

    # Extract region info for layout-aware ALTO generation.
    regions = page.get("overall_layout_res", None)

    xml = convert_to_alto(page, image_path=image_path, regions=regions)

    # Determine output file path
    out = Path(output_path)
    if out.is_dir():
        out = out / f"{Path(image_path).stem}.alto.xml"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")

    print(f"[ALTO]  written   {out}", flush=True)

    v3.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Newspaper images --> ALTO XML via PaddleOCR",
    )

    # --- positional: one or more image paths / directories / URLs ---
    ap.add_argument(
        "images", nargs="+",
        help="Image file(s), directory, or URL(s)",
    )

    # --- output ---
    ap.add_argument(
        "-o", "--output", type=Path, default=Path("output"),
        help="Output directory or ALTO file path  (default: output/)",
    )

    # --- detection tunables (defaults tuned for historical German newspapers) ---
    ap.add_argument("--det-thresh", type=float, default=0.2,
                    help="Detection threshold (lower = more boxes)  (default: 0.2)")
    ap.add_argument("--det-box-thresh", type=float, default=0.3,
                    help="Box confidence threshold (lower = more boxes)  (default: 0.3)")
    ap.add_argument("--det-unclip-ratio", type=float, default=2.5,
                    help="Unclip ratio for box expansion  (default: 2.5)")
    ap.add_argument("--det-limit-side", type=int, default=1600,
                    help="Max image side length for detection  (default: 1600)")
    ap.add_argument("--det-limit-type", default="max",
                    help="Limit strategy: min, max, etc.  (default: max)")

    # --- language / model ---
    ap.add_argument("--lang", default="de", help="Recognition language  (default: de)")
    ap.add_argument("--version", default="PP-OCRv5",
                    help="PaddleOCR model version  (default: PP-OCRv5)")

    args = ap.parse_args()

    # ---------- collect targets ----------
    targets: list[str] = []

    for arg in args.images:
        p = Path(arg)
        if p.exists():
            targets.extend(str(img) for img in _collect_images(p))
        elif arg.startswith("http://") or arg.startswith("https://"):
            # Download to output directory first
            stem = Path(arg).stem
            dest = args.output / f"{stem}.jpg"
            _download(arg, dest)
            targets.append(str(dest))
        else:
            print(f"[WARN] skipping  {arg}  (not found or not a URL)", file=sys.stderr)

    if not targets:
        ap.error("no images found")

    # ---------- build PP-StructureV3 options ----------
    # These are the actual __init__ arguments accepted by PPStructureV3.
    opts = {
        "lang": args.lang,
        "ocr_version": args.version,
        "text_det_thresh": args.det_thresh,
        "text_det_box_thresh": args.det_box_thresh,
        "text_det_unclip_ratio": args.det_unclip_ratio,
        "text_det_limit_side_len": args.det_limit_side,
        "text_det_limit_type": args.det_limit_type,
    }

    # ---------- process ----------
    for img in targets:
        out = args.output / f"{Path(img).stem}.alto.xml"
        _run_ocr_alto(img, str(out), **opts)

    print("[INFO] all done.", flush=True)


if __name__ == "__main__":
    main()
