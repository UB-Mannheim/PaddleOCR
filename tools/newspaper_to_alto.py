#!/usr/bin/env python3
"""
Generate ALTO XML from a single newspaper page.

Usage
-----
    # Single image (local file or URL as positional arg)
    python tools/newspaper_to_alto.py page_01.jpg
    python tools/newspaper_to_alto.py https://example.com/page.jpg
    python tools/newspaper_to_alto.py page.jpg -o output/alto.xml

    # Batch: all supported images in a directory
    python tools/newspaper_to_alto.py /path/to/pages/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow running as `python tools/newspaper_to_alto.py` (from repo root)
# and as `python tools/newspaper_to_alto.py` (from anywhere).
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


def _collect_images(path: Path) -> list[Path]:
    """Return list of image file paths under *path* (file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        imgs = sorted(
            p for e in _IMAGE_EXTS
            for p in path.rglob(f"*{e}")
        )
        if not imgs:
            log.warning("No images found in %s", path)
        return list(imgs)
    return []


def _download_image(url: str, dest: Path) -> Path:
    """Download *url* to *dest*."""
    import requests
    if dest.exists():
        log.info("Already exists: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return dest


def _download_one(url: str, output_dir: Path) -> Path:
    """Download a single image, returning the destination path."""
    stem = Path(url).stem
    dest = output_dir / f"{stem}.jpg"
    return _download_image(url, dest)


def _run_alto(image_path: Path, output_path: Path, **kwargs) -> Path:
    """Run PP-StructureV3 OCR and convert to ALTO XML."""
    from paddleocr import PPStructureV3

    log.info("Initialising PP-StructureV3 (device=%s, lang=%s)",
             kwargs.get("device", "auto"), kwargs.get("lang", "de"))

    v3 = PPStructureV3(**kwargs)

    log.info("Running OCR on %s", image_path)
    t0 = time.monotonic()
    result = v3.predict(str(image_path))
    elapsed = time.monotonic() - t0
    log.info("OCR finished in %.1f s  (%d page(s))", elapsed, len(result))

    from paddleocr_alto import convert_to_alto

    region = result[0]
    regions = region.get("overall_layout_res", None)

    xml = convert_to_alto(region, image_path=str(image_path), regions=regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")

    v3.close()
    log.info("Output: %s", output_path)
    return output_path


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Convert newspaper pages (JPEG/PNG) to ALTO XML",
    )

    # --- positional: one or more image paths / dirs / URLs ---
    ap.add_argument(
        "images", nargs="+",
        help="Image file(s), directory, or URL(s)",
    )

    # --- options ---
    ap.add_argument(
        "--output", "-o", type=Path, default=Path("output"),
        help="Output directory or file path (default: output)",
    )
    ap.add_argument(
        "--download-only", action="store_true",
        help="Only download images, skip OCR",
    )

    # --- OCR tunables (defaults tuned for historical German newspapers) ---
    ap.add_argument("--lang", default="de")
    ap.add_argument("--ocr-version", default="PP-OCRv5")
    ap.add_argument("--det-thresh", type=float, default=0.2)
    ap.add_argument("--det-box-thresh", type=float, default=0.3)
    ap.add_argument("--det-unclip-ratio", type=float, default=2.5)
    ap.add_argument("--det-limit-side", type=int, default=1600)
    ap.add_argument("--det-limit-type", default="max")
    ap.add_argument("--device", default="auto", help="cpu, gpu, or auto")

    args = ap.parse_args()

    # --- build OCR kwargs ---
    ocr_kwargs = {
        "lang": args.lang,
        "ocr_version": args.ocr_version,
        "text_det_thresh": args.det_thresh,
        "text_det_box_thresh": args.det_box_thresh,
        "text_det_unclip_ratio": args.det_unclip_ratio,
        "text_det_limit_side_len": args.det_limit_side,
        "text_det_limit_type": args.det_limit_type,
        "device": args.device,
    }

    output_dir = args.output

    # --- collect targets ---
    targets: list[tuple[Path, bool]] = []  # (image_path, is_url)

    for arg in args.images:
        p = Path(arg)
        if arg.startswith("http://") or arg.startswith("https://"):
            # URL: download to output_dir
            targets.append((_download_one(arg, output_dir), False))
        elif p.exists():
            targets.extend((img, False) for img in _collect_images(p))

    # --- download first, then process ---
    if not args.download_only:
        for img_path, _is_url in targets:
            _run_alto(img_path, output_dir, **ocr_kwargs)
    else:
        log.info("Download-only mode; images already downloaded.")


if __name__ == "__main__":
    _cli()
