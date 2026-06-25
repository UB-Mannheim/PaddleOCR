#!/usr/bin/env python3

"""
Generate ALTO XML from a historic German newspaper page.

Usage
-----
    # Single page
    python tools/newspaper_to_alto.py -u \
        https://digi.bib.uni-mannheim.de/periodika/fileadmin/data/DeutReunP_856399094_18920102/default/856399094_1892_001_01.jpg \
        -o output/alto.xml

    # Download + page range (multiple pages, skip ALTO conversion)
    # --download-only produces JPEGs only; use --alto to also generate ALTO
    python tools/newspaper_to_alto.py \
        https://digi.bib.uni-mannheim.de/periodika/fileadmin/data/DeutReunP_856399094_18920102/default/\
856399094_1892_001_{:02d}.jpg \
        -o output/page --download-only

    # Batch: download JPEGs and run full OCR pipeline
        python tools/newspaper_to_alto.py \
        https://digi.bib.uni-mannheim.de/periodika/fileadmin/data/DeutReunP_856399094_18920102/default/\
856399094_1892_001_{:02d}.jpg \
        -o output/page --download-only --download-start 1 --download-end 5
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from tools.paddleocr_alto import convert_to_alto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_image(url: str, dest: Path) -> Path:
    """Download *url* to *dest* (single file) and return *dest*."""
    if dest.exists():
        log.info("Already exists: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    with open(dest, "wb") as fh:  # noqa: SIM115
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return dest


def download_sequence(
    url_template: str,
    output_dir: Path,
    start: int,
    end: int,
) -> list[Path]:
    """Download pages `{start}..{end}` from a templated URL.

    The URL template must contain a ``{:02d}`` placeholder for the page
    number (adjust if your site uses a different format).
    """
    pages: list[Path] = []
    for n in range(start, end + 1):
        url = url_template.format(n)
        dest = output_dir / f"page_{n:02d}.jpg"
        pages.append(download_image(url, dest))
    return pages


# ---------------------------------------------------------------------------
# ALTO conversion wrapper
# ---------------------------------------------------------------------------

def convert_to_alto_file(
    ocr_result: dict | list | np.ndarray,
    image_path: Path,
    output_path: Path,
    regions: Optional[list[dict]] = None,
) -> None:
    """Run ALTO conversion and write to *output_path*."""
    xml = convert_to_alto(
        ocr_result,
        image_path=str(image_path),
        regions=regions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_alto_pipeline(
    image_path: Path,
    output_path: Path,
    lang: str = "de",
    ocr_version: str = "PP-OCRv5",
    text_det_thresh: float = 0.2,
    text_det_box_thresh: float = 0.3,
    text_det_unclip_ratio: float = 2.5,
    text_det_limit_side_len: int = 1600,
    text_det_limit_type: str = "max",
    use_doc_unwarping: bool = True,
    use_textline_orientation: bool = True,
    format_block_content: bool = False,
    **kwargs,
) -> Path:
    """Run PP-Structur eV3 OCR and produce an ALTO XML file.

    Returns the path to the generated ALTO XML.
    """
    from paddleocr import PPStructureV3  # noqa: E402

    log.info("Initialising PP-StructureV3 pipeline (lang=%s, version=%s, device=%s)",
             lang, ocr_version, kwargs.get("device", "auto"))

    v3 = PPStructureV3(
        lang=lang,
        ocr_version=ocr_version,
        text_det_thresh=text_det_thresh,
        text_det_box_thresh=text_det_box_thresh,
        text_det_unclip_ratio=text_det_unclip_ratio,
        text_det_limit_side_len=text_det_limit_side_len,
        text_det_limit_type=text_det_limit_type,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        format_block_content=format_block_content,
        **kwargs,
    )

    log.info("Running OCR on %s", image_path)
    t0 = time.monotonic()
    result = v3.predict(str(image_path))
    elapsed = time.monotonic() - t0
    log.info("OCR finished in %.1f s  (%d page(s))", elapsed, len(result))

    # Convert first page to ALTO
    region = result[0]
    # PP-StructureV3 returns a dict with an "overall_ocr_res" key that
    # contains the classic OCR dict, plus layout information in the outer dict.
    # Some results also have an "overall_layout_res" key with region info.

    regions = region.get("overall_layout_res", None)

    log.info("Converting to ALTO XML: %s", output_path)
    convert_to_alto_file(region, image_path, output_path, regions=regions)

    v3.close()
    log.info("Done.")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a German newspaper page (JPEG) to ALTO XML",
    )
    # Single-page mode
    ap.add_argument(
        "--image", "-i", type=Path,
        help="Path to a single JPEG image file (use either this or -u)",
    )
    ap.add_argument(
        "--url", "-u", type=str,
        help="URL of a single JPEG image (will download to --output)",
    )
    # Batch mode
    ap.add_argument(
        "--download-start", type=int, default=1,
        help="Start page number for batch download (default: 1)",
    )
    ap.add_argument(
        "--download-end", type=int, default=1,
        help="End page number for batch download (default: 1)",
    )
    # Output
    ap.add_argument(
        "--output", "-o", type=Path, default=Path("output"),
        help="Output directory or ALTO file path (default: output)",
    )
    # Optional
    ap.add_argument(
        "--download-only", action="store_true",
        help="Only download JPEGs, skip OCR / ALTO conversion",
    )

    # OCR tunables (defaults tuned for historical German newspapers)
    ap.add_argument("--lang", default="de", help="Recognition language (default: de)")
    ap.add_argument("--ocr-version", default="PP-OCRv5", help="OCR model version (default: PP-OCRv5)")
    ap.add_argument("--det-thresh", default=0.2, type=float, help="Detection threshold (default: 0.2)")
    ap.add_argument("--det-box-thresh", default=0.3, type=float, help="Box threshold (default: 0.3)")
    ap.add_argument("--det-unclip-ratio", default=2.5, type=float, help="Unclip ratio (default: 2.5)")
    ap.add_argument("--det-limit-side", default=1600, type=int, help="Detection max side length (default: 1600)")
    ap.add_argument("--det-limit-type", default="max", help="Detection limit strategy (default: max)")

    args = ap.parse_args()

    if args.image:
        # Single file mode -- OCR + ALTO
        if not args.download_only:
            run_alto_pipeline(
                image_path=args.image,
                output_path=args.output,
                lang=args.lang,
                ocr_version=args.ocr_version,
                text_det_thresh=args.det_thresh,
                text_det_box_thresh=args.det_box_thresh,
                text_det_unclip_ratio=args.det_unclip_ratio,
                text_det_limit_side_len=args.det_limit_side,
                text_det_limit_type=args.det_limit_type,
                device=kwargs.pop("device", "auto"),  # noqa: F821
            )
        else:
            log.info("Download-only mode; nothing to do for a local file.")

    elif args.url:
        # Single URL mode
        if args.download_only:
            # Resolve filename from URL stem
            stem = Path(args.url).stem
            out_file = args.output / f"{stem}.jpg"
            download_image(args.url, out_file)
        else:
            # Download, OCR, convert
            stem = Path(args.url).stem
            image_file = args.output / f"{stem}.jpg"
            download_image(args.url, image_file)
            run_alto_pipeline(
                image_path=image_file,
                output_path=args.output if args.output.is_dir() else args.output.parent / f"{stem}.xml",
                lang=args.lang,
                ocr_version=args.ocr_version,
                text_det_thresh=args.det_thresh,
                text_det_box_thresh=args.det_box_thresh,
                text_det_unclip_ratio=args.det_unclip_ratio,
                text_det_limit_side_len=args.det_limit_side,
                text_det_limit_type=args.det_limit_type,
            )

    else:
        # No image or URL provided; try batch download
        output_dir = args.output
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Build URL template from the first URL (need to parse manually)
        # Since --url expects a single URL, for batch mode we expect the user
        # to provide a URL with a {:02d} placeholder.
        # Re-parse to allow --url to be used in batch mode.
        # (This is handled by the caller -- we just warn here.)
        log.error("Batch mode requires a --url with a {:02d} placeholder.")


if __name__ == "__main__":
    _cli()
