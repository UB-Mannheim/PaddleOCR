#!/usr/bin/env python3

"""
PAGE XML converter for PaddleOCR output.

Converts PaddleOCR pipeline/PP-StructureV3 OCR results into PAGE XML
(Page Access and Geometry) format used by libraries, the PagesFormat
(omni:us) and PRImA ecosystem.

Supported PaddleOCR output schemas
==================================
1. Classic pipeline (``ocr.predict()``): dict with ``dt_polys``,
   ``rec_boxes``, ``rec_texts``, ``rec_scores``.
2. PP-StructureV3 / doc-parsing pipeline (``pp_structure.predict()``):
   list of layout regions with ``type``, ``bbox``, ``res`` fields.
3. Dict with ``dt_boxes`` (``N,4,2`` polygons) + ``rec_boxes``
   (``N,4`` rects) -- legacy format.

Usage
=====
Python
------
    from paddleocr import PaddleOCR
    from tools.paddleocr_page import convert_to_page

    ocr = PaddleOCR()
    result = ocr.predict("page.png")
    page = result[0]
    xml = convert_to_page(page, image_path="page.png")
    with open("result.xml", "w") as f:
        f.write(xml)

Command line
------------
    python tools/paddleocr_page.py input.json --image page.png --output result.xml
"""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# PRImA PAGE XML namespace
NS = "http://schema.primaresearch.org/PAGE_GTS/pagecontent/2017_07_15/"

LABEL_TO_REGION: Dict[str, str] = {
    "title": "Heading",
    "text": "normal",
    "header": "Header",
    "footer": "Footer",
    "figure": "normal",
    "table": "normal",
    "equation": "normal",
    "formula": "normal",
    "chart": "normal",
    "graph": "normal",
    "image": "normal",
}

_TEXT_LABELS = {"title", "text", "header", "footer", "figure"}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _uniq(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _polypoints(bbox: Tuple[float, ...]) -> str:
    """Convert a 4-tuple bbox (left, top, right, bottom) to a PAGE ``points`` string.

    Points are ordered clockwise starting from top-left.
    """
    left, top, right, bottom = bbox
    return (
        f"{round(left)},{round(top)} "
        f"{round(right)},{round(top)} "
        f"{round(right)},{round(bottom)} "
        f"{round(left)},{round(bottom)}"
    )


# ---------------------------------------------------------------------------
#  Input normalisation
# ---------------------------------------------------------------------------

def _normalise_page(
    result_page: Any,
) -> Tuple[List[Tuple[Tuple[float, ...], str, float]], Dict[str, Any]]:
    """Accept several PaddleOCR output shapes and return:

    (text_lines, meta)

    text_lines  -- list of ((left,top,right,bottom), text, confidence)
    meta        -- {"width": W, "height": H, "image_path": str, ...}
    """

    if isinstance(result_page, dict) and ("rec_texts" in result_page):
        texts = result_page["rec_texts"]
        boxes = result_page.get("rec_boxes", [])
        scores = result_page.get("rec_scores", [])
        polys = result_page.get("dt_polys") or result_page.get("dt_boxes")
        width = result_page.get("img_width", 0)
        height = result_page.get("img_height", 0)
        image_path = result_page.get("file", result_page.get("image_path", ""))

        meta: Dict[str, Any] = {}
        if width and height:
            meta["width"] = int(width)
            meta["height"] = int(height)
        if image_path:
            meta["image_path"] = str(image_path)

        lines: List[Tuple[Tuple[float, ...], str, float]] = []
        n = len(texts)
        for i in range(n):
            txt = texts[i]
            if txt.strip() in ("", "\n"):
                continue
            conf = float(scores[i]) if i < len(scores) else 1.0
            if box := boxes[i] if i < len(boxes) else None:
                lines.append((tuple(box), txt, conf))
            elif polys is not None and i < len(polys):
                poly = polys[i]
                arr = poly.tolist() if hasattr(poly, "tolist") else poly
                xs = [p[0] for p in arr]
                ys = [p[1] for p in arr]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                lines.append((bbox, txt, conf))
        return lines, meta

    if isinstance(result_page, (list, tuple)):
        return _normalise_structure(result_page)

    raise ValueError(
        f"Unrecognised PaddleOCR result type: {type(result_page).__name__}. "
        "Expected a dict with 'rec_texts' or a list of layout regions."
    )


def _normalise_structure(
    regions: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[Tuple[float, ...], str, float]], Dict[str, Any]]:
    """Handle the StructureSystem / PP-StructureV3 return value."""
    lines: List[Tuple[Tuple[float, ...], str, float]] = []
    meta: Dict[str, Any] = {}

    for region in regions:
        rtype = region.get("type", region.get("label", "text"))
        res = region.get("res")
        if res is None:
            continue

        if rtype in _TEXT_LABELS:
            if isinstance(res, list):
                for entry in res:
                    txt = entry.get("text", "")
                    conf = float(entry.get("confidence", 1.0))
                    if not txt.strip():
                        continue
                    entry_region = entry.get("text_region")
                    if entry_region:
                        region_list = (
                            entry_region.tolist()
                            if hasattr(entry_region, "tolist")
                            else entry_region
                        )
                        xs = [float(p[0]) for p in region_list]
                        ys = [float(p[1]) for p in region_list]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                    else:
                        bbox = tuple(region.get("bbox", [0, 0, 0, 0]))
                    lines.append((bbox, txt, conf))
            elif isinstance(res, dict):
                inner = res.get("overall_ocr_res", res)
                if isinstance(inner, dict) and "rec_texts" in inner:
                    texts = inner.get("rec_texts", [])
                    boxes = inner.get("rec_boxes", [])
                    scores = inner.get("rec_scores", [])
                    for i, txt in enumerate(texts):
                        if not txt.strip():
                            continue
                        conf = float(scores[i]) if i < len(scores) else 1.0
                        box = boxes[i] if i < len(boxes) else None
                        bbox = tuple(box) if box else tuple(
                            region.get("bbox", [0, 0, 0, 0])
                        )
                        lines.append((bbox, txt, conf))

        if not meta.get("_has_dims"):
            meta["_has_dims"] = True
            if region.get("img"):
                h, w = region["img"].shape[:2]
                meta["height"] = int(h)
                meta["width"] = int(w)

    return lines, meta


# ---------------------------------------------------------------------------
#  Group text lines into PAGE TextRegions
# ---------------------------------------------------------------------------

def _iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    b1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    b2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = b1 + b2 - inter
    return inter / union if union > 0 else 0.0


def _group_into_regions(
    lines: List[Tuple[Tuple[float, ...], str, float]],
    regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Tuple[str, List[Tuple[Tuple[float, ...], str, float]]]]:
    """Cluster text lines into groups that belong to the same logical region.

    Returns list of (region_role, ordered_lines).
    """
    if not lines:
        return []

    block_assign: Dict[int, int] = {}
    block_order: List[int] = []
    block_labels: Dict[int, str] = {}

    if regions is not None and isinstance(regions, (list, tuple)):
        regions_text = [r for r in regions if r.get("type") in _TEXT_LABELS]
        line_idx = 0
        for ri, region in enumerate(regions_text):
            rbbox = region.get("bbox", [0, 0, 0, 0])
            for _ in range(len(region.get("res", []))):
                if line_idx >= len(lines):
                    break
                lbbox = lines[line_idx][0]
                if _iou(lbbox, rbbox) > 0.1 or line_idx == 0:
                    bid = ri
                    block_assign[line_idx] = bid
                    if bid not in block_order:
                        block_order.append(bid)
                    block_labels[bid] = region.get(
                        "type", region.get("label", "text")
                    )
                    line_idx += 1
                else:
                    if not block_assign:
                        block_assign[line_idx] = block_order[0] if block_order else 0
                        if 0 not in block_order:
                            block_order.append(0)
                        block_labels[0] = "text"
                    line_idx += 1

    if not block_assign and lines:
        block_assign = {i: 0 for i in range(len(lines))}
        block_order = [0]
        block_labels = {0: "text"}

    blocks: Dict[int, List[Tuple[Tuple[float, ...], str, float]]] = {
        bid: [] for bid in block_order
    }
    for idx, line in enumerate(lines):
        bid = block_assign.get(idx, 0)
        blocks[bid].append(line)
    for bid in blocks:
        blocks[bid].sort(key=lambda t: t[0][1])

    result: List[Tuple[str, List[Tuple[Tuple[float, ...], str, float]]]] = []
    for bid in block_order:
        bl = block_labels.get(bid, "text")
        result.append((bl, blocks[bid]))
    return result


# ---------------------------------------------------------------------------
#  PAGE XML document builder
# ---------------------------------------------------------------------------

def build_page_xml(
    lines: Sequence[Tuple[Tuple[float, ...], str, float]],
    image_path: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    regions: Optional[Sequence[Dict[str, Any]]] = None,
    page_id: str = "page_1",
    creator: str = "PaddleOCR",
) -> str:
    """Build a PAGE XML document and return it as a pretty-printed string.

    Parameters
    ----------
    lines : sequence
        Each item: ``(bbox, text, confidence)`` where bbox is
        ``(left, top, right, bottom)`` in pixels.
    image_path : str
        Source image filename.
    image_width, image_height : int, optional
        Page dimensions; inferred from ``lines`` if not provided.
    regions : list of region dicts, optional
        Layout region list for PageTextRegion labelling.
    page_id : str
        ID for the ``<Page>`` element.
    creator : str
        Value for ``<Metadata>/<Creator>``.
    """

    if lines:
        all_x = [b[0] for b, _, _ in lines] + [b[2] for b, _, _ in lines]
        all_y = [b[1] for b, _, _ in lines] + [b[3] for b, _, _ in lines]
        if not image_width:
            image_width = max(int(max(all_x)), 1024)
        if not image_height:
            image_height = max(int(max(all_y)), 1024)

    ns = NS  # namespace URI

    root = ET.Element(
        f"{{{ns}}}PcGts",
        id=f"pcgts_{uuid.uuid4().hex[:8]}",
        **{
            "xmlns": ns,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{ns} https://raw.githubusercontent.com/PRImA-Research-Lab/PAGE-XML/master/schema/pagecontent.xsd",
        },
    )

    # Metadata
    meta_el = ET.SubElement(root, f"{{{ns}}}Metadata")
    ET.SubElement(meta_el, f"{{{ns}}}Creator").text = creator
    ET.SubElement(meta_el, f"{{{ns}}}Created").text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_change = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ET.SubElement(meta_el, f"{{{ns}}}LastChange").text = last_change

    # Page
    page = ET.SubElement(
        root,
        f"{{{ns}}}Page",
        id=page_id,
        imageFilename=image_path,
        imageWidth=str(image_width),
        imageHeight=str(image_height),
    )

    groupings = _group_into_regions(list(lines), regions if regions else None)
    if not groupings:
        return _pretty_print_xml(root)

    idx_counter = 0
    for region_label, region_lines in groupings:
        role = LABEL_TO_REGION.get(region_label, "normal")

        # Compute bounding polygon for the whole region (all lines in it)
        overall_left = int(round(min(b[0] for b, _, _ in region_lines)))
        overall_top = int(round(min(b[1] for b, _, _ in region_lines)))
        overall_right = int(round(max(b[2] for b, _, _ in region_lines)))
        overall_bottom = int(round(max(b[3] for b, _, _ in region_lines)))

        idx_counter += 1
        tr_id = f"region_{idx_counter}"
        tr = ET.SubElement(
            page,
            f"{{{ns}}}TextRegion",
            id=tr_id,
            custom=f"role={role}",
        )
        ET.SubElement(tr, f"{{{ns}}}Coords", points=_polypoints((
            overall_left, overall_top, overall_right, overall_bottom
        )))

        for line_idx, (bbox, txt, conf) in enumerate(region_lines):
            left, top, right, bottom = bbox

            tl_id = f"line_{idx_counter}_{line_idx}"
            tl = ET.SubElement(
                tr,
                f"{{{ns}}}TextLine",
                id=tl_id,
            )

            # Coords for the text line
            ET.SubElement(tl, f"{{{ns}}}Coords", points=_polypoints(bbox))

            # Baseline (approximate -- we don't have line orientation from PaddleOCR)
            baseline_y = int(round(bottom))
            baseline_pts = f"{round(left)},{baseline_y} {round(right)},{baseline_y}"
            ET.SubElement(tl, f"{{{ns}}}Baseline", points=baseline_pts)

            # TextEquiv for the full line
            te = ET.SubElement(tl, f"{{{ns}}}TextEquiv")
            ET.SubElement(te, f"{{{ns}}}Unicode").text = txt
            # Add confidence as attribute on TextEquiv
            te.set("conf", str(round(float(conf), 4)))

            # Word-level decomposition (optional -- split on whitespace)
            words = txt.split() if txt else []
            if words:
                line_w = right - left
                per_word = line_w / len(words)

                for wi, word in enumerate(words):
                    word_id = f"word_{idx_counter}_{line_idx}_{wi}"
                    word_x = int(round(left + wi * per_word))
                    word_w = max(int(round(per_word)), 1)

                    w_el = ET.SubElement(
                        tl,
                        f"{{{ns}}}Word",
                        id=word_id,
                    )
                    ET.SubElement(
                        w_el,
                        f"{{{ns}}}Coords",
                        points=_polypoints((word_x, int(round(top)), word_x + word_w, int(round(bottom)))),
                    )
                    wte = ET.SubElement(w_el, f"{{{ns}}}TextEquiv")
                    ET.SubElement(wte, f"{{{ns}}}Unicode").text = word
                    wte.set("conf", str(round(float(conf), 4)))

    return _pretty_print_xml(root)


def _pretty_print_xml(root: ET.Element) -> str:
    """Pretty-print an ElementTree root to a string."""
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="utf-8"?>\n{xml_bytes}'


# ---------------------------------------------------------------------------
#  High-level API
# ---------------------------------------------------------------------------

def convert_to_page(
    result_page: Any,
    image_path: str = "image.png",
    regions: Optional[Sequence[Dict[str, Any]]] = None,
    page_id: str = "page_1",
    creator: str = "PaddleOCR",
) -> str:
    """Convert a PaddleOCR result dict to a PAGE XML string.

    Parameters
    ----------
    result_page : dict or list
        A single-page result from ``ocr.predict()`` or
        ``pp_structure.predict()``.
    image_path : str
        Source image filename.
    regions : list, optional
        Layout region list for PageTextRegion labelling.
    page_id : str
        ID for the ``<Page>`` element.
    creator : str
        Value for ``<Metadata>/<Creator>``.

    Returns
    -------
    str
        Pretty-printed PAGE XML as a string.
    """
    lines, meta = _normalise_page(result_page)
    return build_page_xml(
        lines,
        image_path=image_path,
        image_width=meta.get("width"),
        image_height=meta.get("height"),
        regions=regions,
        page_id=page_id,
        creator=creator,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR output to PAGE XML"
    )
    parser.add_argument(
        "input",
        help="Path to JSON file with PaddleOCR result, or raw JSON string",
    )
    parser.add_argument(
        "--image", "-i", default="image.png",
        help="Source image path/name (default: image.png)",
    )
    parser.add_argument(
        "--output", "-o", help="Output PAGE XML file (default: stdout)",
    )
    parser.add_argument("--page-id", default="page_1", help="Page ID")
    parser.add_argument("--creator", default="PaddleOCR", help="Creator name")
    parser.add_argument(
        "--regions", help="Path to JSON with region list (optional)",
    )
    args = parser.parse_args()

    inp_path = Path(args.input)
    if inp_path.exists():
        data = json.loads(inp_path.read_text())
    else:
        data = json.loads(args.input)

    regions = None
    if args.regions:
        regions = json.loads(Path(args.regions).read_text())

    xml = convert_to_page(
        data,
        image_path=args.image,
        regions=regions,
        page_id=args.page_id,
        creator=args.creator,
    )

    if args.output:
        Path(args.output).write_text(xml, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(xml)


if __name__ == "__main__":
    _cli()
