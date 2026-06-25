#!/usr/bin/env python3

"""
ALTO XML converter for PaddleOCR output.

Converts PaddleOCR pipeline/PP-StructureV3 OCR results into ALTO (Archive &
Library Text on the Internet) XML format -- LC-standard structured page
description used by libraries and archives.

Supported PaddleOCR output schemas
==================================
1. Classic pipeline (``ocr.predict()``): dict with ``dt_polys``,
   ``rec_boxes``, ``rec_texts``, ``rec_scores``.
2. PP-StructureV3 / doc-parsing pipeline (``pp_structure.predict()``):
   list of layout regions, each containing text regions via
   ``overall_ocr_res`` (classic dict) plus layout metadata.
3. Dict with ``dt_boxes`` (``N,4,2`` polygons) + ``rec_boxes``
   (``N,4`` rects) -- legacy format.

Usage
=====
Python
------
    from paddleocr import PaddleOCR
    from tools.paddleocr_alto import convert_to_alto

    ocr = PaddleOCR()
    result = ocr.predict("page.png")
    page = result[0]
    xml = convert_to_alto(page, image_path="page.png")

Command line
------------
    python tools/paddleocr_alto.py input.json --image page.png --output result.xml
"""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

NS = "http://www.loc.gov/standards/alto/"

LABEL_TO_CLASS: Dict[str, str] = {
    "title": "Heading",
    "text": "Paragraph",
    "header": "Header",
    "footer": "Footer",
    "figure": "Picture",
    "table": "Table",
    "equation": "Formula",
    "formula": "Formula",
    "chart": "Chart",
    "graph": "Chart",
}

_TEXT_LABELS = {"title", "text", "header", "footer"}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _uniq(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _element(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, f"{{{NS}}}{tag}", attrs)


# ---------------------------------------------------------------------------
#  Input normalisation  (bridge multiple PaddleOCR output schemas)
# ---------------------------------------------------------------------------

def _normalise_page(
    result_page: Any,
) -> Tuple[List[Tuple[Tuple[float, ...], str, float]], Dict[str, Any]]:
    """Accept several PaddleOCR output shapes and return:

    (text_lines, meta)

    text_lines  -- list of ((left,top,right,bottom), text, confidence)
    meta        -- {"width": W, "height": H, "image_path": str, ...}

    Supported shapes:
    -----------------
    A. Classic dict (ocr.predict): dt_polys / dt_boxes, rec_boxes,
       rec_texts, rec_scores.
    B. PP-StructureV3 / StructureSystem output (list of regions).
       Each region has ``type``, ``bbox``, ``res`` (list of lines with
       ``text``, ``confidence``, ``text_region``) and optional ``label``.
    C. Dict with ``dt_boxes`` (legacy).
    """

    # --- Shape A / C: classic single-page dict --------------------------
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

    # --- Shape B: PP-Structure / StructureSystem list of regions --------
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
                    region_region = entry.get("text_region")
                    if region_region:
                        region_list = (
                            region_region.tolist()
                            if hasattr(region_region, "tolist")
                            else region_region
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
#  Group text lines into ALTO TextBlocks
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


def _group_into_blocks(
    lines: List[Tuple[Tuple[float, ...], str, float]],
    regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Tuple[str, List[Tuple[Tuple[float, ...], str, float]]]]:
    """Cluster text lines into groups that share a logical block.

    Returns list of (block_class, ordered_lines).
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
        cls = LABEL_TO_CLASS.get(bl, "Paragraph")
        result.append((cls, blocks[bid]))
    return result


# ---------------------------------------------------------------------------
#  ALTO document builder
# ---------------------------------------------------------------------------

def build_alto(
    lines: Sequence[Tuple[Tuple[float, ...], str, float]],
    image_path: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    regions: Optional[Sequence[Dict[str, Any]]] = None,
    measurement_unit: str = "pixel",
    page_id: str = "PAGE_1",
) -> str:
    """Build an ALTO XML document and return it as a pretty-printed string.

    Parameters
    ----------
    lines : sequence
        Each item: ``(bbox, text, confidence)`` where bbox is
        ``(left, top, right, bottom)`` in pixels.
    image_path : str
        Path or filename of the source image.
    image_width, image_height : int, optional
        Page dimensions.  If ``None`` they are inferred from ``lines``.
    regions : list of region dicts, optional
        PP-StructureV3 / StructureSystem region list for block labelling.
    measurement_unit : str
        ALTO ``<MeasurementUnit>`` value, default ``"pixel"``.
    page_id : str
        ID string for the ``<Page>`` element.
    """

    if lines:
        all_x = [b[0] for b, _, _ in lines] + [b[2] for b, _, _ in lines]
        all_y = [b[1] for b, _, _ in lines] + [b[3] for b, _, _ in lines]
        if not image_width:
            image_width = max(int(max(all_x)), 1024)
        if not image_height:
            image_height = max(int(max(all_y)), 1024)

    root = ET.Element(
        f"{{{NS}}}alto",
        **{
            "xmlns": NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{NS} https://www.loc.gov/standards/alto/v4/alto-4-4.xsd",
            "SCHEMAVERSION": "4.4",
        },
    )

    # Description
    desc = ET.SubElement(root, f"{{{NS}}}Description")
    ET.SubElement(desc, f"{{{NS}}}MeasurementUnit").text = measurement_unit
    src = ET.SubElement(desc, f"{{{NS}}}sourceImageInformation")
    ET.SubElement(src, f"{{{NS}}}fileName").text = image_path
    ET.SubElement(
        desc, f"{{{NS}}}Processing",
        PROCESSINGSTEP="OCR PaddleOCR",
        STEPNUMBER="1",
        DATEANDTIME="",
    )

    # Styles
    styles_el = ET.SubElement(root, f"{{{NS}}}Styles")
    ET.SubElement(styles_el, f"{{{NS}}}TextStyle", ID="tsDefault")

    # Layout + Page
    layout_el = ET.SubElement(root, f"{{{NS}}}Layout")
    page = ET.SubElement(
        layout_el, f"{{{NS}}}Page",
        ID=page_id,
        WIDTH=str(image_width),
        HEIGHT=str(image_height),
        PHYSICAL_IMG_TYPE="Main",
    )

    grouped = _group_into_blocks(list(lines), regions if regions else None)
    if not grouped:
        pass
    else:
        cls_counter = 0
        for block_cls, block_lines in grouped:
            cls_counter += 1
            cid = f"ts{cls_counter}"
            ET.SubElement(styles_el, f"{{{NS}}}TextStyle", ID=cid)

            bid = _uniq("TextBlock")
            lb = LABEL_TO_CLASS.get(str(block_cls), "Paragraph")
            first_bbox = block_lines[0][0]
            last_bbox = block_lines[-1][0]
            tb = ET.SubElement(
                page, f"{{{NS}}}TextBlock",
                ID=bid, CLASS=lb, STYLEID=cid,
                HPOS=str(int(round(first_bbox[0]))),
                VPOS=str(int(round(first_bbox[1]))),
                WIDTH=str(int(round(last_bbox[2]) - round(first_bbox[0]))),
                HEIGHT=str(int(round(last_bbox[3]) - round(first_bbox[1]))),
            )
            for bbox, txt, conf in block_lines:
                left, top, right, bottom = bbox
                w = max(int(round(right)) - int(round(left)), 1)
                h = max(int(round(bottom)) - int(round(top)), 1)
                lid = _uniq("TLine")
                tl = ET.SubElement(
                    tb, f"{{{NS}}}TextLine",
                    ID=lid,
                    HPOS=str(int(round(left))),
                    VPOS=str(int(round(top))),
                    WIDTH=str(w),
                    HEIGHT=str(h),
                )
                words = txt.split() if txt else []
                if words and len(words) > 1:
                    line_w = right - left
                    pw = line_w / len(words)
                    cx = left
                    for wi, word in enumerate(words):
                        sid = _uniq("String")
                        sw = max(int(round(pw)), 1)
                        ET.SubElement(
                            tl, f"{{{NS}}}String",
                            ID=sid,
                            HPOS=str(int(round(cx))),
                            VPOS=str(int(round(top))),
                            WIDTH=str(sw),
                            HEIGHT=str(h),
                            CONTENT=word,
                            WC=str(round(float(conf), 4)),
                        )
                        cx += sw + 2
                        if wi < len(words) - 1:
                            ET.SubElement(tl, f"{{{NS}}}SP", WIDTH="2")
                else:
                    content = txt if words else ""
                    sid = _uniq("String")
                    ET.SubElement(
                        tl, f"{{{NS}}}String",
                        ID=sid,
                        HPOS=str(int(round(left))),
                        VPOS=str(int(round(top))),
                        WIDTH=str(w),
                        HEIGHT=str(h),
                        CONTENT=content,
                        WC=str(round(float(conf), 4)),
                    )

    # Pretty-print and return
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="utf-8"?>\n{xml_bytes}'


# ---------------------------------------------------------------------------
#  High-level API
# ---------------------------------------------------------------------------

def convert_to_alto(
    result_page: Any,
    image_path: str = "image.png",
    regions: Optional[Sequence[Dict[str, Any]]] = None,
    measurement_unit: str = "pixel",
    page_id: str = "PAGE_1",
) -> str:
    """Convert a PaddleOCR result dict to an ALTO XML string.

    Parameters
    ----------
    result_page : dict or list
        A single-page result from ``ocr.predict()`` (dict with
        ``rec_texts``) or from ``pp_structure.predict()`` (list of
        layout regions).
    image_path : str
        Source image filename / path (used in ALTO metadata).
    regions : list, optional
        PP-StructureV3 region list used for block labelling.
    measurement_unit : str
        ALTO measurement unit, either ``"pixel"`` or ``"mm10"``.
    page_id : str
        ID for the ``<Page>`` element.

    Returns
    -------
    str
        Pretty-printed ALTO XML as a string.
    """
    lines, meta = _normalise_page(result_page)
    return build_alto(
        lines,
        image_path=image_path,
        image_width=meta.get("width"),
        image_height=meta.get("height"),
        regions=regions,
        measurement_unit=measurement_unit,
        page_id=page_id,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR output to ALTO XML"
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
        "--output", "-o", help="Output ALTO XML file (default: stdout)",
    )
    parser.add_argument(
        "--units", choices=["pixel", "mm10"], default="pixel",
        help="ALTO measurement unit (default: pixel)",
    )
    parser.add_argument("--page-id", default="PAGE_1", help="Page ID")
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

    xml = convert_to_alto(
        data,
        image_path=args.image,
        regions=regions,
        measurement_unit=args.units,
        page_id=args.page_id,
    )

    if args.output:
        Path(args.output).write_text(xml, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(xml)


if __name__ == "__main__":
    _cli()
