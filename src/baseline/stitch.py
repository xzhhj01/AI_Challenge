"""OCR 텍스트 박스를 간판 단위로 묶어 확대 패널을 만들고, 원본 아래에 이어붙인 한 장을 생성."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

STITCH_NOTE = "이미지 구성: 맨 위는 전체 사진이고, 검은 구분선 아래는 글자 영역을 확대한 패널입니다."

Box = Tuple[float, float, float, float]


def _expand(b: Box, dx: float, dy: float) -> Box:
    return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)


def _overlap(a: Box, b: Box) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def cluster_boxes(items: Sequence[Dict], img_w: int, img_h: int, gap_ratio: float = 0.03) -> List[Box]:
    """가까운 텍스트 박스를 병합. 글자 면적 합이 큰 군집(= 주 간판·표제)부터 반환."""
    dx, dy = img_w * gap_ratio, img_h * gap_ratio
    clusters = [(tuple(it["bbox"]), (it["bbox"][2] - it["bbox"][0]) * (it["bbox"][3] - it["bbox"][1]))
                for it in items if it.get("bbox")]
    merged = True
    while merged:
        merged = False
        out: List[Tuple[Box, float]] = []
        for box, area in clusters:
            for i, (ob, oa) in enumerate(out):
                if _overlap(_expand(box, dx, dy), _expand(ob, dx, dy)):
                    out[i] = (_union(box, ob), oa + area)
                    merged = True
                    break
            else:
                out.append((box, area))
        clusters = out
    clusters.sort(key=lambda c: c[1], reverse=True)
    return [c[0] for c in clusters]


def make_stitched(img: Image.Image, items: Sequence[Dict], max_panels: int = 2,
                  sep: int = 16, margin: float = 0.1, max_upscale: float = 3.0) -> Optional[Image.Image]:
    """원본 + 확대 패널(최대 max_panels개)을 세로로 이어붙인 한 장. 박스가 없으면 None."""
    W, H = img.size
    boxes = cluster_boxes(items, W, H)[:max_panels]
    if not boxes:
        return None
    panels = []
    for x1, y1, x2, y2 in boxes:
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        crop = img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx)), min(H, int(y2 + my))))
        cw, ch = crop.size
        if cw < 2 or ch < 2:
            continue
        scale = min(W / cw, max_upscale)
        panels.append(crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.BICUBIC))
    if not panels:
        return None
    total_h = H + sum(p.size[1] + sep for p in panels)
    canvas = Image.new("RGB", (W, total_h), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = H
    for p in panels:
        draw.rectangle([0, y, W, y + sep - 1], fill="black")   # 굵은 구분선
        y += sep
        canvas.paste(p, ((W - p.size[0]) // 2, y))
        y += p.size[1]
    return canvas
