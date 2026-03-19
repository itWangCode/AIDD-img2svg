#!/usr/bin/env python3
"""
bitmap_to_vector.py  v2.0
=================================================
JPG / PNG  →  SVG / EPS  可编辑矢量图
核心：位图描摹（Bitmap Tracing）—— Potrace 算法
      像素轮廓 → 贝塞尔曲线路径

修复说明（v2.0）：
  - 正确保留 potrace <g transform> 坐标系，颜色不再丢失
  - 每个色层用独立 <g> 包裹，AI 可逐层选中编辑
  - EPS 正确嵌入 RGB setrgbcolor，Illustrator 直接打开可编辑
  - numpy uint8 类型全部转 int，避免 hex 格式化失败

用法：
    python bitmap_to_vector.py input.jpg
    python bitmap_to_vector.py input.png --color --colors 12
    python bitmap_to_vector.py folder/  --color --format svg
    python bitmap_to_vector.py *.png    --format eps
"""

import argparse
import os
import re
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageFilter
    import numpy as np
except ImportError:
    sys.exit("请先安装：pip install Pillow numpy")

# 全局：potrace 路径
POTRACE = shutil.which("potrace")

def require_potrace():
    if not POTRACE:
        sys.exit(
            "未找到 potrace。\n"
            "   Linux : sudo apt install potrace\n"
            "   macOS : brew install potrace\n"
            "   Windows: https://potrace.sourceforge.net/"
        )


# ══════════════════════════════════════════════════════════
# 1. 图像加载与预处理
# ══════════════════════════════════════════════════════════

def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    return img


def preprocess(img: Image.Image, max_dim: int = 2048) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / longest
    elif longest < 512:
        scale = 512 / longest
    else:
        scale = 1.0
    if abs(scale - 1.0) > 0.02:
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))
    return img


# ══════════════════════════════════════════════════════════
# 2. 颜色量化
# ══════════════════════════════════════════════════════════

def quantize_colors(img: Image.Image, n_colors: int) -> list:
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)

    # K-Means (纯 numpy)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pixels), n_colors, replace=False)
    centers = pixels[idx].copy()

    for _ in range(40):
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(n_colors)
        ])
        if np.allclose(centers, new_centers, atol=0.5):
            break
        centers = new_centers

    centers_int = np.clip(new_centers, 0, 255).astype(np.uint8)
    label_map = labels.reshape(h, w)

    layers = []
    for k in range(n_colors):
        mask = (label_map == k).astype(np.uint8) * 255
        coverage = mask.sum() / (h * w * 255)
        if coverage < 0.0005:
            continue
        r = int(centers_int[k, 0])
        g = int(centers_int[k, 1])
        b = int(centers_int[k, 2])
        layers.append(((r, g, b), mask))

    layers.sort(
        key=lambda x: x[0][0] * 0.299 + x[0][1] * 0.587 + x[0][2] * 0.114,
        reverse=True
    )
    return layers


# ══════════════════════════════════════════════════════════
# 3. Potrace 封装
# ══════════════════════════════════════════════════════════

def mask_to_pbm(mask: np.ndarray, path: str):
    h, w = mask.shape
    row_bytes = (w + 7) // 8
    with open(path, "wb") as f:
        f.write(f"P4\n{w} {h}\n".encode())
        for row in range(h):
            packed = bytearray(row_bytes)
            for col in range(w):
                if mask[row, col] >= 128:
                    packed[col // 8] |= 0x80 >> (col % 8)
            f.write(bytes(packed))


def run_potrace_svg(pbm: str, svg: str,
                    turdsize: int = 2,
                    alphamax: float = 1.0,
                    opttolerance: float = 0.2) -> bool:
    cmd = [
        POTRACE, "-s",
        f"--turdsize={turdsize}",
        f"--alphamax={alphamax}",
        f"--opttolerance={opttolerance}",
        "-o", svg, pbm,
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and os.path.exists(svg)


# ══════════════════════════════════════════════════════════
# 4. SVG 解析与重组
# ══════════════════════════════════════════════════════════

def parse_potrace_svg(svg_text: str) -> dict:
    """
    解析 potrace SVG，提取：
      viewbox, width, height, transform（坐标系！）, paths
    """
    info = {}

    m = re.search(r'viewBox=["\']([^"\']+)["\']', svg_text)
    info["viewbox"] = m.group(1) if m else "0 0 100 100"

    m = re.search(r'<svg[^>]+width=["\']([^"\']+)["\']', svg_text)
    info["width"] = m.group(1) if m else "100pt"

    m = re.search(r'<svg[^>]+height=["\']([^"\']+)["\']', svg_text)
    info["height"] = m.group(1) if m else "100pt"

    # 关键：potrace 用 <g transform="translate(0,H) scale(0.1,-0.1)">
    # 必须保留，否则坐标系错误 → 图形显示为黑色矩形或空白
    m = re.search(r'<g\s+transform=["\']([^"\']+)["\']', svg_text)
    info["transform"] = m.group(1) if m else ""

    info["paths"] = re.findall(r'<path\s+d=["\']([^"\']+)["\']', svg_text)
    return info


def build_svg(layers_info: list) -> str:
    if not layers_info:
        return ""

    ref = layers_info[0][1]
    viewbox   = ref["viewbox"]
    width     = ref["width"]
    height    = ref["height"]
    transform = ref["transform"]

    groups = []
    for idx, (color, info) in enumerate(layers_info):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        if not info["paths"]:
            continue
        path_elems = "\n    ".join(f'<path d="{d}"/>' for d in info["paths"])
        g_attr = f'transform="{transform}"' if transform else ""
        groups.append(
            f'  <g id="layer_{idx}" inkscape:label="Layer {idx} {hex_color}" '
            f'fill="{hex_color}" stroke="none" {g_attr}>\n'
            f'    {path_elems}\n'
            f'  </g>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        '<svg version="1.1"\n'
        '     xmlns="http://www.w3.org/2000/svg"\n'
        '     xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n'
        f'     viewBox="{viewbox}"\n'
        f'     width="{width}" height="{height}"\n'
        '     preserveAspectRatio="xMidYMid meet">\n'
        '  <title>Traced Vector Image</title>\n'
        '  <desc>bitmap_to_vector.py v2.0 — Potrace tracing</desc>\n'
        + "\n".join(groups) + "\n"
        "</svg>\n"
    )


# ══════════════════════════════════════════════════════════
# 5. EPS 生成
# ══════════════════════════════════════════════════════════

def path_d_to_ps(d: str) -> str:
    tokens = re.findall(
        r'[MLCZmlcz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d
    )
    ps = ["newpath"]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in "Mm":
            i += 1
            if i + 1 >= len(tokens):
                break
            ps.append(f"{tokens[i]} {tokens[i+1]} moveto"); i += 2
        elif t in "Ll":
            i += 1
            if i + 1 >= len(tokens):
                break
            ps.append(f"{tokens[i]} {tokens[i+1]} lineto"); i += 2
        elif t in "Cc":
            i += 1
            if i + 5 >= len(tokens):
                break
            c = tokens[i:i+6]; i += 6
            ps.append(f"{c[0]} {c[1]} {c[2]} {c[3]} {c[4]} {c[5]} curveto")
        elif t in "Zz":
            ps.append("closepath"); i += 1
        else:
            i += 1
    ps.append("fill")
    return "\n".join(ps)


def build_eps(layers_info: list, img_w: int, img_h: int) -> str:
    import datetime
    if not layers_info:
        return ""

    ref_info = layers_info[0][1]
    vb = ref_info["viewbox"].split()
    vb_w = float(vb[2]) if len(vb) >= 4 else float(img_w)
    vb_h = float(vb[3]) if len(vb) >= 4 else float(img_h)

    # 解析 potrace transform 数值
    ref_transform = ref_info.get("transform", "")
    tm = re.findall(r'[-+]?\d+\.?\d*', ref_transform)
    if len(tm) >= 4:
        tx, ty = float(tm[0]), float(tm[1])
        sx, sy = float(tm[2]), float(tm[3])
    else:
        tx, ty = 0.0, vb_h
        sx, sy = 0.1, -0.1

    header = (
        "%!PS-Adobe-3.0 EPSF-3.0\n"
        f"%%BoundingBox: 0 0 {int(vb_w)} {int(vb_h)}\n"
        f"%%HiResBoundingBox: 0 0 {vb_w:.3f} {vb_h:.3f}\n"
        "%%Title: Traced Vector Image\n"
        "%%Creator: bitmap_to_vector.py v2.0\n"
        f"%%CreationDate: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "%%DocumentData: Clean7Bit\n"
        "%%LanguageLevel: 2\n"
        "%%Pages: 1\n"
        "%%EndComments\n\n"
        "%%Page: 1 1\n"
        "gsave\n"
        f"{tx} {ty} translate\n"
        f"{sx} {sy} scale\n"
    )

    body_lines = []
    for idx, (color, info) in enumerate(layers_info):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        if not info["paths"]:
            continue
        body_lines.append(f"\n% Layer {idx}  rgb({r},{g},{b})")
        body_lines.append(f"{rn:.4f} {gn:.4f} {bn:.4f} setrgbcolor")
        for d in info["paths"]:
            body_lines.append(path_d_to_ps(d))

    footer = "\ngrestore\nshowpage\n%%Trailer\n%%EOF\n"
    return header + "\n".join(body_lines) + footer


# ══════════════════════════════════════════════════════════
# 6. 单色描摹
# ══════════════════════════════════════════════════════════

def trace_monochrome(img, out_stem, fmt, threshold, turdsize, alphamax, opttolerance):
    require_potrace()
    gray = np.array(img.convert("L"))
    mask = (gray < threshold).astype(np.uint8) * 255

    with tempfile.TemporaryDirectory() as tmp:
        pbm = os.path.join(tmp, "mono.pbm")
        mask_to_pbm(mask, pbm)
        tmp_svg = os.path.join(tmp, "mono.svg")

        if not run_potrace_svg(pbm, tmp_svg, turdsize, alphamax, opttolerance):
            print("  potrace 描摹失败"); return {}

        svg_text = open(tmp_svg, encoding="utf-8").read()
        svg_info = parse_potrace_svg(svg_text)
        layers_info = [((0, 0, 0), svg_info)]

        results = {}
        if fmt in ("svg", "both"):
            out = f"{out_stem}.svg"
            open(out, "w", encoding="utf-8").write(build_svg(layers_info))
            results["svg"] = out
            print(f"  SVG -> {out}")
        if fmt in ("eps", "both"):
            out = f"{out_stem}.eps"
            open(out, "w", encoding="utf-8").write(build_eps(layers_info, *img.size))
            results["eps"] = out
            print(f"  EPS -> {out}")
    return results


# ══════════════════════════════════════════════════════════
# 7. 彩色分层描摹（核心）
# ══════════════════════════════════════════════════════════

def trace_color(img, out_stem, fmt, n_colors, turdsize, alphamax, opttolerance):
    require_potrace()
    print(f"  颜色量化（{n_colors} 色）…")
    layers = quantize_colors(img, n_colors)
    print(f"  有效色层：{len(layers)} 层")

    layers_info = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, (color, mask) in enumerate(layers):
            r, g, b = color
            print(f"  Layer {idx+1:02d}/{len(layers):02d}  "
                  f"#{r:02x}{g:02x}{b:02x} rgb({r},{g},{b}) ... ",
                  end="", flush=True)
            pbm = os.path.join(tmp, f"l{idx}.pbm")
            svg = os.path.join(tmp, f"l{idx}.svg")
            mask_to_pbm(mask, pbm)
            if run_potrace_svg(pbm, svg, turdsize, alphamax, opttolerance):
                info = parse_potrace_svg(open(svg, encoding="utf-8").read())
                if info["paths"]:
                    layers_info.append(((r, g, b), info))
                    print(f"ok ({len(info['paths'])} paths)")
                else:
                    print("skip (no paths)")
            else:
                print("skip (potrace failed)")

    if not layers_info:
        print("  No valid paths, fallback to monochrome")
        return trace_monochrome(img, out_stem, fmt, 128, turdsize, alphamax, opttolerance)

    results = {}
    if fmt in ("svg", "both"):
        out = f"{out_stem}.svg"
        open(out, "w", encoding="utf-8").write(build_svg(layers_info))
        results["svg"] = out
        print(f"  SVG -> {out}")
    if fmt in ("eps", "both"):
        out = f"{out_stem}.eps"
        open(out, "w", encoding="utf-8").write(build_eps(layers_info, *img.size))
        results["eps"] = out
        print(f"  EPS -> {out}")
    return results


# ══════════════════════════════════════════════════════════
# 8. 主入口
# ══════════════════════════════════════════════════════════

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def collect_files(inputs):
    files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in SUPPORTED:
                files.extend(p.glob(f"*{ext}"))
                files.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        else:
            print(f"  skip: {inp}")
    return sorted(set(files))


def process_file(path, args):
    out_dir = Path(args.output_dir) if args.output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = str(out_dir / path.stem)

    img = load_image(str(path))
    img = preprocess(img, max_dim=args.max_dim)
    print(f"  size: {img.size[0]} x {img.size[1]} px")

    kw = dict(turdsize=args.turdsize, alphamax=args.alphamax,
               opttolerance=args.opttolerance)
    if args.color:
        trace_color(img, out_stem, args.format, args.colors, **kw)
    else:
        trace_monochrome(img, out_stem, args.format,
                         threshold=args.threshold, **kw)


def main():
    p = argparse.ArgumentParser(
        description="bitmap_to_vector.py v2.0 — JPG/PNG to SVG/EPS vector tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("input", nargs="+", help="Input image files or folder")
    p.add_argument("--format", choices=["svg", "eps", "both"], default="both")
    p.add_argument("--color", action="store_true",
                   help="Color layer tracing (recommended for colorful images)")
    p.add_argument("--colors", type=int, default=8, help="Number of color layers (default 8)")
    p.add_argument("--threshold", type=int, default=128,
                   help="Threshold for B&W mode (default 128)")
    p.add_argument("--turdsize", type=int, default=2)
    p.add_argument("--alphamax", type=float, default=1.0)
    p.add_argument("--opttolerance", type=float, default=0.2)
    p.add_argument("--max-dim", type=int, default=2048)
    p.add_argument("--output-dir", metavar="DIR")
    args = p.parse_args()

    files = collect_files(args.input)
    if not files:
        sys.exit("No supported image files found.")

    print(f"\n=== bitmap_to_vector.py v2.0 ===")
    print(f"Files: {len(files)}  Mode: {'COLOR' if args.color else 'MONO'}"
          f"  Format: {args.format.upper()}")
    ok = fail = 0
    for fp in files:
        print(f"\n[{ok+fail+1}/{len(files)}] {fp.name}")
        try:
            process_file(fp, args)
            ok += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            fail += 1

    print(f"\n=== Done: {ok} ok" + (f", {fail} failed" if fail else "") + " ===")


if __name__ == "__main__":
    main()
