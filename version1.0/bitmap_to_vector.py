#!/usr/bin/env python3
"""
bitmap_to_vector.py
====================================
将 JPG / PNG 位图描摹（Bitmap Tracing）为可编辑矢量图 SVG / EPS
核心原理：位图描摹（Potrace 算法）—— 像素 → 贝塞尔曲线路径

使用方法：
    python bitmap_to_vector.py input.jpg               # → SVG + EPS
    python bitmap_to_vector.py input.png --format svg  # 仅 SVG
    python bitmap_to_vector.py input.png --format eps  # 仅 EPS
    python bitmap_to_vector.py *.png --color           # 彩色分层描摹
    python bitmap_to_vector.py folder/ --color         # 批量处理整个文件夹

依赖：
    pip install Pillow numpy scikit-image
    apt install potrace   (或 brew install potrace / winget install potrace)
"""

import argparse
import os
import sys
import subprocess
import shutil
import tempfile
import struct
import zlib
import math
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    sys.exit("❌ 请先安装依赖：pip install Pillow numpy")

# ──────────────────────────────────────────────
# 0. 工具检测
# ──────────────────────────────────────────────
POTRACE = shutil.which("potrace")

def require_potrace():
    if not POTRACE:
        sys.exit(
            "❌ 未找到 potrace。\n"
            "   Linux : sudo apt install potrace\n"
            "   macOS : brew install potrace\n"
            "   Windows: https://potrace.sourceforge.net/"
        )

# ──────────────────────────────────────────────
# 1. 图像预处理
# ──────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    """加载图像，自动处理 EXIF 旋转、透明背景"""
    img = Image.open(path)
    # EXIF 方向修正
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # 透明背景 → 白底
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img)
        img = bg
    else:
        img = img.convert("RGB")
    return img


def preprocess_for_tracing(img: Image.Image, sharpen: bool = True) -> Image.Image:
    """
    预处理流水线：
      1. 上采样（提升描摹精度）
      2. 降噪
      3. 锐化边缘
      4. 转灰度
    """
    # 上采样到合适分辨率（过小会丢失细节）
    MAX_DIM = 2048
    w, h = img.size
    scale = min(MAX_DIM / max(w, h), 2.0)
    if scale > 1.05:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    if sharpen:
        from PIL import ImageFilter, ImageEnhance
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

    return img


def quantize_colors(img: Image.Image, n_colors: int = 8) -> list:
    """
    彩色分层：将图像量化为 n_colors 种颜色，
    返回 [(color_rgb, binary_mask_array), ...] 列表，
    用于逐层描摹（模拟 Adobe Live Trace 彩色模式）。
    """
    try:
        from sklearn.cluster import KMeans
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    arr = np.array(img.convert("RGB"))
    pixels = arr.reshape(-1, 3).astype(np.float32)

    if has_sklearn:
        km = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_.astype(np.uint8)
    else:
        # 回退：PIL 调色板量化
        q = img.convert("RGB").quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        label_arr = np.array(q)
        labels = label_arr.flatten()
        palette = np.array(q.getpalette()[:n_colors * 3]).reshape(n_colors, 3)
        centers = palette.astype(np.uint8)

    layers = []
    h, w = arr.shape[:2]
    label_map = labels.reshape(h, w)
    for idx in range(n_colors):
        color = tuple(centers[idx])
        mask = (label_map == idx).astype(np.uint8) * 255
        # 过滤极小色块（噪点）
        coverage = mask.sum() / (h * w * 255)
        if coverage < 0.001:
            continue
        layers.append((color, mask))

    # 按亮度降序（暗色在下，亮色在上）
    layers.sort(key=lambda x: -(x[0][0] * 0.299 + x[0][1] * 0.587 + x[0][2] * 0.114))
    return layers


# ──────────────────────────────────────────────
# 2. 描摹核心（Potrace 封装）
# ──────────────────────────────────────────────

def image_to_pbm(mask_array: np.ndarray, tmp_dir: str) -> str:
    """将二值掩码写成 PBM（Portable Bitmap）文件供 potrace 读取"""
    h, w = mask_array.shape
    pbm_path = os.path.join(tmp_dir, "layer.pbm")
    # PBM P4 格式（二进制）
    row_bytes = (w + 7) // 8
    with open(pbm_path, "wb") as f:
        header = f"P4\n{w} {h}\n".encode()
        f.write(header)
        for row in range(h):
            packed = bytearray(row_bytes)
            for col in range(w):
                if mask_array[row, col] < 128:  # 暗像素 = 前景（potrace 约定）
                    packed[col // 8] |= (0x80 >> (col % 8))
            f.write(bytes(packed))
    return pbm_path


def run_potrace(pbm_path: str, out_path: str, fmt: str,
                turdsize: int = 2, alphamax: float = 1.0,
                opttolerance: float = 0.2) -> bool:
    """
    调用 potrace 执行描摹。
    关键参数：
      turdsize     - 忽略的最小像素斑点面积（降噪）
      alphamax     - 曲线平滑度（0=全折线, 1.33=最光滑）
      opttolerance - 贝塞尔曲线拟合容差
    """
    fmt_flag = {"svg": "-s", "eps": "-e", "pdf": "-p"}.get(fmt, "-s")
    cmd = [
        POTRACE,
        fmt_flag,
        f"--turdsize={turdsize}",
        f"--alphamax={alphamax}",
        f"--opttolerance={opttolerance}",
        "-o", out_path,
        pbm_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ──────────────────────────────────────────────
# 3. SVG 合并（多色层 → 单 SVG）
# ──────────────────────────────────────────────

def extract_svg_paths(svg_text: str) -> str:
    """从 potrace 输出的 SVG 中提取 <path> 元素内容"""
    import re
    paths = re.findall(r'<path[^>]*/>', svg_text, re.DOTALL)
    if not paths:
        paths = re.findall(r'<path.*?</path>', svg_text, re.DOTALL)
    return "\n".join(paths)


def extract_svg_viewbox(svg_text: str) -> str:
    import re
    m = re.search(r'viewBox=["\']([^"\']+)["\']', svg_text)
    return m.group(1) if m else "0 0 100 100"


def build_color_svg(layers_svg: list, viewbox: str, width: int, height: int) -> str:
    """
    将多个单色描摹层合并为一个彩色 SVG。
    每层是一个独立 <g> 组，在 Illustrator 中可单独选中编辑。
    """
    defs_parts = []
    group_parts = []

    for i, (color, svg_text) in enumerate(layers_svg):
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        paths_raw = extract_svg_paths(svg_text)
        # 移除 potrace 默认黑色 fill，改用我们的颜色
        import re
        paths_raw = re.sub(r'fill="[^"]*"', '', paths_raw)
        paths_raw = re.sub(r"fill='[^']*'", '', paths_raw)
        group = (
            f'  <g id="layer_{i}" '
            f'inkscape:label="Color Layer {i}" '
            f'fill="{hex_color}" '
            f'stroke="none">\n'
            f'    {paths_raw}\n'
            f'  </g>'
        )
        group_parts.append(group)

    vb_vals = viewbox.split()
    w_pt = vb_vals[2] if len(vb_vals) >= 4 else str(width)
    h_pt = vb_vals[3] if len(vb_vals) >= 4 else str(height)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     viewBox="{viewbox}"
     width="{w_pt}pt" height="{h_pt}pt"
     preserveAspectRatio="xMidYMid meet">
  <title>Traced Vector Image</title>
  <desc>Generated by bitmap_to_vector.py using Potrace tracing algorithm</desc>
{chr(10).join(group_parts)}
</svg>"""
    return svg


# ──────────────────────────────────────────────
# 4. EPS 生成（Adobe Illustrator 兼容）
# ──────────────────────────────────────────────

def svg_path_to_eps_path(path_d: str, color_rgb: tuple) -> str:
    """
    将 SVG path d 属性转换为 PostScript 路径指令。
    支持：M L C Z（potrace 仅输出这四种命令）
    """
    import re
    r, g, b = [c / 255 for c in color_rgb]
    tokens = re.findall(r'[MLCZmlcz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', path_d)
    ps_cmds = [f"{r:.4f} {g:.4f} {b:.4f} setrgbcolor", "newpath"]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == 'M':
            i += 1
            x, y = tokens[i], tokens[i+1]; i += 2
            ps_cmds.append(f"{x} {y} moveto")
        elif t == 'L':
            i += 1
            x, y = tokens[i], tokens[i+1]; i += 2
            ps_cmds.append(f"{x} {y} lineto")
        elif t == 'C':
            i += 1
            coords = tokens[i:i+6]; i += 6
            ps_cmds.append(f"{coords[0]} {coords[1]} {coords[2]} {coords[3]} {coords[4]} {coords[5]} curveto")
        elif t == 'Z' or t == 'z':
            ps_cmds.append("closepath")
            i += 1
        else:
            i += 1  # skip unknown
    ps_cmds.append("fill")
    return "\n".join(ps_cmds)


def build_eps(layers_svg_paths: list, width: int, height: int) -> str:
    """
    生成 Adobe Illustrator 兼容的 EPS 文件。
    包含 AI 注释头，支持直接在 AI 中打开并编辑路径。
    """
    import re

    header = f"""%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 {width} {height}
%%HiResBoundingBox: 0 0 {width} {height}
%%Title: Traced Vector Image
%%Creator: bitmap_to_vector.py (Potrace)
%%CreationDate: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
%%DocumentData: Clean7Bit
%%LanguageLevel: 2
%%Pages: 1
%%EndComments

%%BeginProlog
/bd {{ bind def }} bind def
%%EndProlog

%%Page: 1 1
gsave
0 {height} translate
1 -1 scale
"""

    body_parts = []
    for i, (color, svg_text) in enumerate(layers_svg_paths):
        paths = re.findall(r'd="([^"]+)"', svg_text)
        if not paths:
            paths = re.findall(r"d='([^']+)'", svg_text)
        body_parts.append(f"% Layer {i}: rgb{color}")
        for pd in paths:
            body_parts.append(svg_path_to_eps_path(pd, color))

    footer = """grestore
showpage
%%Trailer
%%EOF"""

    return header + "\n".join(body_parts) + "\n" + footer


# ──────────────────────────────────────────────
# 5. 单色（黑白）描摹
# ──────────────────────────────────────────────

def trace_monochrome(img: Image.Image, out_stem: str,
                     fmt: str = "both",
                     threshold: int = 128,
                     turdsize: int = 2,
                     alphamax: float = 1.0,
                     opttolerance: float = 0.2):
    """黑白描摹（速度最快，适合 icon、线条图）"""
    require_potrace()
    gray = img.convert("L")
    arr = np.array(gray)
    mask = (arr < threshold).astype(np.uint8) * 255  # 暗像素 = 前景

    with tempfile.TemporaryDirectory() as tmp:
        pbm = image_to_pbm(mask, tmp)

        results = {}
        for f in (["svg", "eps"] if fmt == "both" else [fmt]):
            out = f"{out_stem}.{f}"
            tmp_out = os.path.join(tmp, f"out.{f}")
            if run_potrace(pbm, tmp_out, f, turdsize, alphamax, opttolerance):
                shutil.copy(tmp_out, out)
                results[f] = out
                print(f"  ✅ {f.upper()} → {out}")
            else:
                print(f"  ❌ potrace 失败（{f}）")
    return results


# ──────────────────────────────────────────────
# 6. 彩色分层描摹
# ──────────────────────────────────────────────

def trace_color(img: Image.Image, out_stem: str,
                fmt: str = "both",
                n_colors: int = 8,
                turdsize: int = 2,
                alphamax: float = 1.0,
                opttolerance: float = 0.2):
    """彩色分层描摹 → 可编辑多色 SVG / EPS"""
    require_potrace()
    print(f"  🎨 颜色量化（{n_colors} 色）…")
    layers = quantize_colors(img, n_colors)
    print(f"  📐 检测到 {len(layers)} 个有效色层")

    w, h = img.size
    layers_svg = []
    first_viewbox = None

    with tempfile.TemporaryDirectory() as tmp:
        for idx, (color, mask) in enumerate(layers):
            print(f"  ↳ 描摹色层 {idx+1}/{len(layers)}  rgb{color} …", end=" ", flush=True)
            pbm = image_to_pbm(mask, tmp)
            tmp_svg = os.path.join(tmp, f"layer_{idx}.svg")
            ok = run_potrace(pbm, tmp_svg, "svg", turdsize, alphamax, opttolerance)
            if ok and os.path.exists(tmp_svg):
                svg_text = open(tmp_svg, encoding="utf-8").read()
                if first_viewbox is None:
                    first_viewbox = extract_svg_viewbox(svg_text)
                layers_svg.append((color, svg_text))
                print("✓")
            else:
                print("skip（无路径）")

    if not layers_svg:
        print("  ⚠️ 未生成任何路径，回退为单色描摹")
        return trace_monochrome(img, out_stem, fmt, turdsize=turdsize,
                                alphamax=alphamax, opttolerance=opttolerance)

    viewbox = first_viewbox or f"0 0 {w} {h}"
    results = {}

    if fmt in ("svg", "both"):
        svg_content = build_color_svg(layers_svg, viewbox, w, h)
        svg_out = f"{out_stem}.svg"
        with open(svg_out, "w", encoding="utf-8") as f:
            f.write(svg_content)
        results["svg"] = svg_out
        print(f"  ✅ SVG → {svg_out}")

    if fmt in ("eps", "both"):
        eps_content = build_eps(layers_svg, w, h)
        eps_out = f"{out_stem}.eps"
        with open(eps_out, "w", encoding="utf-8") as f:
            f.write(eps_content)
        results["eps"] = eps_out
        print(f"  ✅ EPS → {eps_out}")

    return results


# ──────────────────────────────────────────────
# 7. 主入口
# ──────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def collect_inputs(args_input: list) -> list:
    """收集所有待处理的图像文件路径"""
    files = []
    for inp in args_input:
        p = Path(inp)
        if p.is_dir():
            for ext in SUPPORTED_EXT:
                files.extend(p.glob(f"*{ext}"))
                files.extend(p.glob(f"*{ext.upper()}"))
        elif p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
        else:
            # 通配符已由 shell 展开；如果未展开则提示
            print(f"  ⚠️ 跳过不支持的文件：{inp}")
    return sorted(set(files))


def process_file(input_path: Path, args) -> None:
    print(f"\n{'='*60}")
    print(f"📂 处理：{input_path.name}")

    # 输出目录
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = str(out_dir / input_path.stem)

    # 加载 & 预处理
    print("  🔧 加载并预处理图像…")
    img = load_image(str(input_path))
    img = preprocess_for_tracing(img, sharpen=not args.no_sharpen)
    print(f"  📏 尺寸：{img.size[0]}×{img.size[1]} px")

    # 描摹
    if args.color:
        trace_color(img, out_stem,
                    fmt=args.format,
                    n_colors=args.colors,
                    turdsize=args.turdsize,
                    alphamax=args.alphamax,
                    opttolerance=args.opttolerance)
    else:
        trace_monochrome(img, out_stem,
                         fmt=args.format,
                         threshold=args.threshold,
                         turdsize=args.turdsize,
                         alphamax=args.alphamax,
                         opttolerance=args.opttolerance)


def main():
    parser = argparse.ArgumentParser(
        description="位图描摹工具：JPG/PNG → SVG/EPS 可编辑矢量图（基于 Potrace 算法）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="+",
                        help="输入图像文件（JPG/PNG）或文件夹")
    parser.add_argument("--format", choices=["svg", "eps", "both"], default="both",
                        help="输出格式（默认：both）")
    parser.add_argument("--color", action="store_true",
                        help="彩色分层描摹（适合彩色图标；默认为黑白描摹）")
    parser.add_argument("--colors", type=int, default=8, metavar="N",
                        help="彩色模式的颜色层数（默认：8）")
    parser.add_argument("--threshold", type=int, default=128, metavar="0-255",
                        help="黑白二值化阈值（默认：128）")
    parser.add_argument("--turdsize", type=int, default=2,
                        help="忽略的最小斑点面积（降噪，默认：2）")
    parser.add_argument("--alphamax", type=float, default=1.0,
                        help="曲线平滑度 0~1.33（默认：1.0）")
    parser.add_argument("--opttolerance", type=float, default=0.2,
                        help="贝塞尔拟合容差（默认：0.2，越小越精确）")
    parser.add_argument("--no-sharpen", action="store_true",
                        help="禁用预处理锐化")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="统一输出目录（默认与原文件同目录）")
    args = parser.parse_args()

    files = collect_inputs(args.input)
    if not files:
        sys.exit("❌ 未找到任何支持的图像文件。")

    print(f"🚀 共找到 {len(files)} 个文件，开始处理…")
    ok, fail = 0, 0
    for fp in files:
        try:
            process_file(fp, args)
            ok += 1
        except Exception as e:
            print(f"  ❌ 处理失败：{e}")
            fail += 1

    print(f"\n{'='*60}")
    print(f"✅ 完成：成功 {ok} 个  {'，失败 ' + str(fail) + ' 个' if fail else ''}")


if __name__ == "__main__":
    main()
