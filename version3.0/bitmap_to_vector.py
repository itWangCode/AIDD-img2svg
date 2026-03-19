#!/usr/bin/env python3
"""
bitmap_to_vector.py  v3.0
================================================
JPG/PNG → 高精度可编辑 SVG/EPS 矢量图

模块架构：
  M1  ImageLoader      图像加载 + 预处理
  M2  ColorAnalyzer    K-Means 颜色分层
  M3  OCREngine        文字识别 → Times New Roman
  M4  PotracTracer     Potrace 贝塞尔描摹
  M5  SVGAssembler     SVG 智能组装（分组可移动）
  M6  EPSExporter      AI 兼容 EPS 导出
  M7  BatchProcessor   批量入口

用法：
    python bitmap_to_vector.py Fig1.png --color --colors 12 --ocr
    python bitmap_to_vector.py folder/  --color --format svg
    python bitmap_to_vector.py *.jpg    --format eps --ocr
"""

import argparse, os, re, sys, subprocess, shutil, tempfile, math
from pathlib import Path

# ─── 依赖检测 ────────────────────────────────────────────
try:
    from PIL import Image, ImageFilter
    import numpy as np
except ImportError:
    sys.exit("请安装: pip install Pillow numpy")

POTRACE = shutil.which("potrace")
HAS_CV2 = HAS_OCR = HAS_SKLEARN = False

try:
    import cv2; HAS_CV2 = True
except ImportError:
    pass
try:
    import pytesseract; pytesseract.get_tesseract_version(); HAS_OCR = True
except Exception:
    pass
try:
    from sklearn.cluster import KMeans as _SKLearnKMeans; HAS_SKLEARN = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════
# M1  ImageLoader
# ══════════════════════════════════════════════════════════
class ImageLoader:
    @staticmethod
    def load(path: str) -> Image.Image:
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

    @staticmethod
    def preprocess(img: Image.Image, max_dim=2048, sharpen=True):
        w, h = img.size
        longest = max(w, h)
        scale = (max_dim / longest if longest > max_dim
                 else 600 / longest if longest < 600
                 else 1.0)
        if abs(scale - 1.0) > 0.02:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=2))
        return img


# ══════════════════════════════════════════════════════════
# M2  ColorAnalyzer
# ══════════════════════════════════════════════════════════
class ColorAnalyzer:
    @staticmethod
    def _kmeans_np(pixels, k, iters=50):
        rng = np.random.default_rng(0)
        centers = pixels[rng.choice(len(pixels), k, replace=False)].astype(np.float64)
        for _ in range(iters):
            d = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
            labels = np.argmin(d, axis=1)
            new_c = np.array([
                pixels[labels == j].mean(0) if (labels == j).any() else centers[j]
                for j in range(k)
            ])
            if np.allclose(centers, new_c, atol=0.4):
                break
            centers = new_c
        return labels, np.clip(new_c, 0, 255).astype(np.uint8)

    @classmethod
    def quantize(cls, img: Image.Image, n: int) -> list:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        pixels = arr.reshape(-1, 3)
        if HAS_SKLEARN:
            km = _SKLearnKMeans(n_clusters=n, n_init=10, random_state=0, max_iter=300)
            labels = km.fit_predict(pixels)
            centers = km.cluster_centers_.astype(np.uint8)
        else:
            labels, centers = cls._kmeans_np(pixels, n)
        label_map = labels.reshape(h, w)
        layers = []
        for k in range(n):
            mask = (label_map == k).astype(np.uint8) * 255
            if mask.sum() < h * w * 255 * 0.0003:
                continue
            r, g, b = int(centers[k, 0]), int(centers[k, 1]), int(centers[k, 2])
            layers.append(((r, g, b), mask))
        layers.sort(key=lambda x: x[0][0]*0.299 + x[0][1]*0.587 + x[0][2]*0.114, reverse=True)
        return layers

    @staticmethod
    def enhance_mask(mask: np.ndarray, size=2) -> np.ndarray:
        if not HAS_CV2:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return mask


# ══════════════════════════════════════════════════════════
# M3  OCREngine
# ══════════════════════════════════════════════════════════
class OCRResult:
    __slots__ = ['text', 'x', 'y', 'w', 'h', 'conf', 'font_size']
    def __init__(self, text, x, y, w, h, conf):
        self.text, self.x, self.y, self.w, self.h, self.conf = text, x, y, w, h, conf
        self.font_size = max(8, int(h * 0.85))


class OCREngine:
    @staticmethod
    def detect(img: Image.Image) -> list:
        if not HAS_OCR:
            return []
        try:
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT,
                config='--psm 11 -c preserve_interword_spaces=1')
            return [
                OCRResult(data['text'][i].strip(),
                          data['left'][i], data['top'][i],
                          data['width'][i], data['height'][i],
                          int(data['conf'][i]))
                for i in range(len(data['text']))
                if data['text'][i].strip() and int(data['conf'][i]) >= 40
            ]
        except Exception as e:
            print(f"  [OCR] {e}")
            return []

    @staticmethod
    def sample_color(arr: np.ndarray, r: OCRResult) -> tuple:
        x1 = max(0, r.x); y1 = max(0, r.y)
        x2 = min(arr.shape[1], r.x + r.w); y2 = min(arr.shape[0], r.y + r.h)
        if x2 <= x1 or y2 <= y1:
            return (30, 30, 30)
        pix = arr[y1:y2, x1:x2].reshape(-1, 3)
        lum = pix[:, 0]*0.299 + pix[:, 1]*0.587 + pix[:, 2]*0.114
        dark = pix[lum < 180]
        return tuple(dark.mean(0).astype(int)) if len(dark) >= 5 else (30, 30, 30)


# ══════════════════════════════════════════════════════════
# M4  PotracTracer
# ══════════════════════════════════════════════════════════
class TraceResult:
    __slots__ = ['color', 'paths', 'transform', 'viewbox', 'width', 'height']
    def __init__(self, color, paths, transform, viewbox, width, height):
        self.color = color; self.paths = paths; self.transform = transform
        self.viewbox = viewbox; self.width = width; self.height = height


class PotracTracer:
    def __init__(self, turdsize=2, alphamax=1.0, opttolerance=0.2):
        if not POTRACE:
            sys.exit("未找到 potrace。安装: sudo apt install potrace")
        self.turdsize = turdsize
        self.alphamax = alphamax
        self.opttolerance = opttolerance

    def _pbm(self, mask, path):
        h, w = mask.shape
        rb = (w + 7) // 8
        with open(path, "wb") as f:
            f.write(f"P4\n{w} {h}\n".encode())
            for row in range(h):
                packed = bytearray(rb)
                for col in range(w):
                    if mask[row, col] >= 128:
                        packed[col // 8] |= 0x80 >> (col % 8)
                f.write(bytes(packed))

    def _run(self, pbm, svg):
        cmd = [POTRACE, "-s",
               f"--turdsize={self.turdsize}",
               f"--alphamax={self.alphamax}",
               f"--opttolerance={self.opttolerance}",
               "-o", svg, pbm]
        r = subprocess.run(cmd, capture_output=True)
        return r.returncode == 0 and os.path.exists(svg)

    def _parse(self, text):
        def _g(pat, s, grp=1, default=""):
            m = re.search(pat, s)
            return m.group(grp) if m else default
        return {
            "viewbox":   _g(r'viewBox=["\']([^"\']+)["\']', text),
            "width":     _g(r'<svg[^>]+width=["\']([^"\']+)["\']', text, default="100pt"),
            "height":    _g(r'<svg[^>]+height=["\']([^"\']+)["\']', text, default="100pt"),
            "transform": _g(r'<g\s+transform=["\']([^"\']+)["\']', text),
            "paths":     re.findall(r'<path\s+d=["\']([^"\']+)["\']', text),
        }

    def trace_mask(self, mask, color, tmp, tag="l") -> "TraceResult | None":
        pbm = os.path.join(tmp, f"{tag}.pbm")
        svg = os.path.join(tmp, f"{tag}.svg")
        self._pbm(mask, pbm)
        if not self._run(pbm, svg):
            return None
        info = self._parse(open(svg, encoding="utf-8").read())
        if not info["paths"]:
            return None
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        return TraceResult((r, g, b), info["paths"], info["transform"],
                           info["viewbox"], info["width"], info["height"])

    def trace_layers(self, layers, verbose=True):
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (color, mask) in enumerate(layers):
                r, g, b = color
                if verbose:
                    print(f"  [{i+1:02d}/{len(layers):02d}] #{r:02x}{g:02x}{b:02x} "
                          f"rgb({r:3d},{g:3d},{b:3d}) ... ", end="", flush=True)
                enh = ColorAnalyzer.enhance_mask(mask)
                tr = self.trace_mask(enh, color, tmp, f"c{i}")
                if tr:
                    results.append(tr)
                    if verbose:
                        print(f"✓ ({len(tr.paths)} paths)")
                else:
                    if verbose:
                        print("skip")
        return results

    def trace_mono(self, img, threshold=128):
        gray = np.array(img.convert("L"))
        mask = (gray < threshold).astype(np.uint8) * 255
        with tempfile.TemporaryDirectory() as tmp:
            return self.trace_mask(mask, (0, 0, 0), tmp, "mono")


# ══════════════════════════════════════════════════════════
# M5  SVGAssembler
# ══════════════════════════════════════════════════════════
class SVGAssembler:
    """
    组装规则：
    ① 每个色层 = 独立 <g inkscape:groupmode="layer"> 可整体移动
    ② 贝塞尔路径在 <g> 内，颜色写在 <g fill="…"> 上
    ③ potrace transform 完整保留（坐标系正确）
    ④ OCR 文字 → <text> 节点，Times New Roman，可直接编辑
    ⑤ 文字层在最顶层，不遮挡图形选取
    """

    FONT = '"Times New Roman", Times, serif'

    def __init__(self, ocr=None, arr=None):
        self.ocr = ocr or []
        self.arr = arr

    @staticmethod
    def _hex(c): return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    def _tc(self, r):
        if self.arr is not None:
            return self._hex(OCREngine.sample_color(self.arr, r))
        return "#1a1a2e"

    def _text_nodes(self, vb_w, vb_h):
        if not (self.ocr and self.arr is not None):
            return ""
        oh, ow = self.arr.shape[:2]
        sx, sy = vb_w / ow, vb_h / oh
        parts = []
        for r in self.ocr:
            x  = r.x * sx
            # baseline = bottom of bounding box
            y  = (r.y + r.h) * sy
            fs = max(6, r.font_size * sy)
            col = self._tc(r)
            txt = (r.text.replace("&","&amp;").replace("<","&lt;")
                         .replace(">","&gt;").replace('"',"&quot;"))
            parts.append(
                f'    <text x="{x:.1f}" y="{y:.1f}" '
                f'font-family={self.FONT!r} '
                f'font-size="{fs:.1f}" fill="{col}" '
                f'inkscape:label="T:{txt[:15]}">'
                f'{txt}</text>'
            )
        return "\n".join(parts)

    def build(self, results, text=True):
        if not results:
            return ""
        ref = results[0]
        vb   = ref.viewbox
        vbp  = vb.split()
        vb_w = float(vbp[2]) if len(vbp) >= 4 else 100.0
        vb_h = float(vbp[3]) if len(vbp) >= 4 else 100.0

        groups = []
        for i, tr in enumerate(results):
            hx = self._hex(tr.color)
            paths = "\n    ".join(f'<path d="{d}"/>' for d in tr.paths)
            tf = f'transform="{tr.transform}"' if tr.transform else ""
            groups.append(
                f'  <g id="layer_{i:02d}" '
                f'inkscape:label="Color {hx}" '
                f'inkscape:groupmode="layer" '
                f'fill="{hx}" stroke="none" {tf}>\n'
                f'    {paths}\n'
                f'  </g>'
            )

        txt_elems = self._text_nodes(vb_w, vb_h) if text else ""
        txt_group = (
            f'  <g id="text_layer" inkscape:label="Text — Times New Roman" '
            f'inkscape:groupmode="layer">\n{txt_elems}\n  </g>'
            if txt_elems else ""
        )

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
            '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
            '<svg version="1.1"\n'
            '     xmlns="http://www.w3.org/2000/svg"\n'
            '     xmlns:xlink="http://www.w3.org/1999/xlink"\n'
            '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n'
            f'     viewBox="{vb}"\n'
            f'     width="{ref.width}" height="{ref.height}"\n'
            '     preserveAspectRatio="xMidYMid meet">\n'
            '  <title>Traced Vector — bitmap_to_vector v3.0</title>\n'
            + "\n".join(groups)
            + ("\n" + txt_group if txt_group else "")
            + "\n</svg>\n"
        )


# ══════════════════════════════════════════════════════════
# M6  EPSExporter
# ══════════════════════════════════════════════════════════
class EPSExporter:
    @staticmethod
    def _d2ps(d):
        toks = re.findall(
            r'[MLCZmlcz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
        ps = ["newpath"]; i = 0
        while i < len(toks):
            t = toks[i]
            if t in "Mm":
                i += 1
                if i+1 >= len(toks): break
                ps.append(f"{toks[i]} {toks[i+1]} moveto"); i += 2
            elif t in "Ll":
                i += 1
                if i+1 >= len(toks): break
                ps.append(f"{toks[i]} {toks[i+1]} lineto"); i += 2
            elif t in "Cc":
                i += 1
                if i+5 >= len(toks): break
                c = toks[i:i+6]; i += 6
                ps.append(f"{c[0]} {c[1]} {c[2]} {c[3]} {c[4]} {c[5]} curveto")
            elif t in "Zz":
                ps.append("closepath"); i += 1
            else:
                i += 1
        ps.append("fill")
        return "\n".join(ps)

    @classmethod
    def export(cls, results, ocr=None, arr=None, img_size=(100,100)):
        import datetime
        if not results:
            return ""
        ref = results[0]
        vbp = ref.viewbox.split()
        vb_w = float(vbp[2]) if len(vbp) >= 4 else float(img_size[0])
        vb_h = float(vbp[3]) if len(vbp) >= 4 else float(img_size[1])
        tm = re.findall(r'[-+]?\d+\.?\d*', ref.transform)
        if len(tm) >= 4:
            tx, ty, sx, sy = float(tm[0]), float(tm[1]), float(tm[2]), float(tm[3])
        else:
            tx, ty, sx, sy = 0.0, vb_h, 0.1, -0.1

        hdr = (
            "%!PS-Adobe-3.0 EPSF-3.0\n"
            f"%%BoundingBox: 0 0 {int(vb_w)} {int(vb_h)}\n"
            f"%%HiResBoundingBox: 0 0 {vb_w:.3f} {vb_h:.3f}\n"
            "%%Title: Traced Vector Image\n"
            "%%Creator: bitmap_to_vector.py v3.0\n"
            f"%%CreationDate: {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
            "%%DocumentData: Clean7Bit\n%%LanguageLevel: 2\n%%Pages: 1\n"
            "%%DocumentFonts: Times-Roman Times-Bold\n%%EndComments\n\n"
            "%%Page: 1 1\ngsave\n"
            f"{tx} {ty} translate\n{sx} {sy} scale\n"
        )

        body = []
        for i, tr in enumerate(results):
            r, g, b = int(tr.color[0]), int(tr.color[1]), int(tr.color[2])
            body.append(f"\n% Layer {i:02d}  rgb({r},{g},{b})")
            body.append(f"{r/255:.4f} {g/255:.4f} {b/255:.4f} setrgbcolor")
            for d in tr.paths:
                body.append(cls._d2ps(d))

        txt_ps = []
        if ocr and arr is not None:
            oh, ow = arr.shape[:2]
            txt_ps += ["\ngrestore", "gsave"]
            for r_obj in ocr:
                tc = OCREngine.sample_color(arr, r_obj)
                ex = r_obj.x * (vb_w / ow)
                ey = vb_h - (r_obj.y + r_obj.h) * (vb_h / oh)
                fs = max(6, r_obj.font_size * (vb_h / oh))
                esc = r_obj.text.replace("(","\\(").replace(")","\\)")
                txt_ps.append(
                    f"{tc[0]/255:.3f} {tc[1]/255:.3f} {tc[2]/255:.3f} setrgbcolor\n"
                    f"/Times-Roman {fs:.1f} selectfont\n"
                    f"{ex:.1f} {ey:.1f} moveto ({esc}) show"
                )

        return hdr + "\n".join(body) + "\n".join(txt_ps) + "\ngrestore\nshowpage\n%%Trailer\n%%EOF\n"


# ══════════════════════════════════════════════════════════
# M7  BatchProcessor
# ══════════════════════════════════════════════════════════
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


def collect(inputs):
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
            print(f"  ⚠ skip: {inp}")
    return sorted(set(files))


def process(path: Path, args) -> bool:
    out_dir = Path(args.output_dir) if args.output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(out_dir / path.stem)

    # M1
    print("  [M1] 加载 + 预处理...")
    img = ImageLoader.load(str(path))
    img = ImageLoader.preprocess(img, args.max_dim, not args.no_sharpen)
    arr = np.array(img)
    print(f"       {img.size[0]} × {img.size[1]} px")

    # M3 OCR
    ocr = []
    if args.ocr:
        if HAS_OCR:
            print("  [M3] OCR 文字识别...")
            ocr = OCREngine.detect(img)
            print(f"       {len(ocr)} 个文字区域识别完成")
        else:
            print("  [M3] ⚠ pytesseract 未安装，跳过 OCR")

    # M4 Tracer
    tracer = PotracTracer(args.turdsize, args.alphamax, args.opttolerance)
    if args.color:
        print(f"  [M2] 颜色量化（{args.colors} 色）...")
        layers = ColorAnalyzer.quantize(img, args.colors)
        print(f"       有效色层: {len(layers)}")
        print("  [M4] Potrace 分层描摹...")
        results = tracer.trace_layers(layers)
    else:
        print(f"  [M4] Potrace 单色描摹（阈值 {args.threshold}）...")
        tr = tracer.trace_mono(img, args.threshold)
        results = [tr] if tr else []

    if not results:
        print("  ✗ 无有效路径"); return False

    total = sum(len(r.paths) for r in results)
    print(f"       {len(results)} 层 / {total} 路径")

    # M5 SVG
    asm = SVGAssembler(ocr=ocr, arr=arr)
    fmt = args.format
    if fmt in ("svg", "both"):
        print("  [M5] 组装 SVG...")
        svg = asm.build(results, text=args.ocr)
        out = f"{stem}.svg"
        open(out, "w", encoding="utf-8").write(svg)
        print(f"  ✓ SVG  {out}  ({os.path.getsize(out)//1024} KB)")

    # M6 EPS
    if fmt in ("eps", "both"):
        print("  [M6] 导出 EPS...")
        eps = EPSExporter.export(results, ocr if args.ocr else None, arr, img.size)
        out = f"{stem}.eps"
        open(out, "w", encoding="utf-8").write(eps)
        print(f"  ✓ EPS  {out}  ({os.path.getsize(out)//1024} KB)")

    return True


def main():
    ap = argparse.ArgumentParser(
        description="bitmap_to_vector v3.0 — 高精度位图→矢量描摹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("input", nargs="+")
    ap.add_argument("--format",  choices=["svg","eps","both"], default="both")
    ap.add_argument("--color",   action="store_true", help="彩色分层描摹")
    ap.add_argument("--colors",  type=int, default=8)
    ap.add_argument("--threshold", type=int, default=128)
    ap.add_argument("--ocr",     action="store_true",
                    help="OCR文字→Times New Roman <text>节点")
    ap.add_argument("--turdsize",      type=int,   default=2)
    ap.add_argument("--alphamax",      type=float, default=1.0)
    ap.add_argument("--opttolerance",  type=float, default=0.2)
    ap.add_argument("--max-dim",       type=int,   default=2048)
    ap.add_argument("--no-sharpen",    action="store_true")
    ap.add_argument("--output-dir",    metavar="DIR")
    ap.add_argument("--verbose",       action="store_true")
    args = ap.parse_args()

    files = collect(args.input)
    if not files:
        sys.exit("未找到图像文件")

    print(f"\n{'='*58}")
    print(f"  bitmap_to_vector  v3.0")
    print(f"  {len(files)} 个文件  |  {'彩色' if args.color else '黑白'}模式  |  "
          f"格式: {args.format.upper()}")
    print(f"  OCR: {'✓' if args.ocr and HAS_OCR else '✗'}  "
          f"OpenCV: {'✓' if HAS_CV2 else '✗'}  "
          f"sklearn: {'✓' if HAS_SKLEARN else '✗'}")
    print(f"{'='*58}")

    ok = fail = 0
    for fp in files:
        print(f"\n[{ok+fail+1}/{len(files)}] {fp.name}")
        try:
            if process(fp, args): ok += 1
            else: fail += 1
        except Exception as e:
            print(f"  ✗ {e}")
            if args.verbose:
                import traceback; traceback.print_exc()
            fail += 1

    print(f"\n{'='*58}")
    print(f"  完成: {ok} 成功" + (f"  {fail} 失败" if fail else ""))
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
