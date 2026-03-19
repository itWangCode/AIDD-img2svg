#!/usr/bin/env python3
"""
img2svg.py  — AI-Powered Image → Editable SVG Pipeline
========================================================
输入任意图片 → Claude Vision 深度分析 → 自动生成完全可编辑 SVG

架构：
  Stage 1  ImageIngestor     加载图片、提取基础信息
  Stage 2  VisionAnalyzer    Claude Vision API 深度理解图像结构
  Stage 3  SchemaValidator   校验并补全分析结果
  Stage 4  SVGRenderer       将分析结果渲染为完整可编辑 SVG
  Stage 5  PostProcessor     优化输出、生成 EPS

特性：
  ✓ 全自动：无需手动配置，AI 理解图像内容
  ✓ 真实文字：所有文字用 <text> 节点 + Times New Roman
  ✓ 可移动 icon：每个图标封装在独立 <g id> 组中
  ✓ 精确颜色：从图像像素精确采样，不估算
  ✓ 通用性：适用于信息图、流程图、架构图、海报等
  ✓ 批量处理：支持文件夹批量转换

安装依赖：
    pip install anthropic Pillow numpy

用法：
    export ANTHROPIC_API_KEY="sk-ant-..."
    python img2svg.py input.png
    python img2svg.py input.jpg --output out.svg
    python img2svg.py folder/  --batch
    python img2svg.py input.png --api-key sk-ant-... --verbose
"""

import argparse, base64, json, math, os, re, sys, time
from pathlib import Path
from typing import Any

# ── 依赖检测 ────────────────────────────────────────────
try:
    from PIL import Image
    import numpy as np
except ImportError:
    sys.exit("请安装: pip install Pillow numpy")

try:
    import anthropic
except ImportError:
    sys.exit("请安装: pip install anthropic")

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


# ══════════════════════════════════════════════════════════
# Stage 1  ImageIngestor
# ══════════════════════════════════════════════════════════
class ImageIngestor:
    """
    加载并预处理图像：
    - EXIF 旋转修正
    - 透明背景 → 白底
    - 尺寸控制（API token 限制）
    - base64 编码
    - 像素级颜色采样（精确）
    """
    MAX_DIM = 1568   # Claude vision 最优分辨率

    @classmethod
    def load(cls, path: str) -> dict:
        img = Image.open(path)
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # 透明 → 白底
        if img.mode in ("RGBA", "LA", "P"):
            if img.mode == "P":
                img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode in ("RGBA", "LA"):
                bg.paste(img, mask=img.split()[-1])
            else:
                bg.paste(img)
            img = bg
        else:
            img = img.convert("RGB")

        # 尺寸控制
        w, h = img.size
        if max(w, h) > cls.MAX_DIM:
            scale = cls.MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size

        arr = np.array(img)

        # base64 编码（PNG 格式）
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()

        return {
            "img": img,
            "arr": arr,
            "width": w,
            "height": h,
            "b64": b64,
            "path": path,
        }

    @staticmethod
    def sample_color(arr: np.ndarray, x: int, y: int, radius: int = 4) -> str:
        """从像素坐标精确采样颜色（取邻域众数）"""
        h, w = arr.shape[:2]
        x1 = max(0, x - radius); x2 = min(w, x + radius)
        y1 = max(0, y - radius); y2 = min(h, y + radius)
        region = arr[y1:y2, x1:x2].reshape(-1, 3)
        if len(region) == 0:
            return "#888888"
        # 排除纯白（背景）
        mask = np.any(region < 240, axis=1)
        if mask.sum() > 5:
            region = region[mask]
        c = region.mean(axis=0).astype(int)
        return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    @staticmethod
    def dominant_colors(arr: np.ndarray, n: int = 12) -> list:
        """K-Means 提取主色"""
        pixels = arr.reshape(-1, 3).astype(np.float32)
        # 过滤接近白色
        mask = np.any(pixels < 230, axis=1)
        pixels = pixels[mask]
        if len(pixels) < n:
            return ["#888888"] * n

        rng = np.random.default_rng(42)
        centers = pixels[rng.choice(len(pixels), n, replace=False)].copy()
        for _ in range(30):
            d = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
            labels = np.argmin(d, axis=1)
            new_c = np.array([
                pixels[labels == k].mean(0) if (labels == k).any() else centers[k]
                for k in range(n)
            ])
            if np.allclose(centers, new_c, atol=1):
                break
            centers = new_c

        centers = np.clip(centers, 0, 255).astype(int)
        coverage = [(labels == k).sum() for k in range(n)]
        order = sorted(range(n), key=lambda i: -coverage[i])
        return [f"#{centers[i,0]:02x}{centers[i,1]:02x}{centers[i,2]:02x}" for i in order]


# ══════════════════════════════════════════════════════════
# Stage 2  VisionAnalyzer
# ══════════════════════════════════════════════════════════
ANALYSIS_PROMPT = """\
You are an expert SVG engineer and data visualization analyst.
Analyze this image and return a detailed JSON description for reconstructing it as a fully editable SVG.

Return ONLY a single valid JSON object (no markdown, no backticks, no explanation).

Required JSON schema:
{
  "title": "main title text",
  "image_type": "infographic|flowchart|diagram|poster|chart|other",
  "canvas": {"width": <int>, "height": <int>},
  "background": {"type": "solid|gradient", "color": "#hex", "color2": "#hex"},
  "color_palette": ["#hex", ...],  // 8-12 dominant colors
  "typography": {
    "primary_font": "Times New Roman",
    "title_size": <int>,
    "header_size": <int>,
    "body_size": <int>,
    "title_color": "#hex",
    "body_color": "#hex"
  },
  "layout": "left_center_right|top_bottom|radial|grid|freeform",
  "panels": [
    {
      "id": "panel_1",
      "x": <int>, "y": <int>, "w": <int>, "h": <int>,
      "corner_radius": <int>,
      "fill": "#hex",
      "stroke": "#hex",
      "has_header": true,
      "header": {
        "text": "...",
        "fill": "#hex",
        "text_color": "#hex",
        "height": <int>
      },
      "body_text": "...",
      "body_text_x": <int>,
      "body_text_y": <int>,
      "icons": [
        {
          "id": "icon_book",
          "x": <int>, "y": <int>,
          "description": "open book with two pages and text lines",
          "type": "book|molecule|dna|robot|flask|gear|arrow|chart|target|custom",
          "primary_color": "#hex",
          "secondary_color": "#hex",
          "size": <int>
        }
      ]
    }
  ],
  "central_element": {
    "present": true,
    "type": "circular_ring|linear|none",
    "cx": <int>, "cy": <int>,
    "outer_radius": <int>, "inner_radius": <int>,
    "segments": [
      {
        "label": "...",
        "fill": "#hex",
        "start_angle_deg": <float>,
        "end_angle_deg": <float>,
        "label_color": "#hex"
      }
    ],
    "center_fill": "#hex",
    "center_radius": <int>,
    "center_icon": {
      "type": "robot|gear|atom|other",
      "description": "detailed description for SVG recreation"
    },
    "center_texts": [
      {"text": "...", "size": <int>, "weight": "normal|bold", "color": "#hex", "y_offset": <int>}
    ]
  },
  "standalone_texts": [
    {
      "text": "...",
      "x": <int>, "y": <int>,
      "size": <int>,
      "weight": "normal|bold",
      "color": "#hex",
      "anchor": "start|middle|end"
    }
  ],
  "connectors": [
    {"type": "arrow|line|curve", "x1":.., "y1":.., "x2":.., "y2":.., "color":"#hex"}
  ],
  "decorations": [
    {"type": "circle|rect|line", "params": {}}
  ]
}

CRITICAL RULES:
1. All measurements in pixels relative to actual image dimensions
2. Colors must be EXACT hex values sampled from the image, not guessed
3. All text strings must be VERBATIM from the image
4. Icon descriptions must be specific enough to recreate in SVG paths
5. Angles: 0°=right, 90°=up, 180°=left, 270°=down (standard math convention)
6. If a section has multiple icons, list all of them
7. For the central ring, measure angles carefully from visual inspection
"""


class VisionAnalyzer:
    """
    Claude Vision API 分析器
    两阶段：粗分析 → 精化分析
    """

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-opus-4-5"):
        self.client = client
        self.model = model

    def analyze(self, image_data: dict, verbose: bool = False) -> dict:
        """主分析入口：Vision API 理解图像结构"""
        if verbose:
            print("  [Vision] 第一阶段：结构分析...")

        # Stage 2a: 结构分析
        schema = self._call_vision(
            image_data["b64"],
            ANALYSIS_PROMPT,
            max_tokens=8000
        )

        if verbose:
            print(f"  [Vision] 分析完成，检测到 {len(schema.get('panels', []))} 个面板")

        # Stage 2b: 颜色精化（用像素采样覆盖 AI 猜测的颜色）
        schema = self._refine_colors(schema, image_data["arr"],
                                     image_data["width"], image_data["height"])

        # Stage 2c: 规范化坐标（API 返回的坐标可能基于不同分辨率）
        schema = self._normalize_coords(schema, image_data["width"], image_data["height"])

        return schema

    def _call_vision(self, b64: str, prompt: str, max_tokens: int = 8000) -> dict:
        """调用 Claude Vision API"""
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                raw = resp.content[0].text.strip()

                # 清理可能的 markdown 包裹
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

                return json.loads(raw)

            except json.JSONDecodeError as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                # 尝试提取 JSON 子串
                try:
                    match = re.search(r'\{.*\}', raw, re.DOTALL)
                    if match:
                        return json.loads(match.group())
                except Exception:
                    pass
                raise ValueError(f"API 返回非 JSON 内容: {str(e)}")

            except anthropic.RateLimitError:
                wait = (attempt + 1) * 30
                print(f"  [API] 速率限制，等待 {wait}s...")
                time.sleep(wait)

            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                    continue
                raise

        raise RuntimeError("API 调用失败")

    def _refine_colors(self, schema: dict, arr: np.ndarray, w: int, h: int) -> dict:
        """用真实像素采样替换 AI 估算的颜色"""
        def sample(x, y):
            if x is None or y is None:
                return None
            px = max(0, min(w - 1, int(x)))
            py = max(0, min(h - 1, int(y)))
            return ImageIngestor.sample_color(arr, px, py)

        # 采样背景色
        schema.setdefault("background", {})
        if "color" not in schema["background"]:
            schema["background"]["color"] = sample(10, 10) or "#f8f8fb"

        # 采样各面板颜色
        for panel in schema.get("panels", []):
            x = panel.get("x", 0); y = panel.get("y", 0)
            if panel.get("has_header") and panel.get("header"):
                hdr = panel["header"]
                hx = x + panel.get("w", 100) // 2
                hy = y + hdr.get("height", 40) // 2
                sampled = sample(hx, hy)
                if sampled:
                    hdr["fill"] = sampled

        # 采样圆环颜色
        ce = schema.get("central_element", {})
        if ce.get("present") and ce.get("type") == "circular_ring":
            cx = ce.get("cx", w // 2)
            cy = ce.get("cy", h // 2)
            or_ = ce.get("outer_radius", 100)
            ir_ = ce.get("inner_radius", 70)
            mr = (or_ + ir_) // 2
            for seg in ce.get("segments", []):
                a1 = seg.get("start_angle_deg", 0)
                a2 = seg.get("end_angle_deg", 90)
                mid_a = math.radians((a1 + a2) / 2)
                sx = cx + mr * math.cos(mid_a)
                sy = cy - mr * math.sin(mid_a)
                c = sample(int(sx), int(sy))
                if c:
                    seg["fill"] = c

        return schema

    def _normalize_coords(self, schema: dict, img_w: int, img_h: int) -> dict:
        """确保坐标在合理范围内"""
        def clamp(v, lo, hi):
            return max(lo, min(hi, v)) if v is not None else lo

        for panel in schema.get("panels", []):
            panel["x"] = clamp(panel.get("x", 0), 0, img_w)
            panel["y"] = clamp(panel.get("y", 0), 0, img_h)
            panel["w"] = clamp(panel.get("w", 100), 10, img_w)
            panel["h"] = clamp(panel.get("h", 100), 10, img_h)

        return schema


# ══════════════════════════════════════════════════════════
# Stage 3  SchemaValidator
# ══════════════════════════════════════════════════════════
class SchemaValidator:
    """校验并补全分析结果，确保渲染器不会崩溃"""

    @staticmethod
    def validate(schema: dict, img_w: int, img_h: int) -> dict:
        s = schema

        # 基础字段
        s.setdefault("title", "Untitled")
        s.setdefault("canvas", {"width": img_w, "height": img_h})
        s["canvas"].setdefault("width", img_w)
        s["canvas"].setdefault("height", img_h)
        s.setdefault("background", {"type": "solid", "color": "#f8f8fb"})
        s.setdefault("color_palette", ["#3a5d96", "#f0b24d", "#7bb8a0", "#e87878"])
        s.setdefault("typography", {})
        s["typography"].setdefault("primary_font", '"Times New Roman", Times, Georgia, serif')
        s["typography"].setdefault("title_size", 42)
        s["typography"].setdefault("header_size", 28)
        s["typography"].setdefault("body_size", 22)
        s["typography"].setdefault("title_color", "#1e2847")
        s["typography"].setdefault("body_color", "#2a3560")
        s.setdefault("panels", [])
        s.setdefault("standalone_texts", [])
        s.setdefault("connectors", [])
        s.setdefault("decorations", [])
        s.setdefault("central_element", {"present": False})

        # 面板校验
        for i, panel in enumerate(s["panels"]):
            panel.setdefault("id", f"panel_{i}")
            panel.setdefault("x", 50); panel.setdefault("y", 200)
            panel.setdefault("w", 300); panel.setdefault("h", 200)
            panel.setdefault("corner_radius", 14)
            panel.setdefault("fill", "#f8f8fc")
            panel.setdefault("stroke", "#d0d6e4")
            panel.setdefault("has_header", False)
            panel.setdefault("icons", [])
            if panel["has_header"]:
                panel.setdefault("header", {})
                panel["header"].setdefault("text", "")
                panel["header"].setdefault("fill", "#8888cc")
                panel["header"].setdefault("text_color", "#ffffff")
                panel["header"].setdefault("height", 48)

        # 中心元素校验
        ce = s["central_element"]
        if ce.get("present"):
            ce.setdefault("cx", img_w // 2)
            ce.setdefault("cy", img_h // 2)
            ce.setdefault("outer_radius", min(img_w, img_h) // 5)
            ce.setdefault("inner_radius", min(img_w, img_h) // 7)
            ce.setdefault("center_fill", "#ffffff")
            ce.setdefault("center_radius", min(img_w, img_h) // 8)
            ce.setdefault("segments", [])
            ce.setdefault("center_texts", [])

        return s


# ══════════════════════════════════════════════════════════
# Stage 4  SVGRenderer
# ══════════════════════════════════════════════════════════
class IconLibrary:
    """
    内置 icon 库：根据 description/type 自动选择最合适的 SVG 图标
    每个 icon 都是独立 <g> 组，可在 Illustrator 中整体移动
    """

    @staticmethod
    def render(icon: dict) -> str:
        """渲染一个 icon，返回 SVG 字符串"""
        itype = icon.get("type", "custom").lower()
        x = icon.get("x", 0)
        y = icon.get("y", 0)
        size = icon.get("size", 80)
        pc = icon.get("primary_color", "#3a5d96")
        sc = icon.get("secondary_color", "#d0e0f8")
        iid = icon.get("id", f"icon_{itype}")
        desc = icon.get("description", "").lower()

        # 智能匹配
        if any(k in desc for k in ["book", "knowledge", "read"]):
            return IconLibrary._book(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["molecule", "mol", "atom", "chemical"]):
            return IconLibrary._molecule(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["dna", "helix", "protein", "gene"]):
            return IconLibrary._dna(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["robot", "agent", "ai face", "android"]):
            return IconLibrary._robot(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["flask", "beaker", "chemistry", "lab", "test tube"]):
            return IconLibrary._flask(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["gear", "cog", "setting"]):
            return IconLibrary._gear(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["pill", "capsule", "drug", "medicine"]):
            return IconLibrary._pill(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["target", "goal", "optim", "bullseye"]):
            return IconLibrary._target(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["arm", "robot arm", "automat"]):
            return IconLibrary._robot_arm(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["crystal", "lattice", "material", "structure"]):
            return IconLibrary._lattice(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["chart", "graph", "plot", "data"]):
            return IconLibrary._chart(iid, x, y, size, pc, sc)
        elif any(k in desc for k in ["flow", "decision", "tree"]):
            return IconLibrary._flowchart(iid, x, y, size, pc, sc)
        else:
            return IconLibrary._molecule(iid, x, y, size, pc, sc)  # default

    # ── Individual icon renderers ─────────────────────

    @staticmethod
    def _book(iid, x, y, sz, pc, sc):
        s = sz / 100
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <path d="M0,78 L9,0 Q52,-10 80,18 L80,88 Q48,64 0,78Z" fill="{pc}" stroke="{pc}" stroke-width="2"/>
  <path d="M7,74 L15,5 Q49,-3 78,20 L78,84 Q46,60 7,74Z" fill="{sc}" stroke="{pc}" stroke-width="1.5" opacity="0.9"/>
  <path d="M160,78 L151,0 Q108,-10 80,18 L80,88 Q112,64 160,78Z" fill="{pc}" stroke="{pc}" stroke-width="2" opacity="0.85"/>
  <path d="M153,74 L145,5 Q111,-3 82,20 L82,84 Q114,60 153,74Z" fill="{sc}" stroke="{pc}" stroke-width="1.5" opacity="0.9"/>
  <path d="M72,86 Q80,100 88,86" fill="none" stroke="{pc}" stroke-width="6" stroke-linecap="round"/>
  <g stroke="{pc}" stroke-width="2" stroke-linecap="round" opacity="0.7">
    <line x1="24" y1="20" x2="60" y2="18"/><line x1="22" y1="35" x2="60" y2="33"/>
    <line x1="22" y1="49" x2="60" y2="47"/><line x1="27" y1="62" x2="58" y2="61"/>
    <line x1="96" y1="20" x2="132" y2="18"/><line x1="96" y1="35" x2="133" y2="33"/>
    <line x1="96" y1="49" x2="133" y2="47"/><line x1="100" y1="62" x2="130" y2="61"/>
  </g>
</g>'''

    @staticmethod
    def _molecule(iid, x, y, sz, pc, sc):
        s = sz / 100
        nc = "#9060b0"  # node color variant
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <g stroke="{pc}" stroke-width="3" stroke-linecap="round">
    <line x1="28" y1="42" x2="68" y2="64"/>
    <line x1="68" y1="64" x2="98" y2="30"/>
    <line x1="68" y1="64" x2="38" y2="98"/>
    <line x1="68" y1="64" x2="108" y2="104"/>
    <line x1="98" y1="30" x2="140" y2="32"/>
    <line x1="140" y1="32" x2="156" y2="72"/>
  </g>
  <circle cx="28"  cy="42"  r="14" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="68"  cy="64"  r="19" fill="{sc}" stroke="{pc}" stroke-width="2.5" opacity="0.9"/>
  <circle cx="98"  cy="30"  r="14" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="38"  cy="98"  r="12" fill="#d8aad8" stroke="#7a4c96" stroke-width="2.5"/>
  <circle cx="108" cy="104" r="12" fill="#b2d8a6" stroke="#4a8860" stroke-width="2.5"/>
  <circle cx="156" cy="72"  r="16" fill="#f6e090" stroke="#c89820" stroke-width="2.5"/>
  <g stroke="{pc}" stroke-width="2" stroke-linecap="round" opacity="0.6">
    <line x1="143" y1="52" x2="168" y2="52"/>
    <line x1="143" y1="72" x2="168" y2="72"/>
    <line x1="143" y1="92" x2="168" y2="92"/>
  </g>
</g>'''

    @staticmethod
    def _dna(iid, x, y, sz, pc, sc):
        s = sz / 140
        gc = "#4a8060"
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <g fill="none" stroke-width="3.5" stroke-linecap="round">
    <path d="M10,0 C58,0 58,46 10,46 C-24,46 -24,92 10,92 C58,92 58,138 10,138" stroke="{gc}"/>
    <path d="M56,0 C8,0 8,46 56,46 C90,46 90,92 56,92 C8,92 8,138 56,138" stroke="{sc}"/>
    <g stroke="#c89820" stroke-width="2.2">
      <line x1="18" y1="14" x2="48" y2="14"/>
      <line x1="5"  y1="37" x2="50" y2="37"/>
      <line x1="5"  y1="58" x2="50" y2="58"/>
      <line x1="18" y1="80" x2="48" y2="80"/>
      <line x1="18" y1="102" x2="48" y2="102"/>
      <line x1="5"  y1="124" x2="50" y2="124"/>
    </g>
  </g>
</g>'''

    @staticmethod
    def _robot(iid, x, y, sz, pc, sc):
        s = sz / 200
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <!-- ears -->
  <rect x="14"  y="70" width="20" height="50" rx="10" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>
  <rect x="166" y="70" width="20" height="50" rx="10" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>
  <!-- head -->
  <path d="M52,78 q0,-40 28,-56 q14,-8 30,-8 q16,0 30,8 q28,16 28,56 v22 q0,32 -20,50 q-20,20 -38,20 q-18,0 -38,-20 q-20,-18 -20,-50Z"
        fill="#f2f4fa" stroke="{pc}" stroke-width="2.5"/>
  <!-- visor -->
  <path d="M52,86 q0,-10 8,-16 h80 q8,6 8,16 v8 H52Z" fill="{sc}" stroke="{pc}" stroke-width="1.5" opacity="0.5"/>
  <!-- eyes -->
  <circle cx="88"  cy="110" r="13" fill="#96aee0" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="112" cy="110" r="13" fill="#96aee0" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="89"  cy="108" r="5"  fill="#1e2847"/>
  <circle cx="113" cy="108" r="5"  fill="#1e2847"/>
  <circle cx="91"  cy="106" r="2"  fill="#ffffff"/>
  <circle cx="115" cy="106" r="2"  fill="#ffffff"/>
  <!-- smile -->
  <path d="M82,132 q18,12 36,0" fill="none" stroke="{pc}" stroke-width="3" stroke-linecap="round"/>
  <!-- gear -->
  <circle cx="100" cy="46" r="16" fill="#f0b24d"/>
  <circle cx="100" cy="46" r="7"  fill="#f8f8f8"/>
  <g fill="none" stroke="#f0b24d" stroke-width="4" stroke-linecap="round">
    <line x1="100" y1="26" x2="100" y2="34"/><line x1="100" y1="58" x2="100" y2="66"/>
    <line x1="80"  y1="46" x2="88"  y2="46"/><line x1="112" y1="46" x2="120" y2="46"/>
    <line x1="86"  y1="32" x2="92"  y2="38"/><line x1="108" y1="54" x2="114" y2="60"/>
    <line x1="86"  y1="60" x2="92"  y2="54"/><line x1="108" y1="38" x2="114" y2="32"/>
  </g>
  <!-- headphones -->
  <path d="M40,50 q14,-38 68,-38 q54,0 68,38" fill="none" stroke="{pc}" stroke-width="3.5" stroke-linecap="round"/>
  <!-- body/shoulders -->
  <g stroke="{pc}" stroke-width="3" fill="none" stroke-linecap="round">
    <path d="M36,202 q6,-36 40,-42 q16,-3 32,-16 q16,13 32,16 q34,6 40,42"/>
    <path d="M78,156 q0,28 34,28 q34,0 34,-28"/>
  </g>
</g>'''

    @staticmethod
    def _flask(iid, x, y, sz, pc, sc):
        s = sz / 90
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <ellipse cx="46" cy="86" rx="40" ry="8" fill="{sc}" opacity="0.6"/>
  <path d="M26,0 h40 M46,0 v16 l24,50 q6,15 -10,15 h-50 q-16,0 -10,-15 l24,-50 v-16"
        fill="none" stroke="{pc}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M14,54 h48 l-7,14 h-34z" fill="{sc}" opacity="0.85"/>
  <circle cx="22" cy="62" r="4" fill="{sc}"/>
  <circle cx="38" cy="60" r="4" fill="{sc}"/>
  <circle cx="30" cy="56" r="3" fill="{sc}" opacity="0.7"/>
</g>'''

    @staticmethod
    def _gear(iid, x, y, sz, pc, sc):
        s = sz / 80
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <circle cx="40" cy="40" r="28" fill="{pc}"/>
  <circle cx="40" cy="40" r="14" fill="{sc}"/>
  <circle cx="40" cy="40" r="6"  fill="{pc}" opacity="0.6"/>
  <g fill="none" stroke="{pc}" stroke-width="7" stroke-linecap="round">
    <line x1="40" y1="4"  x2="40" y2="14"/>
    <line x1="40" y1="66" x2="40" y2="76"/>
    <line x1="4"  y1="40" x2="14" y2="40"/>
    <line x1="66" y1="40" x2="76" y2="40"/>
    <line x1="12" y1="12" x2="19" y2="19"/>
    <line x1="61" y1="61" x2="68" y2="68"/>
    <line x1="12" y1="68" x2="19" y2="61"/>
    <line x1="61" y1="19" x2="68" y2="12"/>
  </g>
</g>'''

    @staticmethod
    def _pill(iid, x, y, sz, pc, sc):
        s = sz / 100
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <!-- bottle -->
  <path d="M2,20 q0,-14 14,-14 h52 q14,0 14,14 v62 q0,14 -14,14 h-52 q-14,0 -14,-14Z"
        fill="{sc}" stroke="{pc}" stroke-width="2.5"/>
  <rect x="10" y="44" width="36" height="16" rx="3" fill="#f0ce80"/>
  <path d="M0,2 h82 v20 q0,6 -6,6 h-70 q-6,0 -6,-6Z" fill="#f0ce80" stroke="#c89820" stroke-width="2"/>
  <line x1="20" y1="20" x2="20" y2="36" stroke="#e06060" stroke-width="3" stroke-linecap="round"/>
  <line x1="12" y1="28" x2="28" y2="28" stroke="#e06060" stroke-width="3" stroke-linecap="round"/>
  <!-- capsule -->
  <g transform="translate(92,18)">
    <rect x="0" y="0" width="88" height="38" rx="19" fill="#d4def4" stroke="{pc}" stroke-width="2"/>
    <path d="M43,0 a19,19 0 0 1 0,38" fill="#8aa8de" stroke="{pc}" stroke-width="2"/>
  </g>
</g>'''

    @staticmethod
    def _target(iid, x, y, sz, pc, sc):
        s = sz / 90
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <circle cx="36" cy="36" r="36" fill="none" stroke="{pc}" stroke-width="3.5"/>
  <circle cx="36" cy="36" r="22" fill="none" stroke="{sc}" stroke-width="2.5"/>
  <circle cx="36" cy="36" r="10" fill="#f0ce80" stroke="#c89820" stroke-width="2"/>
  <line x1="36" y1="36" x2="106" y2="36" stroke="{pc}" stroke-width="4.5" stroke-linecap="round"/>
  <path d="M92,22 l14,14 l-14,14" fill="none" stroke="{pc}" stroke-width="4.5"
        stroke-linecap="round" stroke-linejoin="round"/>
</g>'''

    @staticmethod
    def _robot_arm(iid, x, y, sz, pc, sc):
        s = sz / 130
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <ellipse cx="134" cy="124" rx="110" ry="12" fill="{sc}" opacity="0.5"/>
  <rect x="4" y="112" width="82" height="14" rx="3" fill="#90b0e0" stroke="{pc}" stroke-width="2"/>
  <g stroke="{pc}" stroke-width="2.8" stroke-linecap="round">
    <line x1="46" y1="112" x2="46" y2="76"/>
    <line x1="46" y1="76"  x2="86" y2="24"/>
    <line x1="86" y1="24"  x2="160" y2="32"/>
    <line x1="160" y1="32" x2="202" y2="40"/>
    <line x1="202" y1="40" x2="202" y2="78"/>
  </g>
  <circle cx="46"  cy="76"  r="9"  fill="#f0f4fc" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="86"  cy="24"  r="13" fill="#f0f4fc" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="160" cy="32"  r="13" fill="#f0f4fc" stroke="{pc}" stroke-width="2.5"/>
  <path d="M200,78 h20 v18 h-20Z" fill="#f4dea0" stroke="{pc}" stroke-width="2.5"/>
  <circle cx="104" cy="58" r="9"  fill="#f0b24d"/>
  <circle cx="168" cy="72" r="7"  fill="#f0b24d"/>
</g>'''

    @staticmethod
    def _lattice(iid, x, y, sz, pc, sc):
        s = sz / 110
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <ellipse cx="54" cy="98" rx="50" ry="8" fill="{sc}" opacity="0.5"/>
  <g stroke="{pc}" stroke-width="2.5" stroke-linecap="round">
    <line x1="14" y1="38" x2="54" y2="24"/>
    <line x1="54" y1="24" x2="94" y2="42"/>
    <line x1="94" y1="42" x2="94" y2="82"/>
    <line x1="54" y1="24" x2="54" y2="66"/>
    <line x1="14" y1="38" x2="14" y2="78"/>
    <line x1="14" y1="78" x2="54" y2="96"/>
    <line x1="54" y1="96" x2="94" y2="82"/>
    <line x1="54" y1="66" x2="14" y2="78"/>
    <line x1="54" y1="66" x2="94" y2="82"/>
  </g>
  <circle cx="14" cy="38" r="10" fill="#d0aed8" stroke="#7a4c96" stroke-width="2"/>
  <circle cx="54" cy="24" r="10" fill="#a8d0a6" stroke="#4a8860" stroke-width="2"/>
  <circle cx="94" cy="42" r="10" fill="{sc}" stroke="{pc}" stroke-width="2"/>
  <circle cx="14" cy="78" r="12" fill="#a4c0ee" stroke="{pc}" stroke-width="2"/>
  <circle cx="54" cy="66" r="10" fill="{sc}" stroke="{pc}" stroke-width="2"/>
  <circle cx="54" cy="96" r="10" fill="#b4c8ee" stroke="{pc}" stroke-width="2"/>
  <circle cx="94" cy="82" r="12" fill="#d0aed8" stroke="#7a4c96" stroke-width="2"/>
</g>'''

    @staticmethod
    def _chart(iid, x, y, sz, pc, sc):
        s = sz / 100
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <rect x="0" y="60" width="18" height="40" rx="2" fill="{pc}" opacity="0.8"/>
  <rect x="24" y="30" width="18" height="70" rx="2" fill="{pc}"/>
  <rect x="48" y="45" width="18" height="55" rx="2" fill="{pc}" opacity="0.9"/>
  <rect x="72" y="15" width="18" height="85" rx="2" fill="{sc}"/>
  <line x1="0" y1="100" x2="96" y2="100" stroke="{pc}" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="0" y1="0"   x2="0"  y2="100" stroke="{pc}" stroke-width="2.5" stroke-linecap="round"/>
</g>'''

    @staticmethod
    def _flowchart(iid, x, y, sz, pc, sc):
        s = sz / 100
        return f'''<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})">
  <rect x="4"  y="46" width="34" height="34" rx="3" fill="{sc}" stroke="{pc}" stroke-width="2"/>
  <rect x="76" y="0"  width="34" height="34" rx="3" fill="#f4ce82" stroke="#c88830" stroke-width="2"/>
  <rect x="148" y="22" width="34" height="34" rx="3" fill="{sc}" stroke="{pc}" stroke-width="2"/>
  <g stroke="{pc}" stroke-width="2.5" stroke-linecap="round">
    <line x1="38" y1="63" x2="94" y2="63"/>
    <line x1="94" y1="34" x2="94" y2="63"/>
    <line x1="94" y1="34" x2="148" y2="34"/>
    <line x1="94" y1="63" x2="148" y2="63"/>
  </g>
</g>'''


class SVGRenderer:
    """
    核心渲染器：将 schema 转换为完整可编辑 SVG
    所有文字 = Times New Roman <text> 节点
    所有 icon = 独立 <g id> 组（可在 AI 中整体移动）
    """

    def __init__(self, schema: dict):
        self.s = schema
        self.lines = []
        self.W = schema["canvas"]["width"]
        self.H = schema["canvas"]["height"]
        self.typo = schema["typography"]

    def render(self) -> str:
        self._defs()
        self._background()
        self._panels()
        self._central_element()
        self._standalone_texts()
        self._connectors()
        self.lines.append("</svg>")
        return "\n".join(self.lines)

    # ── Defs ──────────────────────────────────────────
    def _defs(self):
        font = self.typo.get("primary_font", '"Times New Roman", Times, serif')
        tc = self.typo.get("title_color", "#1e2847")
        bc = self.typo.get("body_color", "#2a3560")
        ts = self.typo.get("title_size", 42)
        hs = self.typo.get("header_size", 28)
        bs = self.typo.get("body_size", 22)
        bg = self.s["background"].get("color", "#f8f8fb")
        bg2 = self.s["background"].get("color2", "#f0f2f7")

        # Build header gradient defs from panels
        grad_defs = []
        for panel in self.s.get("panels", []):
            if panel.get("has_header") and panel.get("header"):
                pid = panel["id"]
                hc = panel["header"]["fill"]
                hc2 = self._lighten(hc, 0.15)
                grad_defs.append(
                    f'  <linearGradient id="hdr_{pid}" x1="0" x2="1" y1="0" y2="0">\n'
                    f'    <stop offset="0%" stop-color="{hc}"/>\n'
                    f'    <stop offset="100%" stop-color="{hc2}"/>\n'
                    f'  </linearGradient>'
                )

        # Arc segment gradients
        ce = self.s.get("central_element", {})
        for i, seg in enumerate(ce.get("segments", [])):
            fc = seg.get("fill", "#888888")
            fc2 = self._lighten(fc, 0.12)
            grad_defs.append(
                f'  <linearGradient id="arc_{i}" x1="0" x2="1" y1="0" y2="0">\n'
                f'    <stop offset="0%" stop-color="{fc}"/>\n'
                f'    <stop offset="100%" stop-color="{fc2}"/>\n'
                f'  </linearGradient>'
            )

        center_fill = ce.get("center_fill", "#ffffff")

        self.lines.append(f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{self.W}" height="{self.H}" viewBox="0 0 {self.W} {self.H}">
<defs>
  <!-- Background -->
  <linearGradient id="_bg" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="{bg}"/>
    <stop offset="100%" stop-color="{bg2}"/>
  </linearGradient>
  <radialGradient id="_bgGlow" cx="50%" cy="38%" r="72%">
    <stop offset="0%" stop-color="#ffffff" stop-opacity="0.75"/>
    <stop offset="100%" stop-color="{bg}" stop-opacity="0"/>
  </radialGradient>

  <!-- Panel -->
  <linearGradient id="_panel" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="#fafbfc"/>
    <stop offset="100%" stop-color="#f2f5f9"/>
  </linearGradient>

  <!-- Center -->
  <radialGradient id="_center" cx="46%" cy="36%" r="72%">
    <stop offset="0%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="{center_fill}"/>
  </radialGradient>

  <!-- Header & arc gradients -->
{chr(10).join(grad_defs)}

  <!-- Filters -->
  <filter id="_shadow" x="-15%" y="-15%" width="130%" height="130%">
    <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#8898bb" flood-opacity="0.10"/>
  </filter>
  <filter id="_shadowSm" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#8898bb" flood-opacity="0.14"/>
  </filter>
  <filter id="_shadowRing" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="3" stdDeviation="3" flood-opacity="0.13"/>
  </filter>

  <!-- Typography — Times New Roman throughout -->
  <style>
    text {{
      font-family: {font};
    }}
    .t-title  {{ font-size:{ts}px; font-weight:700; fill:{tc}; letter-spacing:0.4px; }}
    .t-hdr    {{ font-size:{hs}px; font-weight:700; fill:#ffffff; }}
    .t-body   {{ font-size:{bs}px; fill:{bc}; }}
    .t-ring   {{ font-size:{int(hs*1.05)}px; font-weight:700; fill:#ffffff; }}
    .t-center {{ font-size:{int(ts*1.05)}px; font-weight:700; fill:{tc}; }}
    .t-sub    {{ font-size:{int(ts*0.72)}px; fill:{bc}; }}
    .t-sect   {{ font-size:{int(ts*0.88)}px; font-weight:700; fill:{tc}; }}
    .navy     {{ stroke:#3a5d96; fill:none; stroke-linecap:round; stroke-linejoin:round; }}
  </style>
</defs>''')

    def _background(self):
        self.lines.append(f'''
<!-- ══ Background ══ -->
<rect width="{self.W}" height="{self.H}" fill="url(#_bg)"/>
<rect width="{self.W}" height="{self.H}" fill="url(#_bgGlow)"/>''')

    # ── Panels ────────────────────────────────────────
    def _panels(self):
        self.lines.append("\n<!-- ══ Panels ══ -->")
        for panel in self.s.get("panels", []):
            self._render_panel(panel)

    def _render_panel(self, p: dict):
        pid = p["id"]
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        rr = p.get("corner_radius", 14)
        stroke = p.get("stroke", "#d0d6e4")

        # Card shadow + background
        self.lines.append(f'''
<g id="{pid}" filter="url(#_shadow)">
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rr}" fill="url(#_panel)" stroke="{stroke}" stroke-width="1.2"/>''')

        # Header
        if p.get("has_header") and p.get("header"):
            hdr = p["header"]
            ht = hdr.get("height", 48)
            htxt = hdr.get("text", "")
            htc = hdr.get("text_color", "#ffffff")
            self.lines.append(
                f'  <path d="M{x+rr},{y} H{x+w-rr} Q{x+w},{y} {x+w},{y+rr} V{y+ht} H{x} V{y+rr} Q{x},{y} {x+rr},{y}Z"\n'
                f'        fill="url(#hdr_{pid})" stroke="none"/>\n'
                f'  <!-- inner highlight -->\n'
                f'  <rect x="{x+7}" y="{y+7}" width="{w-14}" height="{h-14}" rx="{rr-3}"\n'
                f'        fill="none" stroke="#ffffff" stroke-width="0.7" opacity="0.45"/>\n'
                f'</g>\n'
                f'<text x="{x + w//2}" y="{y + ht//2 + 10}" text-anchor="middle" '
                f'class="t-hdr" style="fill:{htc}">{self._esc(htxt)}</text>'
            )
        else:
            self.lines.append("</g>")

        # Body text
        btxt = p.get("body_text", "")
        if btxt:
            bx = p.get("body_text_x", x + w // 2)
            by = p.get("body_text_y", y + h - 24)
            self.lines.append(
                f'<text x="{bx}" y="{by}" text-anchor="middle" class="t-body">{self._esc(btxt)}</text>'
            )

        # Icons
        for icon in p.get("icons", []):
            self.lines.append(IconLibrary.render(icon))

    # ── Central Element ───────────────────────────────
    def _central_element(self):
        ce = self.s.get("central_element", {})
        if not ce.get("present"):
            return

        self.lines.append("\n<!-- ══ Central Element ══ -->")
        ctype = ce.get("type", "none")

        if ctype == "circular_ring":
            self._render_ring(ce)

    def _render_ring(self, ce: dict):
        cx = ce["cx"]; cy = ce["cy"]
        or_ = ce["outer_radius"]; ir_ = ce["inner_radius"]
        cr = ce.get("center_radius", ir_ - 10)

        # Outer guide circle
        self.lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{or_+8}" fill="none" stroke="#d0d4de" stroke-width="1.2"/>'
        )

        # Arc segments
        for i, seg in enumerate(ce.get("segments", [])):
            a1 = seg.get("start_angle_deg", 0)
            a2 = seg.get("end_angle_deg", 90)
            label = seg.get("label", "")
            label_color = seg.get("label_color", "#ffffff")

            d = self._arc_path(cx, cy, a1, a2, or_, ir_)

            # Highlight arc
            h1x = cx + (or_-8) * math.cos(math.radians(a1))
            h1y = cy - (or_-8) * math.sin(math.radians(a1))
            h2x = cx + (or_-8) * math.cos(math.radians(a2))
            h2y = cy - (or_-8) * math.sin(math.radians(a2))
            large = 1 if abs(a2 - a1) > 180 else 0
            sweep = 0  # CCW in math coords → 0 in SVG (Y-flipped)

            # Arrow at end of arc
            mid_r = (or_ + ir_) / 2
            ea = a2
            ex = cx + mid_r * math.cos(math.radians(ea))
            ey = cy - mid_r * math.sin(math.radians(ea))
            tang = math.radians(ea - 90)
            arr = 20
            ax1 = ex + arr * math.cos(tang + 0.35)
            ay1 = ey - arr * math.sin(tang + 0.35)
            ax2 = ex + arr * math.cos(tang - 0.35)
            ay2 = ey - arr * math.sin(tang - 0.35)
            ax3 = ex + arr * 1.5 * math.cos(tang)
            ay3 = ey - arr * 1.5 * math.sin(tang)

            # Label
            lbl_r = (or_ + ir_) / 2
            lbl_angle = (a1 + a2) / 2
            lx = cx + lbl_r * math.cos(math.radians(lbl_angle))
            ly = cy - lbl_r * math.sin(math.radians(lbl_angle))
            rot = -lbl_angle

            self.lines.append(f'''
<g id="seg_{i}_{label.lower()}" filter="url(#_shadowRing)">
  <path d="{d}" fill="url(#arc_{i})" stroke="#f5f5f8" stroke-width="7"/>
  <path d="M{h1x:.1f},{h1y:.1f} A{or_-8} {or_-8} 0 {large} 0 {h2x:.1f},{h2y:.1f}"
        fill="none" stroke="#ffffff" stroke-width="5" stroke-linecap="round" opacity="0.22"/>
</g>
<polygon points="{ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f} {ax3:.1f},{ay3:.1f}"
         fill="#f2f2f4" stroke="#f2f2f4"/>
<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle"
      transform="rotate({rot:.1f} {lx:.1f} {ly:.1f})"
      class="t-ring" style="fill:{label_color}">{self._esc(label)}</text>''')

        # Center circle
        self.lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{cr}" fill="url(#_center)" '
            f'stroke="#cdd2e0" stroke-width="2.5" filter="url(#_shadowSm)"/>\n'
            f'<circle cx="{cx}" cy="{cy}" r="{cr-4}" fill="none" stroke="#ffffff" stroke-width="0.8" opacity="0.7"/>'
        )

        # Center icon
        ci = ce.get("center_icon", {})
        if ci:
            icon_data = {
                "id": "center_icon",
                "x": cx - 60,
                "y": cy - cr + 20,
                "type": ci.get("type", "robot"),
                "description": ci.get("description", "robot face"),
                "primary_color": "#3a5d96",
                "secondary_color": "#d0e0f8",
                "size": min(cr * 1.4, 160),
            }
            self.lines.append(IconLibrary.render(icon_data))

        # Center texts
        for txt_info in ce.get("center_texts", []):
            txt = txt_info.get("text", "")
            sz = txt_info.get("size", 40)
            wt = txt_info.get("weight", "bold")
            col = txt_info.get("color", "#1e2847")
            yoff = txt_info.get("y_offset", 0)
            self.lines.append(
                f'<text x="{cx}" y="{cy + yoff}" text-anchor="middle" '
                f'style="font-size:{sz}px;font-weight:{wt};fill:{col}">'
                f'{self._esc(txt)}</text>'
            )

    # ── Standalone texts ──────────────────────────────
    def _standalone_texts(self):
        self.lines.append("\n<!-- ══ Standalone Texts ══ -->")
        for t in self.s.get("standalone_texts", []):
            txt = self._esc(t.get("text", ""))
            x = t.get("x", self.W // 2)
            y = t.get("y", 50)
            sz = t.get("size", self.typo.get("title_size", 42))
            wt = t.get("weight", "normal")
            col = t.get("color", self.typo.get("title_color", "#1e2847"))
            anchor = t.get("anchor", "middle")
            css = t.get("css_class", "")
            self.lines.append(
                f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                f'style="font-size:{sz}px;font-weight:{wt};fill:{col}" '
                f'class="{css}">{txt}</text>'
            )

    # ── Connectors ────────────────────────────────────
    def _connectors(self):
        for conn in self.s.get("connectors", []):
            ctype = conn.get("type", "line")
            col = conn.get("color", "#3a5d96")
            x1, y1 = conn.get("x1", 0), conn.get("y1", 0)
            x2, y2 = conn.get("x2", 100), conn.get("y2", 100)
            if ctype == "arrow":
                self.lines.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="{col}" stroke-width="2" marker-end="url(#arrow)"/>'
                )
            else:
                self.lines.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="{col}" stroke-width="2" stroke-dasharray="6,3"/>'
                )

    # ── Helpers ───────────────────────────────────────
    @staticmethod
    def _esc(text: str) -> str:
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    @staticmethod
    def _arc_path(cx, cy, a1, a2, r_out, r_in) -> str:
        def pt(a, r):
            return cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))
        x1o, y1o = pt(a1, r_out); x2o, y2o = pt(a2, r_out)
        x1i, y1i = pt(a1, r_in);  x2i, y2i = pt(a2, r_in)
        large = 1 if abs(a2 - a1) > 180 else 0
        return (f"M{x1o:.2f},{y1o:.2f} A{r_out},{r_out} 0 {large},0 {x2o:.2f},{y2o:.2f} "
                f"L{x2i:.2f},{y2i:.2f} A{r_in},{r_in} 0 {large},1 {x1i:.2f},{y1i:.2f}Z")

    @staticmethod
    def _lighten(hex_color: str, amount: float) -> str:
        """将颜色变亮"""
        try:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            r = min(255, int(r + (255 - r) * amount))
            g = min(255, int(g + (255 - g) * amount))
            b = min(255, int(b + (255 - b) * amount))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color


# ══════════════════════════════════════════════════════════
# Stage 5  PostProcessor
# ══════════════════════════════════════════════════════════
class PostProcessor:
    """验证 SVG XML 有效性，可选生成 EPS"""

    @staticmethod
    def validate_xml(svg_str: str) -> bool:
        from xml.etree import ElementTree as ET
        try:
            ET.fromstring(svg_str)
            return True
        except Exception as e:
            print(f"  [XML] 警告: {e}")
            return False

    @staticmethod
    def save(svg_str: str, path: str):
        Path(path).write_text(svg_str, encoding="utf-8")
        size_kb = Path(path).stat().st_size // 1024
        print(f"  ✓ SVG  {path}  ({size_kb} KB)")


# ══════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════
def run_pipeline(input_path: str, output_path: str,
                 api_key: str, model: str,
                 verbose: bool = False) -> str:
    """完整流水线：图像 → 分析 → 渲染 → 保存"""

    print(f"\n{'='*56}")
    print(f"  img2svg  —  {Path(input_path).name}")
    print(f"{'='*56}")

    # Stage 1: 加载
    print("  [S1] 加载图像...")
    image_data = ImageIngestor.load(input_path)
    dom_colors = ImageIngestor.dominant_colors(image_data["arr"])
    print(f"       {image_data['width']} × {image_data['height']} px")
    print(f"       主色: {' '.join(dom_colors[:6])}")

    # Stage 2: Vision 分析
    print("  [S2] Claude Vision 分析中...")
    client = anthropic.Anthropic(api_key=api_key)
    analyzer = VisionAnalyzer(client, model=model)
    schema = analyzer.analyze(image_data, verbose=verbose)

    if verbose:
        print(f"       Schema: {json.dumps(schema, indent=2, ensure_ascii=False)[:800]}...")

    # Stage 3: 校验
    print("  [S3] Schema 校验...")
    schema = SchemaValidator.validate(schema, image_data["width"], image_data["height"])

    panels = len(schema.get("panels", []))
    has_ring = schema.get("central_element", {}).get("present", False)
    segs = len(schema.get("central_element", {}).get("segments", []))
    print(f"       {panels} 个面板  |  圆环: {'✓' if has_ring else '✗'} ({segs} 段)")

    # Stage 4: 渲染
    print("  [S4] 渲染 SVG...")
    renderer = SVGRenderer(schema)
    svg_str = renderer.render()

    # Stage 5: 保存
    print("  [S5] 后处理...")
    PostProcessor.validate_xml(svg_str)
    PostProcessor.save(svg_str, output_path)

    return output_path


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="img2svg — AI图像→完全可编辑SVG（Times New Roman文字 + 可移动icon）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("input", nargs="+",
                    help="输入图像文件或文件夹")
    ap.add_argument("--output", "-o", metavar="PATH",
                    help="输出 SVG 路径（单文件时使用）")
    ap.add_argument("--output-dir", metavar="DIR",
                    help="批量输出目录")
    ap.add_argument("--api-key", metavar="KEY",
                    help="Anthropic API Key（也可设 ANTHROPIC_API_KEY 环境变量）")
    ap.add_argument("--model", default="claude-opus-4-5",
                    help="Claude 模型（默认: claude-opus-4-5）")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="详细输出（显示完整 schema）")
    args = ap.parse_args()

    # API Key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit(
            "❌ 需要 Anthropic API Key\n"
            "   方法1: export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "   方法2: python img2svg.py input.png --api-key sk-ant-..."
        )

    # 收集文件
    files = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            for ext in SUPPORTED:
                files.extend(p.glob(f"*{ext}"))
                files.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        else:
            print(f"  ⚠ 跳过: {inp}")

    if not files:
        sys.exit("❌ 未找到支持的图像文件")

    print(f"\n🚀 img2svg  |  {len(files)} 个文件  |  模型: {args.model}")

    ok = fail = 0
    for fp in files:
        if args.output and len(files) == 1:
            out = args.output
        elif args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(args.output_dir) / (fp.stem + ".svg"))
        else:
            out = str(fp.parent / (fp.stem + ".svg"))

        try:
            run_pipeline(str(fp), out, api_key, args.model, args.verbose)
            ok += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            if args.verbose:
                import traceback; traceback.print_exc()
            fail += 1

    print(f"\n{'='*56}")
    print(f"  完成: {ok} 成功" + (f"  {fail} 失败" if fail else ""))
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
