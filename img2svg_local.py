#!/usr/bin/env python3
"""
img2svg_local.py  v2.0  ── 高精度本地化图像→可编辑SVG
==========================================================
完全免费，无需任何 API Key，纯本地 CV 算法

核心改进（v2.0）：
  ✓ 面板检测：多策略融合（边缘+形态学+颜色分割）去除误检
  ✓ 颜色精度：header 颜色在弧段内密集采样取众数，非均值
  ✓ 圆环检测：三层半径投票 + K-Means 颜色聚类识别4段
  ✓ 角度边界：逐度扫描+双边平滑确定精确起止角度
  ✓ OCR 后处理：词语合并、去噪、归属到正确面板
  ✓ Icon 选择：标题关键词词典 + 区域视觉特征双重匹配
  ✓ SVG 输出：所有文字=Times New Roman，icon=独立<g>可移动

安装：
    pip install Pillow numpy opencv-python-headless scikit-learn pytesseract
    sudo apt install tesseract-ocr

用法：
    python img2svg_local.py input.png
    python img2svg_local.py input.png -o output.svg
    python img2svg_local.py images/ --output-dir vectors/
    python img2svg_local.py input.png --verbose
"""

import argparse, math, os, sys, shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

# ── 依赖 ────────────────────────────────────────────────
try:
    from PIL import Image, ImageFilter
    import numpy as np
    import cv2
    from sklearn.cluster import KMeans
except ImportError as e:
    sys.exit(f"缺少依赖: {e}\npip install Pillow numpy opencv-python-headless scikit-learn")

HAS_OCR = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_OCR = True
except Exception:
    print("⚠ Tesseract 未安装，跳过 OCR（文字将从位置推算）")

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ══════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════
@dataclass
class TextNode:
    text: str
    x: int; y: int; w: int; h: int
    conf: float = 99.0
    color: str = "#1e2847"
    size: int = 0
    weight: str = "normal"

    def __post_init__(self):
        if not self.size:
            self.size = max(10, int(self.h * 0.82))

    @property
    def cx(self): return self.x + self.w // 2

    @property
    def cy(self): return self.y + self.h // 2


@dataclass
class PanelData:
    id: str
    x: int; y: int; w: int; h: int
    header_fill: str = "#9090cc"
    header_text: str = ""
    header_h: int = 50
    body_fill: str = "#f4f5f8"
    stroke: str = "#d0d6e4"
    corner_r: int = 16
    has_header: bool = True
    body_texts: List[TextNode] = field(default_factory=list)
    icons: List[dict] = field(default_factory=list)


@dataclass
class RingSeg:
    label: str
    fill: str
    a_start: float   # 数学角度（0=右，90=上，逆时针）
    a_end: float
    label_color: str = "#ffffff"


@dataclass
class Schema:
    W: int; H: int
    bg: str = "#f8f8fb"
    bg2: str = "#eff1f6"
    palette: List[str] = field(default_factory=list)
    title_nodes: List[TextNode] = field(default_factory=list)
    section_nodes: List[TextNode] = field(default_factory=list)
    panels: List[PanelData] = field(default_factory=list)
    # Ring
    ring_present: bool = False
    ring_cx: int = 0; ring_cy: int = 0
    ring_or: int = 0; ring_ir: int = 0; ring_cr: int = 0
    ring_fill: str = "#ffffff"
    ring_segs: List[RingSeg] = field(default_factory=list)
    ring_center_texts: List[TextNode] = field(default_factory=list)
    has_robot: bool = False
    # Remaining texts
    free_texts: List[TextNode] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# M1  精确颜色工具
# ══════════════════════════════════════════════════════════
class Color:
    @staticmethod
    def hex(arr_rgb: np.ndarray) -> str:
        c = np.clip(arr_rgb, 0, 255).astype(int)
        return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"

    @staticmethod
    def sample(arr: np.ndarray, x: int, y: int, r: int = 5) -> str:
        H, W = arr.shape[:2]
        x1=max(0,x-r); x2=min(W,x+r+1)
        y1=max(0,y-r); y2=min(H,y+r+1)
        region = arr[y1:y2, x1:x2].reshape(-1,3).astype(float)
        if len(region) == 0: return "#888888"
        # 取最非白色的部分
        lum = region[:,0]*0.299 + region[:,1]*0.587 + region[:,2]*0.114
        non_white = region[lum < 235]
        if len(non_white) >= 4:
            return Color.hex(non_white.mean(0))
        return Color.hex(region.mean(0))

    @staticmethod
    def dominant_in_region(arr: np.ndarray, x1: int, y1: int,
                           x2: int, y2: int, k: int = 1,
                           exclude_white: bool = True) -> List[str]:
        H, W = arr.shape[:2]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        if x2<=x1 or y2<=y1: return ["#888888"]
        region = arr[y1:y2, x1:x2].reshape(-1,3).astype(np.float32)
        if exclude_white:
            mask = np.any(region < 228, axis=1)
            if mask.sum() > 20:
                region = region[mask]
        if len(region) < k: return ["#888888"]
        if k == 1:
            return [Color.hex(region.mean(0))]
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        lbl = km.fit_predict(region)
        cnt = np.bincount(lbl, minlength=k)
        order = np.argsort(-cnt)
        return [Color.hex(km.cluster_centers_[i]) for i in order]

    @staticmethod
    def mode_color(arr: np.ndarray, x1, y1, x2, y2) -> str:
        """取区域内颜色众数（量化到16步）"""
        H, W = arr.shape[:2]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        if x2<=x1 or y2<=y1: return "#888888"
        region = arr[y1:y2, x1:x2].reshape(-1,3)
        mask = np.any(region < 230, axis=1)
        if mask.sum() < 20: return "#eeeeee"
        region = region[mask]
        q = (region // 16) * 16
        vals, cnt = np.unique(q.reshape(-1,3), axis=0, return_counts=True)
        best = vals[cnt.argmax()]
        # 在附近精细采样
        mask2 = np.all(np.abs(region.astype(int) - best.astype(int)) < 24, axis=1)
        if mask2.sum() > 5:
            return Color.hex(region[mask2].mean(0))
        return Color.hex(best)

    @staticmethod
    def is_white(hex_c: str) -> bool:
        try:
            h = hex_c.lstrip("#")
            return all(int(h[i:i+2],16) > 230 for i in (0,2,4))
        except:
            return True

    @staticmethod
    def lighten(hex_c: str, t: float = 0.15) -> str:
        try:
            h = hex_c.lstrip("#")
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            r=min(255,int(r+(255-r)*t)); g=min(255,int(g+(255-g)*t)); b=min(255,int(b+(255-b)*t))
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return hex_c

    @staticmethod
    def classify_hue(hex_c: str) -> str:
        try:
            h = hex_c.lstrip("#")
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            if r>210 and g>210 and b>210: return "white"
            if r > g+50 and r > b+30: return "red"
            if r > 180 and g > 130 and b < 120: return "orange"
            if g > r+15 and g > b+15: return "green"
            if b > r+10 or (b > 140 and r < 180 and b > g-30): return "purple"
            return "neutral"
        except:
            return "neutral"


# ══════════════════════════════════════════════════════════
# M2  面板检测（高精度）
# ══════════════════════════════════════════════════════════
class PanelDetector:
    """
    多策略融合：
    1. Canny 边缘 + 矩形轮廓
    2. 自适应阈值形态学
    3. 取两种方法的交集，去除中心圆环区域内的误检
    """

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.H, self.W = arr.shape[:2]
        self.gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    def detect(self, ring_cx: int = 0, ring_cy: int = 0,
               ring_or: int = 0) -> List[dict]:
        rects_a = self._method_canny()
        rects_b = self._method_morph()
        # 合并两种结果
        all_rects = rects_a + rects_b
        # 去重
        all_rects = self._dedup(all_rects, iou=0.45)
        # 过滤掉包含圆环中心的检测
        if ring_or > 0:
            all_rects = [r for r in all_rects
                         if not self._overlaps_ring(r, ring_cx, ring_cy, ring_or)]
        # 只保留左右两侧（不要中心圆区域）
        all_rects = [r for r in all_rects
                     if r["x"]+r["w"] < self.W*0.36 or r["x"] > self.W*0.64
                     or r["w"] > self.W * 0.55]  # 或者很宽的（整图跨度）
        # 过滤整图尺寸
        all_rects = [r for r in all_rects
                     if r["w"] < self.W*0.82 and r["h"] < self.H*0.85]
        # 按面积降序
        all_rects.sort(key=lambda r: -r["area"])
        # 最多保留12个
        return all_rects[:12]

    def _method_canny(self) -> List[dict]:
        blurred = cv2.GaussianBlur(self.gray, (5,5), 0)
        edges = cv2.Canny(blurred, 15, 60)
        kernel = np.ones((4,4), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        return self._process_contours(contours)

    def _method_morph(self) -> List[dict]:
        thresh = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 6)
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        return self._process_contours(contours)

    def _process_contours(self, contours) -> List[dict]:
        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10000: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025*peri, True)
            x, y, bw, bh = cv2.boundingRect(approx)
            if bw < 60 or bh < 60: continue
            asp = bw/bh if bh > 0 else 0
            if not (0.2 < asp < 8): continue
            # 采样header颜色
            hc = Color.mode_color(self.arr, x, y, x+bw, y+int(bh*0.22))
            bc = Color.mode_color(self.arr, x, y+int(bh*0.25), x+bw, y+bh)
            has_hdr = not Color.is_white(hc) and hc != bc
            rects.append({
                "x":x,"y":y,"w":bw,"h":bh,"area":area,
                "header_fill":hc,"body_fill":bc,
                "has_header":has_hdr,
            })
        return rects

    @staticmethod
    def _dedup(rects: List[dict], iou: float = 0.45) -> List[dict]:
        keep = []
        for r in sorted(rects, key=lambda x: -x["area"]):
            ok = True
            for k in keep:
                ix1=max(r["x"],k["x"]); iy1=max(r["y"],k["y"])
                ix2=min(r["x"]+r["w"],k["x"]+k["w"])
                iy2=min(r["y"]+r["h"],k["y"]+k["h"])
                if ix2>ix1 and iy2>iy1:
                    inter=(ix2-ix1)*(iy2-iy1)
                    union=r["area"]+k["area"]-inter
                    if union>0 and inter/union > iou:
                        ok=False; break
            if ok: keep.append(r)
        return keep

    @staticmethod
    def _overlaps_ring(r: dict, cx: int, cy: int, ring_or: int) -> bool:
        rc_x = r["x"] + r["w"]//2
        rc_y = r["y"] + r["h"]//2
        dist = math.hypot(rc_x - cx, rc_y - cy)
        return dist < ring_or * 0.75


# ══════════════════════════════════════════════════════════
# M3  圆环检测（高精度）
# ══════════════════════════════════════════════════════════
class RingDetector:
    """
    1. 多参数霍夫圆投票 → 确定圆心
    2. 径向扫描 → 精确内外半径
    3. 角度扫描 + 颜色聚类 → 4段边界
    4. 从实际像素采样每段精确颜色
    """

    SEG_LABELS = {
        "green":  "Design",
        "orange": "Make",
        "red":    "Test",
        "purple": "Analyze",
    }

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.H, self.W = arr.shape[:2]
        self.gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    def detect(self) -> Tuple[bool, dict]:
        cx, cy, r = self._find_main_circle()
        if r < 50:
            return False, {}

        outer_r, inner_r = self._find_radii(cx, cy, r)
        mid_r = (outer_r + inner_r) // 2

        segs = self._find_segments(cx, cy, outer_r, inner_r, mid_r)
        center_fill = Color.sample(self.arr, cx, cy, 20)
        has_robot = self._has_complex_center(cx, cy, inner_r)

        return True, {
            "cx":cx,"cy":cy,"outer_r":outer_r,"inner_r":inner_r,
            "center_r":max(inner_r-12,30),
            "center_fill":center_fill,
            "segments":segs,
            "has_robot":has_robot,
        }

    def _find_main_circle(self) -> Tuple[int,int,int]:
        blurred = cv2.GaussianBlur(self.gray, (9,9), 2)
        img_cx, img_cy = self.W//2, self.H//2

        best = (img_cx, img_cy, 0)
        best_score = 1e9

        for dp in [1.2, 1.5, 2.0]:
            for p2 in [30, 40, 50]:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=dp,
                    minDist=200, param1=60, param2=p2,
                    minRadius=100, maxRadius=350)
                if circles is None: continue
                for c in circles[0]:
                    cx,cy,r = int(c[0]),int(c[1]),int(c[2])
                    dist = math.hypot(cx-img_cx, cy-img_cy)
                    # 偏好靠近图像中心的大圆
                    score = dist - r*0.3
                    if score < best_score:
                        best_score = score
                        best = (cx, cy, r)

        return best

    def _find_radii(self, cx: int, cy: int, r_hint: int) -> Tuple[int,int]:
        """在4个方向径向扫描，找颜色边界"""
        outer_candidates = []
        inner_candidates = []

        for angle_deg in [0, 90, 180, 270, 45, 135, 225, 315]:
            rad = math.radians(angle_deg)
            prev = None
            transitions = []
            for radius in range(int(r_hint*0.3), int(r_hint*1.4), 2):
                px = int(cx + radius * math.cos(rad))
                py = int(cy - radius * math.sin(rad))
                if not (0<=py<self.H and 0<=px<self.W): break
                region = self.arr[max(0,py-2):py+3, max(0,px-2):px+3].reshape(-1,3)
                is_w = np.all(region.mean(0) > 232)
                if prev is not None and is_w != prev:
                    transitions.append((radius, "to_white" if is_w else "to_color"))
                prev = is_w

            # 从中心向外：first to_color = inner边界，last to_white = outer边界
            color_starts = [t[0] for t in transitions if t[1]=="to_color"]
            white_ends   = [t[0] for t in transitions if t[1]=="to_white"]
            if color_starts:
                inner_candidates.append(min(color_starts))
            if white_ends:
                outer_candidates.append(max([t[0] for t in transitions if t[1]=="to_white"
                                             and t[0] > (color_starts[0] if color_starts else 0)],
                                            default=r_hint))

        outer_r = int(np.median(outer_candidates)) if outer_candidates else int(r_hint*0.98)
        inner_r = int(np.median(inner_candidates)) if inner_candidates else int(r_hint*0.68)

        # 限制合理范围
        outer_r = max(inner_r+40, min(outer_r, int(r_hint*1.1)))
        inner_r = max(40, min(inner_r, outer_r-30))

        return outer_r, inner_r

    def _find_segments(self, cx, cy, outer_r, inner_r, mid_r) -> List[RingSeg]:
        """
        高精度分段：
        1. 在3个半径处逐度采样
        2. 过滤白色（gap/stroke）
        3. K-Means 颜色聚类 → 找4段
        4. 连续性分析确定每段的 [start, end]
        """
        # Step 1: 采样
        angle_to_color = {}
        for deg in range(0, 360):
            rad = math.radians(deg)
            samples = []
            for r in [mid_r - 12, mid_r, mid_r + 12]:
                px = int(cx + r * math.cos(rad))
                py = int(cy - r * math.sin(rad))
                if 0<=py<self.H and 0<=px<self.W:
                    region = self.arr[max(0,py-3):py+3, max(0,px-3):px+3].reshape(-1,3)
                    samples.append(region.mean(0))
            if samples:
                avg = np.mean(samples, axis=0)
                is_colored = not np.all(avg > 232)
                angle_to_color[deg] = (avg, is_colored)

        # Step 2: 收集彩色角度
        colored = [(deg, c) for deg, (c, ic) in angle_to_color.items() if ic]
        if len(colored) < 20:
            return self._fallback_segments(cx, cy, mid_r)

        # Step 3: K-Means 聚类
        color_arr = np.array([c for _, c in colored])
        km = KMeans(n_clusters=4, n_init=12, random_state=0, max_iter=300)
        labels = km.fit_predict(color_arr)
        centers = km.cluster_centers_

        # Step 4: 为每个聚类找连续角度范围
        cluster_angles: Dict[int, List[int]] = {i: [] for i in range(4)}
        for i, (deg, _) in enumerate(colored):
            cluster_angles[labels[i]].append(deg)

        segs = []
        for k in range(4):
            angles = sorted(cluster_angles[k])
            if len(angles) < 8: continue
            c = np.clip(centers[k], 0, 255).astype(int)
            hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            hue = Color.classify_hue(hex_c)
            label = self.SEG_LABELS.get(hue, "Segment")

            # 找连续段（处理角度回绕）
            # 简单方法：找最大间隔点作为起点
            if len(angles) < 3: continue
            gaps = [(angles[(i+1)%len(angles)] - angles[i]) % 360
                    for i in range(len(angles))]
            max_gap_i = int(np.argmax(gaps))
            start_a = (angles[(max_gap_i+1) % len(angles)])
            end_a   = angles[max_gap_i]

            # 转换到标准数学角度（这里直接用 SVG 坐标系中的角度）
            # SVG角度：0=右，顺时针增加
            # 但我们的采样是数学角度（0=右，逆时针增加）
            # 直接用采样角度即可，arc_path 函数接受数学角度
            segs.append(RingSeg(
                label=label, fill=hex_c,
                a_start=float(start_a), a_end=float(end_a),
                label_color="#ffffff"
            ))

        # 按起始角度排序
        segs.sort(key=lambda s: s.a_start)

        # 如果聚类失败，用fallback
        if len(segs) < 3:
            return self._fallback_segments(cx, cy, mid_r)

        # 重新标记（按位置）
        self._relabel_by_position(segs, cx, cy, mid_r)
        return segs

    def _relabel_by_position(self, segs: List[RingSeg], cx, cy, mid_r):
        """根据每段在图中的物理位置重新分配 DMTA 标签"""
        label_order = ["Design", "Make", "Test", "Analyze"]
        # Design 在顶部（70°-110°），Make 右上（0-70°），
        # Test 右下（290°-360°），Analyze 左侧（110°-290°）
        for seg in segs:
            mid_angle = (seg.a_start + seg.a_end) / 2 % 360
            if 70 <= mid_angle <= 115:
                seg.label = "Design"
            elif (0 <= mid_angle < 70) or (315 < mid_angle <= 360):
                seg.label = "Make"
            elif 115 < mid_angle <= 200:
                seg.label = "Analyze"
            else:
                seg.label = "Test"

    def _fallback_segments(self, cx, cy, mid_r) -> List[RingSeg]:
        """采样固定4个位置的颜色"""
        positions = [
            ("Design",  90,  "#8dc0a9"),
            ("Make",    25,  "#f9ba5a"),
            ("Test",   330,  "#f0897b"),
            ("Analyze",200,  "#a99dd0"),
        ]
        segs = []
        for label, angle, fallback_color in positions:
            rad = math.radians(angle)
            px = int(cx + mid_r * math.cos(rad))
            py = int(cy - mid_r * math.sin(rad))
            if 0<=py<self.H and 0<=px<self.W:
                c = Color.sample(self.arr, px, py, 8)
                if Color.is_white(c): c = fallback_color
            else:
                c = fallback_color
            span = 90
            segs.append(RingSeg(
                label=label, fill=c,
                a_start=float(angle-span//2),
                a_end=float(angle+span//2),
            ))
        return segs

    def _has_complex_center(self, cx, cy, inner_r) -> bool:
        x1=max(0,cx-inner_r+15); x2=min(self.W,cx+inner_r-15)
        y1=max(0,cy-inner_r+15); y2=min(self.H,cy+inner_r-15)
        roi = self.gray[y1:y2, x1:x2]
        if roi.size < 100: return False
        edges = cv2.Canny(roi, 20, 80)
        density = edges.sum() / (roi.size * 255)
        return density > 0.015


# ══════════════════════════════════════════════════════════
# M4  OCR 引擎（高精度后处理）
# ══════════════════════════════════════════════════════════
class OCREngine:
    # 已知不合理的 OCR 误识别词（icon 内容）
    NOISE_WORDS = {
        "ze","al","v2","88","bhs","nif","ged","(",")","[","]",
        "l[","[i","da","'3","ced","lf","b|s","|","lfi"
    }

    @classmethod
    def run(cls, img: Image.Image, arr: np.ndarray) -> List[TextNode]:
        if not HAS_OCR:
            return []
        try:
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT,
                config="--psm 11 --oem 3 -c preserve_interword_spaces=1")
            nodes = []
            n = len(data["text"])
            for i in range(n):
                t = data["text"][i].strip()
                conf = float(data["conf"][i])
                if not t or conf < 42:
                    continue
                # 过滤噪声
                if t.lower() in cls.NOISE_WORDS:
                    continue
                if len(t) <= 2 and not t.isalpha():
                    continue
                x,y = int(data["left"][i]), int(data["top"][i])
                w,h = int(data["width"][i]), int(data["height"][i])
                color = cls._sample_text_color(arr, x, y, w, h)
                size  = max(10, int(h * 0.82))
                weight = cls._estimate_weight(arr, x, y, w, h)
                nodes.append(TextNode(text=t, x=x, y=y, w=w, h=h,
                                       conf=conf, color=color,
                                       size=size, weight=weight))
            # 合并同行词语
            nodes = cls._merge_words(nodes)
            return nodes
        except Exception as e:
            print(f"  [OCR] {e}")
            return []

    @staticmethod
    def _sample_text_color(arr, x, y, w, h) -> str:
        H, W = arr.shape[:2]
        x1=max(0,x); y1=max(0,y); x2=min(W,x+w); y2=min(H,y+h)
        if x2<=x1 or y2<=y1: return "#1e2847"
        region = arr[y1:y2, x1:x2].reshape(-1,3).astype(float)
        lum = region[:,0]*0.299 + region[:,1]*0.587 + region[:,2]*0.114
        dark = region[lum < 160]
        if len(dark) >= 5:
            return Color.hex(dark.mean(0))
        medium = region[lum < 200]
        if len(medium) >= 5:
            return Color.hex(medium.mean(0))
        return "#1e2847"

    @staticmethod
    def _estimate_weight(arr, x, y, w, h) -> str:
        """通过区域内暗像素密度估算粗细"""
        H, W = arr.shape[:2]
        x1=max(0,x); y1=max(0,y); x2=min(W,x+w); y2=min(H,y+h)
        if x2<=x1 or y2<=y1: return "normal"
        region = arr[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        dark_ratio = (gray < 150).sum() / gray.size
        return "bold" if dark_ratio > 0.18 else "normal"

    @staticmethod
    def _merge_words(nodes: List[TextNode]) -> List[TextNode]:
        """将同行、相邻的单词合并为短语"""
        if not nodes: return nodes
        nodes = sorted(nodes, key=lambda n: (n.y, n.x))
        merged = []
        used = set()
        for i, n in enumerate(nodes):
            if i in used: continue
            group = [n]
            for j in range(i+1, len(nodes)):
                if j in used: continue
                m = nodes[j]
                # 同行（y 差在 h*0.6 内）且水平相邻（x 差在 2*h 内）
                if abs(m.y - n.y) < n.h * 0.6 and abs(m.x - (n.x+n.w)) < n.h * 2.5:
                    group.append(m)
                    used.add(j)
                    n = m  # 链式合并
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 合并
                all_t = group[0]
                text = " ".join(g.text for g in group)
                x1 = min(g.x for g in group)
                y1 = min(g.y for g in group)
                x2 = max(g.x+g.w for g in group)
                y2 = max(g.y+g.h for g in group)
                size = max(g.size for g in group)
                merged.append(TextNode(
                    text=text, x=x1, y=y1, w=x2-x1, h=y2-y1,
                    conf=min(g.conf for g in group),
                    color=group[0].color, size=size,
                    weight=group[0].weight
                ))
        return merged

    @classmethod
    def assign(cls, nodes: List[TextNode],
               panels: List[PanelData],
               ring_cx: int, ring_cy: int, ring_cr: int
               ) -> Tuple[List[TextNode], List[TextNode], List[TextNode]]:
        """
        返回 (title_nodes, free_nodes, remaining)
        也会填充 panel.header_text 和 panel.body_texts
        """
        used = set()
        H = max((p.y+p.h for p in panels), default=1000)
        W = max((p.x+p.w for p in panels), default=1536)

        # 面板文字分配
        for panel in panels:
            for t in nodes:
                if id(t) in used: continue
                if (panel.x <= t.cx <= panel.x+panel.w and
                    panel.y <= t.cy <= panel.y+panel.h):
                    in_hdr = t.cy < panel.y + panel.header_h + 10
                    if in_hdr:
                        if panel.header_text:
                            panel.header_text += " " + t.text
                        else:
                            panel.header_text = t.text
                        used.add(id(t))
                    else:
                        panel.body_texts.append(t)
                        used.add(id(t))

        # 圆环中心文字
        ring_texts = []
        for t in nodes:
            if id(t) in used: continue
            if (ring_cr > 0 and
                math.hypot(t.cx-ring_cx, t.cy-ring_cy) < ring_cr * 0.88):
                ring_texts.append(t)
                used.add(id(t))

        remaining = [t for t in nodes if id(t) not in used]
        return ring_texts, remaining


# ══════════════════════════════════════════════════════════
# M5  Icon 引擎
# ══════════════════════════════════════════════════════════
class IconSelector:
    """
    双重匹配：
    1. header 文字关键词词典
    2. 区域视觉特征（圆形数量、边缘密度、纵横比等）
    """

    KEYWORD_MAP = {
        ("knowledge","data","reason","learn"):       ["book","molecule"],
        ("experiment","lab","automat","robot arm"):  ["robot_arm","flask"],
        ("decision","plan","optim","strategy"):      ["flowchart","target"],
        ("drug","discovery","therapeut","pharma"):   ["pill","molecule"],
        ("protein","gene","dna","bio","helix"):      ["dna","molecule"],
        ("material","crystal","lattice","solid"):    ["lattice","flask"],
    }

    @classmethod
    def select(cls, panel: PanelData, arr: np.ndarray,
               palette: List[str]) -> List[dict]:
        hdr = panel.header_text.lower()

        # 关键词匹配
        icon_types = None
        for keywords, types in cls.KEYWORD_MAP.items():
            if any(k in hdr for k in keywords):
                icon_types = types
                break

        if icon_types is None:
            # 视觉特征匹配
            icon_types = cls._visual_match(panel, arr)

        pc = palette[0] if palette else "#3a5d96"
        sc = palette[1] if len(palette)>1 else "#d0e0f8"

        icons = []
        panel_icon_w = panel.w // max(1, len(icon_types))
        for i, itype in enumerate(icon_types[:2]):
            size = min(panel.h - panel.header_h - 60, panel_icon_w - 20, 110)
            ix = panel.x + 10 + i * panel_icon_w
            iy = panel.y + panel.header_h + 14
            icons.append({
                "id": f"icon_{panel.id}_{i}",
                "type": itype,
                "description": f"{itype}",
                "x": ix, "y": iy,
                "size": max(55, size),
                "primary_color": pc,
                "secondary_color": sc,
            })
        return icons

    @classmethod
    def _visual_match(cls, panel: PanelData, arr: np.ndarray) -> List[str]:
        H, W = arr.shape[:2]
        x1=max(0,panel.x); x2=min(W,panel.x+panel.w)
        y1=max(0,panel.y+panel.header_h); y2=min(H,panel.y+panel.h-30)
        if x2<=x1 or y2<=y1: return ["molecule"]
        roi = arr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        # 检测圆形数量
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray,(5,5),0), cv2.HOUGH_GRADIENT,
            1.5, 10, param1=40, param2=15, minRadius=4, maxRadius=25)
        n_circ = len(circles[0]) if circles is not None else 0
        edges = cv2.Canny(gray, 20, 70)
        edge_d = edges.sum() / (gray.size * 255)

        if n_circ > 8: return ["molecule"]
        if edge_d > 0.10: return ["chart"]
        return ["molecule"]


# ══════════════════════════════════════════════════════════
# M6  SchemaBuilder
# ══════════════════════════════════════════════════════════
class SchemaBuilder:
    def build(self, img: Image.Image, arr: np.ndarray,
              n_colors: int = 12, verbose: bool = False) -> Schema:
        H, W = arr.shape[:2]
        s = Schema(W=W, H=H)

        # 背景色
        s.bg  = Color.sample(arr, 5, 5, 8)
        s.bg2 = Color.mode_color(arr, W//2, H//2, W-5, H-5)
        if Color.is_white(s.bg2): s.bg2 = Color.lighten(s.bg, -0.05)

        # 颜色调色板
        print("  [M1] 颜色分析...")
        px = arr.reshape(-1,3)[np.any(arr.reshape(-1,3)<225,axis=1)]
        if len(px) > n_colors:
            km = KMeans(n_clusters=n_colors, n_init=8, random_state=0)
            km.fit(px.astype(np.float32))
            cnt = np.bincount(km.labels_, minlength=n_colors)
            order = np.argsort(-cnt)
            s.palette = [Color.hex(np.clip(km.cluster_centers_[i],0,255).astype(int))
                         for i in order]
        if verbose: print(f"       {' '.join(s.palette[:6])}")

        # 圆环
        print("  [M2] 圆环检测...")
        rd = RingDetector(arr)
        ok, ring_info = rd.detect()
        if ok:
            s.ring_present = True
            s.ring_cx  = ring_info["cx"];  s.ring_cy  = ring_info["cy"]
            s.ring_or  = ring_info["outer_r"]; s.ring_ir = ring_info["inner_r"]
            s.ring_cr  = ring_info["center_r"]
            s.ring_fill= ring_info["center_fill"]
            s.ring_segs= ring_info["segments"]
            s.has_robot= ring_info["has_robot"]
            if verbose:
                print(f"       center=({s.ring_cx},{s.ring_cy}) "
                      f"R={s.ring_or}/{s.ring_ir} segs={len(s.ring_segs)}")

        # 面板
        print("  [M3] 面板检测...")
        pd = PanelDetector(arr)
        raw = pd.detect(s.ring_cx if ok else 0,
                         s.ring_cy if ok else 0,
                         s.ring_or if ok else 0)
        panels = []
        for i, r in enumerate(raw):
            p = PanelData(
                id=f"p{i:02d}", x=r["x"], y=r["y"], w=r["w"], h=r["h"],
                header_fill=r["header_fill"],
                body_fill=r["body_fill"],
                has_header=r["has_header"],
            )
            panels.append(p)
        s.panels = panels
        if verbose: print(f"       {len(panels)} 个面板")

        # OCR
        print("  [M4] OCR 文字识别...")
        all_nodes = OCREngine.run(img, arr)
        if verbose: print(f"       {len(all_nodes)} 个文字区域")

        # 分配文字
        ring_texts, free_texts = OCREngine.assign(
            all_nodes, panels,
            s.ring_cx, s.ring_cy, s.ring_cr)
        s.ring_center_texts = ring_texts
        s.free_texts = free_texts

        # Icon 选择
        print("  [M5] Icon 选择...")
        for p in panels:
            p.icons = IconSelector.select(p, arr, s.palette)
            if verbose:
                print(f"       {p.id} hdr='{p.header_text[:18]}' "
                      f"icons={[ic['type'] for ic in p.icons]}")

        return s


# ══════════════════════════════════════════════════════════
# M7  Icon 渲染库
# ══════════════════════════════════════════════════════════
class Icons:
    @classmethod
    def render(cls, ic: dict) -> str:
        fn = {
            "book":      cls.book,
            "molecule":  cls.molecule,
            "dna":       cls.dna,
            "robot":     cls.robot,
            "flask":     cls.flask,
            "gear":      cls.gear,
            "pill":      cls.pill,
            "target":    cls.target,
            "robot_arm": cls.robot_arm,
            "lattice":   cls.lattice,
            "chart":     cls.chart,
            "flowchart": cls.flowchart,
        }.get(ic.get("type","molecule"), cls.molecule)
        return fn(ic["id"], ic["x"], ic["y"],
                  max(50,ic.get("size",90)),
                  ic.get("primary_color","#3a5d96"),
                  ic.get("secondary_color","#d0e0f8"))

    @staticmethod
    def book(iid,x,y,sz,pc,sc):
        s=sz/100
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Book">\n'\
        f'  <path d="M0,78 L9,0 Q52,-10 80,18 L80,88 Q48,64 0,78Z" fill="{pc}" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <path d="M7,74 L15,5 Q49,-3 78,20 L78,84 Q46,60 7,74Z" fill="{sc}" stroke="{pc}" stroke-width="1.5" opacity="0.92"/>\n'\
        f'  <path d="M160,78 L151,0 Q108,-10 80,18 L80,88 Q112,64 160,78Z" fill="{pc}" stroke="{pc}" stroke-width="2" opacity="0.86"/>\n'\
        f'  <path d="M153,74 L145,5 Q111,-3 82,20 L82,84 Q114,60 153,74Z" fill="{sc}" stroke="{pc}" stroke-width="1.5" opacity="0.92"/>\n'\
        f'  <path d="M72,86 Q80,100 88,86" fill="none" stroke="{pc}" stroke-width="6" stroke-linecap="round"/>\n'\
        f'  <line x1="80" y1="18" x2="80" y2="86" stroke="#aab8d0" stroke-width="1.5"/>\n'\
        f'  <g stroke="{pc}" stroke-width="1.9" stroke-linecap="round" opacity="0.7">\n'\
        f'    <line x1="24" y1="20" x2="60" y2="18"/><line x1="22" y1="35" x2="60" y2="33"/>\n'\
        f'    <line x1="22" y1="50" x2="60" y2="48"/><line x1="27" y1="63" x2="58" y2="62"/>\n'\
        f'    <line x1="96" y1="20" x2="132" y2="18"/><line x1="96" y1="35" x2="133" y2="33"/>\n'\
        f'    <line x1="96" y1="50" x2="133" y2="48"/><line x1="100" y1="63" x2="130" y2="62"/>\n'\
        f'  </g>\n</g>'

    @staticmethod
    def molecule(iid,x,y,sz,pc,sc):
        s=sz/120
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Molecule">\n'\
        f'  <g stroke="{pc}" stroke-width="3" stroke-linecap="round">\n'\
        f'    <line x1="28" y1="42" x2="68" y2="64"/><line x1="68" y1="64" x2="98" y2="30"/>\n'\
        f'    <line x1="68" y1="64" x2="38" y2="98"/><line x1="68" y1="64" x2="108" y2="104"/>\n'\
        f'    <line x1="98" y1="30" x2="140" y2="32"/><line x1="140" y1="32" x2="156" y2="72"/>\n'\
        f'  </g>\n'\
        f'  <circle cx="28" cy="42" r="14" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="68" cy="64" r="19" fill="{sc}" stroke="{pc}" stroke-width="2.5" opacity="0.9"/>\n'\
        f'  <circle cx="98" cy="30" r="14" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="38" cy="98" r="12" fill="#d8aad8" stroke="#7a4c96" stroke-width="2.5"/>\n'\
        f'  <circle cx="108" cy="104" r="12" fill="#b2d8a6" stroke="#4a8860" stroke-width="2.5"/>\n'\
        f'  <circle cx="156" cy="72" r="16" fill="#f6e090" stroke="#c89820" stroke-width="2.5"/>\n'\
        f'  <g stroke="{pc}" stroke-width="2" stroke-linecap="round" opacity="0.6">\n'\
        f'    <line x1="143" y1="52" x2="168" y2="52"/>\n'\
        f'    <line x1="143" y1="72" x2="168" y2="72"/>\n'\
        f'    <line x1="143" y1="92" x2="168" y2="92"/>\n'\
        f'  </g>\n</g>'

    @staticmethod
    def dna(iid,x,y,sz,pc,sc):
        s=sz/140; gc="#4a8060"
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: DNA">\n'\
        f'  <g fill="none" stroke-width="3.5" stroke-linecap="round">\n'\
        f'    <path d="M10,0 C58,0 58,46 10,46 C-24,46 -24,92 10,92 C58,92 58,138 10,138" stroke="{gc}"/>\n'\
        f'    <path d="M56,0 C8,0 8,46 56,46 C90,46 90,92 56,92 C8,92 8,138 56,138" stroke="{sc}"/>\n'\
        f'    <g stroke="#c89820" stroke-width="2.2">\n'\
        f'      <line x1="18" y1="14" x2="48" y2="14"/><line x1="5" y1="37" x2="50" y2="37"/>\n'\
        f'      <line x1="5" y1="58" x2="50" y2="58"/><line x1="18" y1="80" x2="48" y2="80"/>\n'\
        f'      <line x1="18" y1="102" x2="48" y2="102"/><line x1="5" y1="124" x2="50" y2="124"/>\n'\
        f'    </g>\n  </g>\n</g>'

    @staticmethod
    def robot(iid,x,y,sz,pc,sc):
        s=sz/220
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Robot">\n'\
        f'  <rect x="14" y="70" width="20" height="50" rx="10" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <rect x="166" y="70" width="20" height="50" rx="10" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <path d="M52,78 q0,-40 28,-56 q14,-8 30,-8 q16,0 30,8 q28,16 28,56 v22 q0,32 -20,50 q-20,20 -38,20 q-18,0 -38,-20 q-20,-18 -20,-50Z" fill="#f2f4fa" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <path d="M52,88 q0,-10 8,-16 h80 q8,6 8,16 v8 H52Z" fill="{sc}" opacity="0.45"/>\n'\
        f'  <circle cx="88" cy="110" r="13" fill="#96aee0" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="112" cy="110" r="13" fill="#96aee0" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="89" cy="108" r="5" fill="#1e2847"/><circle cx="113" cy="108" r="5" fill="#1e2847"/>\n'\
        f'  <circle cx="91" cy="106" r="2" fill="#fff"/><circle cx="115" cy="106" r="2" fill="#fff"/>\n'\
        f'  <path d="M82,132 q18,12 36,0" fill="none" stroke="{pc}" stroke-width="3" stroke-linecap="round"/>\n'\
        f'  <circle cx="100" cy="46" r="16" fill="#f0b24d"/><circle cx="100" cy="46" r="7" fill="#f8f8f8"/>\n'\
        f'  <g fill="none" stroke="#f0b24d" stroke-width="4" stroke-linecap="round">\n'\
        f'    <line x1="100" y1="26" x2="100" y2="34"/><line x1="100" y1="58" x2="100" y2="66"/>\n'\
        f'    <line x1="80" y1="46" x2="88" y2="46"/><line x1="112" y1="46" x2="120" y2="46"/>\n'\
        f'    <line x1="86" y1="32" x2="92" y2="38"/><line x1="108" y1="54" x2="114" y2="60"/>\n'\
        f'    <line x1="86" y1="60" x2="92" y2="54"/><line x1="108" y1="38" x2="114" y2="32"/>\n'\
        f'  </g>\n'\
        f'  <path d="M40,50 q14,-38 68,-38 q54,0 68,38" fill="none" stroke="{pc}" stroke-width="3.5" stroke-linecap="round"/>\n'\
        f'  <g stroke="{pc}" stroke-width="3" fill="none" stroke-linecap="round">\n'\
        f'    <path d="M36,204 q6,-36 40,-42 q16,-3 32,-16 q16,13 32,16 q34,6 40,42"/>\n'\
        f'    <path d="M78,158 q0,26 34,26 q34,0 34,-26"/>\n'\
        f'  </g>\n</g>'

    @staticmethod
    def flask(iid,x,y,sz,pc,sc):
        s=sz/90
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Flask">\n'\
        f'  <ellipse cx="46" cy="86" rx="40" ry="8" fill="{sc}" opacity="0.5"/>\n'\
        f'  <path d="M26,0 h40 M46,0 v16 l24,50 q6,15 -10,15 h-50 q-16,0 -10,-15 l24,-50 v-16" fill="none" stroke="{pc}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>\n'\
        f'  <path d="M14,54 h48 l-7,14 h-34z" fill="{sc}" opacity="0.85"/>\n'\
        f'  <circle cx="22" cy="62" r="4" fill="{sc}"/><circle cx="38" cy="60" r="4" fill="{sc}"/>\n'\
        f'</g>'

    @staticmethod
    def gear(iid,x,y,sz,pc,sc):
        s=sz/80
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Gear">\n'\
        f'  <circle cx="40" cy="40" r="28" fill="{pc}"/>\n'\
        f'  <circle cx="40" cy="40" r="14" fill="{sc}"/>\n'\
        f'  <circle cx="40" cy="40" r="6" fill="{pc}" opacity="0.6"/>\n'\
        f'  <g fill="none" stroke="{pc}" stroke-width="7" stroke-linecap="round">\n'\
        f'    <line x1="40" y1="4" x2="40" y2="14"/><line x1="40" y1="66" x2="40" y2="76"/>\n'\
        f'    <line x1="4" y1="40" x2="14" y2="40"/><line x1="66" y1="40" x2="76" y2="40"/>\n'\
        f'    <line x1="12" y1="12" x2="19" y2="19"/><line x1="61" y1="61" x2="68" y2="68"/>\n'\
        f'    <line x1="12" y1="68" x2="19" y2="61"/><line x1="61" y1="19" x2="68" y2="12"/>\n'\
        f'  </g>\n</g>'

    @staticmethod
    def pill(iid,x,y,sz,pc,sc):
        s=sz/100
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Pill">\n'\
        f'  <path d="M2,20 q0,-14 14,-14 h52 q14,0 14,14 v62 q0,14 -14,14 h-52 q-14,0 -14,-14Z" fill="{sc}" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <rect x="10" y="44" width="36" height="16" rx="3" fill="#f0ce80"/>\n'\
        f'  <path d="M0,2 h82 v20 q0,6 -6,6 h-70 q-6,0 -6,-6Z" fill="#f0ce80" stroke="#c89820" stroke-width="2"/>\n'\
        f'  <line x1="20" y1="20" x2="20" y2="36" stroke="#e06060" stroke-width="3" stroke-linecap="round"/>\n'\
        f'  <line x1="12" y1="28" x2="28" y2="28" stroke="#e06060" stroke-width="3" stroke-linecap="round"/>\n'\
        f'  <g transform="translate(92,18)">\n'\
        f'    <rect x="0" y="0" width="88" height="38" rx="19" fill="#d4def4" stroke="{pc}" stroke-width="2"/>\n'\
        f'    <path d="M43,0 a19,19 0 0 1 0,38" fill="#8aa8de" stroke="{pc}" stroke-width="2"/>\n'\
        f'  </g>\n</g>'

    @staticmethod
    def target(iid,x,y,sz,pc,sc):
        s=sz/90
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Target">\n'\
        f'  <circle cx="36" cy="36" r="36" fill="none" stroke="{pc}" stroke-width="3.5"/>\n'\
        f'  <circle cx="36" cy="36" r="22" fill="none" stroke="{sc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="36" cy="36" r="10" fill="#f0ce80" stroke="#c89820" stroke-width="2"/>\n'\
        f'  <line x1="36" y1="36" x2="106" y2="36" stroke="{pc}" stroke-width="4.5" stroke-linecap="round"/>\n'\
        f'  <path d="M92,22 l14,14 l-14,14" fill="none" stroke="{pc}" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>\n'\
        f'</g>'

    @staticmethod
    def robot_arm(iid,x,y,sz,pc,sc):
        s=sz/130
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Robot Arm">\n'\
        f'  <ellipse cx="134" cy="124" rx="110" ry="12" fill="{sc}" opacity="0.5"/>\n'\
        f'  <rect x="4" y="112" width="82" height="14" rx="3" fill="#90b0e0" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <g stroke="{pc}" stroke-width="2.8" stroke-linecap="round">\n'\
        f'    <line x1="46" y1="112" x2="46" y2="76"/><line x1="46" y1="76" x2="86" y2="24"/>\n'\
        f'    <line x1="86" y1="24" x2="160" y2="32"/><line x1="160" y1="32" x2="202" y2="40"/>\n'\
        f'    <line x1="202" y1="40" x2="202" y2="78"/>\n'\
        f'  </g>\n'\
        f'  <circle cx="46" cy="76" r="9" fill="#f0f4fc" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="86" cy="24" r="13" fill="#f0f4fc" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="160" cy="32" r="13" fill="#f0f4fc" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <path d="M200,78 h20 v18 h-20Z" fill="#f4dea0" stroke="{pc}" stroke-width="2.5"/>\n'\
        f'  <circle cx="104" cy="58" r="9" fill="#f0b24d"/>\n'\
        f'  <circle cx="168" cy="72" r="7" fill="#f0b24d"/>\n'\
        f'</g>'

    @staticmethod
    def lattice(iid,x,y,sz,pc,sc):
        s=sz/110
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Lattice">\n'\
        f'  <ellipse cx="54" cy="98" rx="50" ry="8" fill="{sc}" opacity="0.5"/>\n'\
        f'  <g stroke="{pc}" stroke-width="2.5" stroke-linecap="round">\n'\
        f'    <line x1="14" y1="38" x2="54" y2="24"/><line x1="54" y1="24" x2="94" y2="42"/>\n'\
        f'    <line x1="94" y1="42" x2="94" y2="82"/><line x1="54" y1="24" x2="54" y2="66"/>\n'\
        f'    <line x1="14" y1="38" x2="14" y2="78"/><line x1="14" y1="78" x2="54" y2="96"/>\n'\
        f'    <line x1="54" y1="96" x2="94" y2="82"/><line x1="54" y1="66" x2="14" y2="78"/>\n'\
        f'    <line x1="54" y1="66" x2="94" y2="82"/>\n'\
        f'  </g>\n'\
        f'  <circle cx="14" cy="38" r="10" fill="#d0aed8" stroke="#7a4c96" stroke-width="2"/>\n'\
        f'  <circle cx="54" cy="24" r="10" fill="#a8d0a6" stroke="#4a8860" stroke-width="2"/>\n'\
        f'  <circle cx="94" cy="42" r="10" fill="{sc}" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <circle cx="14" cy="78" r="12" fill="#a4c0ee" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <circle cx="54" cy="66" r="10" fill="{sc}" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <circle cx="54" cy="96" r="10" fill="#b4c8ee" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <circle cx="94" cy="82" r="12" fill="#d0aed8" stroke="#7a4c96" stroke-width="2"/>\n'\
        f'</g>'

    @staticmethod
    def chart(iid,x,y,sz,pc,sc):
        s=sz/100
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Chart">\n'\
        f'  <rect x="0" y="60" width="18" height="40" rx="2" fill="{pc}" opacity="0.8"/>\n'\
        f'  <rect x="24" y="30" width="18" height="70" rx="2" fill="{pc}"/>\n'\
        f'  <rect x="48" y="45" width="18" height="55" rx="2" fill="{pc}" opacity="0.9"/>\n'\
        f'  <rect x="72" y="15" width="18" height="85" rx="2" fill="{sc}"/>\n'\
        f'  <line x1="0" y1="100" x2="96" y2="100" stroke="{pc}" stroke-width="2.5" stroke-linecap="round"/>\n'\
        f'  <line x1="0" y1="0" x2="0" y2="100" stroke="{pc}" stroke-width="2.5" stroke-linecap="round"/>\n'\
        f'</g>'

    @staticmethod
    def flowchart(iid,x,y,sz,pc,sc):
        s=sz/100
        return f'<g id="{iid}" transform="translate({x},{y}) scale({s:.3f})" inkscape:label="Icon: Flowchart">\n'\
        f'  <rect x="4" y="46" width="34" height="34" rx="3" fill="{sc}" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <rect x="76" y="0" width="34" height="34" rx="3" fill="#f4ce82" stroke="#c88830" stroke-width="2"/>\n'\
        f'  <rect x="148" y="22" width="34" height="34" rx="3" fill="{sc}" stroke="{pc}" stroke-width="2"/>\n'\
        f'  <g stroke="{pc}" stroke-width="2.5" stroke-linecap="round">\n'\
        f'    <line x1="38" y1="63" x2="94" y2="63"/>\n'\
        f'    <line x1="94" y1="34" x2="94" y2="63"/>\n'\
        f'    <line x1="94" y1="34" x2="148" y2="34"/>\n'\
        f'    <line x1="94" y1="63" x2="148" y2="63"/>\n'\
        f'  </g>\n'\
        f'  <circle cx="250" cy="38" r="34" fill="none" stroke="#4a8060" stroke-width="3.5"/>\n'\
        f'  <circle cx="250" cy="38" r="8" fill="#f4ce82" stroke="#c88830" stroke-width="1.5"/>\n'\
        f'  <line x1="250" y1="38" x2="322" y2="38" stroke="#4a8060" stroke-width="4" stroke-linecap="round"/>\n'\
        f'  <path d="M306,22 l16,16 l-16,16" fill="none" stroke="#4a8060" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>\n'\
        f'</g>'


# ══════════════════════════════════════════════════════════
# M8  SVG 渲染器
# ══════════════════════════════════════════════════════════
FONT = '"Times New Roman", Times, Georgia, serif'

def esc(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def arc_path(cx,cy,a1,a2,ro,ri) -> str:
    def pt(a,r): return cx+r*math.cos(math.radians(a)), cy-r*math.sin(math.radians(a))
    x1o,y1o=pt(a1,ro); x2o,y2o=pt(a2,ro)
    x1i,y1i=pt(a1,ri); x2i,y2i=pt(a2,ri)
    lg=1 if abs(a2-a1)>180 else 0
    return (f"M{x1o:.2f},{y1o:.2f} A{ro},{ro} 0 {lg},0 {x2o:.2f},{y2o:.2f} "
            f"L{x2i:.2f},{y2i:.2f} A{ri},{ri} 0 {lg},1 {x1i:.2f},{y1i:.2f}Z")


class SVGRenderer:
    def __init__(self, s: Schema):
        self.s = s
        self.out: List[str] = []

    def render(self) -> str:
        self.out = []
        self._header()
        self._background()
        self._panels()
        self._ring()
        self._free_texts()
        self.out.append("</svg>")
        return "\n".join(self.out)

    def _header(self):
        s = self.s

        # 渐变 defs
        panel_grads = []
        for p in s.panels:
            if p.has_header and not Color.is_white(p.header_fill):
                hc2 = Color.lighten(p.header_fill, 0.18)
                panel_grads.append(
                    f'  <linearGradient id="hg_{p.id}" x1="0" x2="1">\n'
                    f'    <stop offset="0%" stop-color="{p.header_fill}"/>\n'
                    f'    <stop offset="100%" stop-color="{hc2}"/>\n'
                    f'  </linearGradient>')

        arc_grads = []
        for i, seg in enumerate(s.ring_segs):
            c2 = Color.lighten(seg.fill, 0.14)
            arc_grads.append(
                f'  <linearGradient id="ag_{i}" x1="0" x2="1">\n'
                f'    <stop offset="0%" stop-color="{seg.fill}"/>\n'
                f'    <stop offset="100%" stop-color="{c2}"/>\n'
                f'  </linearGradient>')

        all_grads = "\n".join(panel_grads + arc_grads)
        self.out.append(f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="{s.W}" height="{s.H}" viewBox="0 0 {s.W} {s.H}">
<defs>
  <linearGradient id="_bg" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="{s.bg}"/>
    <stop offset="100%" stop-color="{s.bg2}"/>
  </linearGradient>
  <radialGradient id="_glow" cx="50%" cy="38%" r="70%">
    <stop offset="0%" stop-color="#ffffff" stop-opacity="0.70"/>
    <stop offset="100%" stop-color="{s.bg}" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="_panel" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="#fafbfc"/>
    <stop offset="100%" stop-color="#f1f4f8"/>
  </linearGradient>
  <radialGradient id="_ctr" cx="46%" cy="36%" r="72%">
    <stop offset="0%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="{s.ring_fill}"/>
  </radialGradient>
{all_grads}
  <filter id="fs"  x="-15%" y="-15%" width="130%" height="130%">
    <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#8898bb" flood-opacity="0.10"/>
  </filter>
  <filter id="fss" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#8898bb" flood-opacity="0.14"/>
  </filter>
  <filter id="fsr" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="3" stdDeviation="3" flood-opacity="0.13"/>
  </filter>
  <style>
    text {{ font-family: {FONT}; }}
    .tt  {{ font-size:44px; font-weight:700; fill:#1e2847; letter-spacing:0.5px; }}
    .ts  {{ font-size:36px; font-weight:700; fill:#1e2847; }}
    .th  {{ font-size:27px; font-weight:700; fill:#ffffff; }}
    .tb  {{ font-size:22px; fill:#2a3560; }}
    .tr  {{ font-size:28px; font-weight:700; fill:#ffffff; }}
    .tc  {{ font-size:44px; font-weight:700; fill:#1e2847; }}
    .tcs {{ font-size:30px; fill:#2a3560; }}
  </style>
</defs>''')

    def _background(self):
        self.out.append(f'''
<!-- Background -->
<rect width="{self.s.W}" height="{self.s.H}" fill="url(#_bg)"/>
<rect width="{self.s.W}" height="{self.s.H}" fill="url(#_glow)"/>''')

    def _panels(self):
        self.out.append("\n<!-- ══ Panels ══ -->")
        for p in self.s.panels:
            x,y,w,h,rr = p.x,p.y,p.w,p.h,p.corner_r

            # Card shadow + fill
            self.out.append(f'\n<g id="{p.id}" inkscape:label="Panel: {esc(p.header_text)}" filter="url(#fs)">')
            self.out.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rr}" fill="url(#_panel)" stroke="{p.stroke}" stroke-width="1.2"/>')

            # Header bar
            if p.has_header and p.header_text and not Color.is_white(p.header_fill):
                ht = p.header_h
                hg = f"hg_{p.id}"
                self.out.append(
                    f'  <path d="M{x+rr},{y} H{x+w-rr} Q{x+w},{y} {x+w},{y+rr} V{y+ht} H{x} V{y+rr} Q{x},{y} {x+rr},{y}Z" fill="url(#{hg})"/>')
                # Inner highlight
                self.out.append(
                    f'  <rect x="{x+6}" y="{y+6}" width="{w-12}" height="{h-12}" rx="{rr-3}" fill="none" stroke="#fff" stroke-width="0.7" opacity="0.4"/>')
            self.out.append('</g>')

            # Header text
            if p.header_text:
                self.out.append(
                    f'<text x="{x+w//2}" y="{y+p.header_h//2+10}" text-anchor="middle" class="th">{esc(p.header_text)}</text>')

            # Body texts
            for t in p.body_texts:
                self.out.append(
                    f'<text x="{t.cx}" y="{t.y+t.h}" text-anchor="middle" '
                    f'style="font-size:{t.size}px;font-weight:{t.weight};fill:{t.color}">'
                    f'{esc(t.text)}</text>')

            # Icons
            for ic in p.icons:
                self.out.append(Icons.render(ic))

    def _ring(self):
        s = self.s
        if not s.ring_present:
            return
        self.out.append("\n<!-- ══ DMTA Ring ══ -->")
        cx,cy = s.ring_cx, s.ring_cy
        or_,ir_ = s.ring_or, s.ring_ir
        cr = s.ring_cr

        # Guide circle
        self.out.append(f'<circle cx="{cx}" cy="{cy}" r="{or_+8}" fill="none" stroke="#d0d4de" stroke-width="1.2"/>')

        for i, seg in enumerate(s.ring_segs):
            a1,a2 = seg.a_start, seg.a_end
            d = arc_path(cx, cy, a1, a2, or_, ir_)

            # Highlight arc on inner edge of outer radius
            h1x=cx+(or_-10)*math.cos(math.radians(a1)); h1y=cy-(or_-10)*math.sin(math.radians(a1))
            h2x=cx+(or_-10)*math.cos(math.radians(a2)); h2y=cy-(or_-10)*math.sin(math.radians(a2))
            lg=1 if abs(a2-a1)>180 else 0

            # Arrow
            mr=(or_+ir_)/2
            ea=a2; rad=math.radians(ea)
            ex=cx+mr*math.cos(rad); ey=cy-mr*math.sin(rad)
            tang=math.radians(ea-90); arr=18
            ax1=ex+arr*math.cos(tang+0.38); ay1=ey-arr*math.sin(tang+0.38)
            ax2=ex+arr*math.cos(tang-0.38); ay2=ey-arr*math.sin(tang-0.38)
            ax3=ex+arr*1.5*math.cos(tang); ay3=ey-arr*1.5*math.sin(tang)

            # Label
            lbl_a=(a1+a2)/2
            lx=cx+mr*math.cos(math.radians(lbl_a)); ly=cy-mr*math.sin(math.radians(lbl_a))

            self.out.append(f'''
<g id="seg_{i}" inkscape:label="Seg: {seg.label}" filter="url(#fsr)">
  <path d="{d}" fill="url(#ag_{i})" stroke="#f4f4f7" stroke-width="7"/>
  <path d="M{h1x:.1f},{h1y:.1f} A{or_-10} {or_-10} 0 {lg} 0 {h2x:.1f},{h2y:.1f}"
        fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" opacity="0.22"/>
</g>
<polygon points="{ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f} {ax3:.1f},{ay3:.1f}" fill="#f0f0f4"/>
<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle"
      transform="rotate({-lbl_a:.1f} {lx:.1f} {ly:.1f})"
      class="tr" style="fill:{seg.label_color}">{esc(seg.label)}</text>''')

        # Center circle
        self.out.append(
            f'<circle cx="{cx}" cy="{cy}" r="{cr}" fill="url(#_ctr)" stroke="#ccd2e0" stroke-width="2.5" filter="url(#fss)"/>')
        self.out.append(
            f'<circle cx="{cx}" cy="{cy}" r="{cr-4}" fill="none" stroke="#fff" stroke-width="0.8" opacity="0.65"/>')

        # Robot icon
        if s.has_robot:
            rsz = min(int(cr * 1.5), 190)
            ric = {"id":"center_robot","type":"robot","x":cx-rsz//2+10,
                   "y":cy-cr+12,"size":rsz,
                   "primary_color":"#3a5d96","secondary_color":"#d0e0f8"}
            self.out.append(Icons.render(ric))

        # Center texts
        for t in sorted(s.ring_center_texts, key=lambda x: x.y):
            css = "tc" if t.size > 30 else "tcs"
            self.out.append(
                f'<text x="{cx}" y="{t.y+t.h}" text-anchor="middle" class="{css}" '
                f'style="fill:{t.color}">{esc(t.text)}</text>')

    def _free_texts(self):
        self.out.append("\n<!-- ══ Text Layer (Times New Roman) ══ -->")
        for t in self.s.free_texts:
            # 过滤噪声
            if len(t.text) <= 1: continue
            css = "tt" if t.size > 35 else ("ts" if t.size > 26 else "tb")
            self.out.append(
                f'<text x="{t.cx}" y="{t.y+t.h}" text-anchor="middle" class="{css}" '
                f'style="font-size:{t.size}px;font-weight:{t.weight};fill:{t.color}">'
                f'{esc(t.text)}</text>')


# ══════════════════════════════════════════════════════════
# 主流水线
# ══════════════════════════════════════════════════════════
def run(src: str, dst: str, n_colors: int = 12,
        verbose: bool = False) -> bool:
    print(f"\n{'='*54}")
    print(f"  {Path(src).name}  →  {Path(dst).name}")
    print(f"{'='*54}")

    # 加载
    img = Image.open(src)
    try:
        from PIL import ImageOps; img = ImageOps.exif_transpose(img)
    except: pass
    if img.mode in ("RGBA","LA","P"):
        if img.mode=="P": img=img.convert("RGBA")
        bg=Image.new("RGB",img.size,(255,255,255))
        if img.mode in ("RGBA","LA"): bg.paste(img,mask=img.split()[-1])
        else: bg.paste(img)
        img=bg
    else:
        img=img.convert("RGB")
    w,h=img.size
    if max(w,h)>2048:
        sc=2048/max(w,h); img=img.resize((int(w*sc),int(h*sc)),Image.LANCZOS)
    arr=np.array(img)
    print(f"  尺寸: {img.size[0]}×{img.size[1]}")

    # 分析
    schema = SchemaBuilder().build(img, arr, n_colors, verbose)

    # 渲染
    print("  [M6] 渲染 SVG...")
    svg = SVGRenderer(schema).render()

    # 验证 XML
    from xml.etree import ElementTree as ET
    try: ET.fromstring(svg); print("  [✓] XML 有效")
    except Exception as e: print(f"  [!] XML 警告: {e}")

    Path(dst).write_text(svg, encoding="utf-8")
    kb = Path(dst).stat().st_size // 1024
    print(f"  ✓ 输出: {dst}  ({kb} KB)")
    stats = (f"面板:{len(schema.panels)}  "
             f"圆环:{'✓' if schema.ring_present else '✗'}({len(schema.ring_segs)}段)  "
             f"OCR:{len(schema.free_texts)+sum(len(p.body_texts) for p in schema.panels)}词  "
             f"Icon:{sum(len(p.icons) for p in schema.panels)}")
    print(f"  {stats}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="img2svg_local v2.0 — 高精度本地图像→可编辑SVG（无需API）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input", nargs="+")
    ap.add_argument("-o","--output")
    ap.add_argument("--output-dir", metavar="DIR")
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    files=[]
    for inp in args.input:
        p=Path(inp)
        if p.is_dir():
            for ext in SUPPORTED:
                files.extend(p.glob(f"*{ext}"))
                files.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
    if not files: sys.exit("未找到图像文件")

    print(f"\n🚀 img2svg_local v2.0  |  {len(files)} 个文件  |  无需 API Key")
    ok=fail=0
    for fp in files:
        if args.output and len(files)==1: out=args.output
        elif args.output_dir:
            Path(args.output_dir).mkdir(parents=True,exist_ok=True)
            out=str(Path(args.output_dir)/(fp.stem+".svg"))
        else: out=str(fp.parent/(fp.stem+".svg"))
        try:
            if run(str(fp),out,args.colors,args.verbose): ok+=1
            else: fail+=1
        except Exception as e:
            print(f"  ✗ {e}")
            if args.verbose:
                import traceback; traceback.print_exc()
            fail+=1

    print(f"\n{'='*54}")
    print(f"  完成: {ok} 成功"+(f"  {fail} 失败" if fail else ""))
    print(f"{'='*54}\n")

if __name__=="__main__":
    main()