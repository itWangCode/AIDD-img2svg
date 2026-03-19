# **VectorTracer：一种集成颜色量化、OCR 识别与 Potrace 的高保真位图矢量化方法**

## **研究方法**

本研究提出了一种名为 VectorTracer 的高精度位图矢量化方法，旨在将 JPG/PNG 等栅格图像转换为可编辑的分层矢量图形（SVG/EPS）。方法采用模块化设计，由图像预处理、颜色分析、OCR 文字识别、矢量描摹、SVG 组装及 EPS 导出六个核心模块构成，各模块协同工作以实现高质量、可编辑的矢量化输出。

**图像预处理模块**
首先利用 PIL 库加载图像，自动校正 EXIF 方向，并将 RGBA/LA 模式转换为 RGB 背景。采用 Lanczos 重采样对图像进行自适应缩放，确保最长边介于 600 至 2048 像素之间，同时应用 Unsharp Mask 锐化滤波增强边缘细节，为后续处理奠定基础。

**颜色分析模块**
为支持彩色分层描摹，引入颜色聚类算法。基于像素 RGB 值，采用 K-Means 聚类（支持 scikit-learn 加速或原生 NumPy 实现）将图像划分为指定数量的颜色层。通过迭代优化聚类中心，生成各颜色层的二值掩膜，并剔除面积占比过小的层。随后利用 OpenCV 形态学闭运算与开运算对掩膜进行去噪和平滑，消除孤立噪点并填补细小空洞。

**OCR 文字识别模块**
针对图像中包含的文字区域，集成 Tesseract OCR 引擎进行检测与识别。识别结果包括文本内容、位置坐标、置信度及尺寸信息。通过采样文字区域内的像素亮度，确定适宜的文本颜色，确保最终矢量文字与原始图像视觉一致。

**矢量描摹模块**
采用 Potrace 开源描摹引擎作为贝塞尔曲线生成核心。将每个颜色层的二值掩膜转换为 PBM 格式，调用 Potrace 生成对应的 SVG 路径，并通过自定义参数（如 turdsize、alphamax、opttolerance）控制路径平滑度与细节保留程度。描摹结果包括路径数据及必要的坐标变换信息。

**SVG 组装模块**
将各颜色层的路径与 OCR 文字整合为结构化的 SVG 文档。每个颜色层封装为独立的 `<g>` 元素，并添加 `inkscape:groupmode="layer"` 属性，使图层在 Inkscape 等矢量编辑软件中可单独移动、隐藏或编辑。文字区域则转换为 `<text>` 节点，字体设定为 Times New Roman，确保文字内容可编辑且风格统一。所有路径均继承所在图层的填充颜色，无需重复定义。

**EPS 导出模块**
为满足出版与印刷需求，将矢量图形导出为 EPS 格式。通过解析 SVG 路径命令，将其转换为 PostScript 路径绘制指令，并嵌入颜色定义与文字渲染代码。EPS 文件包含 BoundingBox、字体引用等元数据，兼容 Adobe Illustrator 等专业软件。

**批量处理能力**
方法支持单文件及批量文件夹处理，用户可通过命令行参数灵活控制输出格式、颜色层数、OCR 开关及描摹参数，适用于学术插图、技术图纸及自然图像的矢量化任务。

## 使用说明

### 安装依赖（一次性）

```bash
pip install Pillow numpy scikit-image
# Linux:
sudo apt install potrace
# macOS:
brew install potrace
# Windows: 下载 https://potrace.sourceforge.net/
```

### 使用命令

| 场景                   | 命令                                                     |
| ---------------------- | -------------------------------------------------------- |
| 单张图 → SVG + EPS     | `python bitmap_to_vector.py logo.png`                    |
| 彩色图标（多色分层）   | `python bitmap_to_vector.py icon.png --color`            |
| 仅输出 SVG             | `python bitmap_to_vector.py img.jpg --format svg`        |
| 批量处理整个文件夹     | `python bitmap_to_vector.py ./images/ --color`           |
| 精细描摹（更多颜色层） | `python bitmap_to_vector.py img.png --color --colors 16` |
| 线条/黑白图标          | `python bitmap_to_vector.py icon.png --threshold 100`    |

### 核心技术原理

```
位图像素
   ↓
① 预处理：上采样 2× + USM 锐化边缘
   ↓
② 颜色量化：K-Means 聚类 → N 个颜色层（彩色模式）
   ↓
③ 每层二值化 → PBM 格式
   ↓
④ Potrace 算法：
   像素轮廓提取 → 多边形路径 → 贝塞尔曲线拟合（最小二乘）
   ↓
⑤ 合并所有色层 → SVG（每层独立 <g> 组）+ EPS
   ↓
Adobe Illustrator 中每个色层可单独选中/编辑路径节点
```

**在 Illustrator 中打开后**，每个颜色区域是独立的可编辑路径组，支持修改颜色、调整锚点、缩放无损。



## v3.0 完整升级说明

### 本次根本性改进

**v1/v2 颜色全黑/白的真正原因 + v3 修复：**

| 问题             | v1/v2                                          | v3 修复                    |
| ---------------- | ---------------------------------------------- | -------------------------- |
| potrace SVG 结构 | 只提取 `<path>`，丢失 `<g transform>`          | 完整解析并保留坐标系       |
| 颜色替换位置     | 替换 `<path fill=…>`（potrace 不在这里放颜色） | 正确写在 `<g fill="…">` 上 |
| numpy 类型       | `np.uint8` hex格式化异常                       | 全部 `int()` 转换          |

### 新增核心功能

**① 文字变为真实可编辑节点**

```
OCR 识别 37 个文字区域 → <text> 节点
font-family: "Times New Roman", Times, serif
颜色自动从原图采样（不是猜的）
```

**② 16 色层分离，可整体移动**

```
每层 = <g inkscape:groupmode="layer">
整个色层在 AI 里可一次性选中 → 移动/变色
3788 条贝塞尔路径，精确复刻原图形状
```

**③ 推荐使用参数（针对这张图）**

```bash
# 最精细（20色层）
python bitmap_to_vector.py Fig1.png --color --colors 20 --ocr

# 标准质量（当前配置）
python bitmap_to_vector.py Fig1.png --color --colors 16 --ocr

# 快速预览
python bitmap_to_vector.py Fig1.png --color --colors 8
```
