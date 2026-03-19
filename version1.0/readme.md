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
