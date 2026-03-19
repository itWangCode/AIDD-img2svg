## 全自动 AI 驱动流水线

### 立即使用（3步）

```bash
# 1. 安装依赖
pip install anthropic Pillow numpy

# 2. 设置 API Key
export ANTHROPIC_API_KEY="sk-ant-api03-你的key"

# 3. 运行
python img2svg.py Fig1.png
python img2svg.py Fig1.png --output my_output.svg
python img2svg.py ./images/  --output-dir ./vectors/   # 批量
```

### 完整自动化流水线架构

```
输入图片
   ↓
[S1] ImageIngestor
   - 加载 + EXIF修正 + 透明→白底
   - 像素精确颜色采样
   - base64编码 → API就绪
   ↓
[S2] VisionAnalyzer  ← Claude Vision API (claude-opus-4-5)
   - 深度理解图像：布局/颜色/文字/图标/几何
   - 返回结构化 JSON schema
   - 自动用真实像素颜色覆盖AI猜测
   ↓
[S3] SchemaValidator
   - 校验所有字段，补全缺失值
   - 坐标范围保护
   ↓
[S4] SVGRenderer
   - 所有文字 → <text font-family="Times New Roman">
   - 每个 icon → 独立 <g id="icon_xxx">（可整体移动）
   - 面板/圆环/箭头 → 精确几何 SVG 元素
   ↓
[S5] PostProcessor
   - XML 有效性验证
   - 保存 .svg
```

