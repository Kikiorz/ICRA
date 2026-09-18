# 单张杯子图片 → VGGT 彩色点云

2026-09-18 已在 RTX PRO 6000 Blackwell 96GB 上实际运行，生成 268,324 个彩色点。原始输入为 `examples/long_task01_ep0001.png`，256×256。官方预处理将其放大为 518×518，没有裁剪；点数对应放大后的像素数，不代表获得了更多真实观测。

- [交互页面](../pointcloud.html)：拖动旋转、滚轮缩放、右键平移；支持原始相机、斜视、侧视、背面视角。
- 在左侧图像拖动框选，可只查看对应像素的三维点。框选是图像区域筛选，并非语义分割。
- 置信度滑块按分数分位数过滤点，默认 0%，显示全部有效点。模型分数不是正确概率。
- [原始结果](../results/vggt_cup_single/)包括 PLY、原始预测、相机参数、输入、深度预览和页面截图。

使用已有的结果服务时访问 `http://localhost:8080/pointcloud.html`；服务端监听 `127.0.0.1:18080`，通过既有 SSH 转发访问。

## 模型与几何处理

[VGGT 官方代码](https://github.com/facebookresearch/vggt)固定在 `a288dd0f14786c93483e45524328726ab7b1b4ce`；[facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)权重固定在 `860abec7937da0a4c03c41d3c269c366e82abdf9`。外部代码位于忽略的 `external/vggt`，权重位于服务器 `/workspace/models/VGGT-1B`。

使用 PyTorch 2.11.0 CUDA 12.8，FP32 权重、BF16 autocast、eval 模式和 seed 42。原始上游依赖指定的较旧 PyTorch 没有覆盖安装。当前机器复用独立 HAMSTER 环境中已经安装的公共依赖；VGGT 不调用 HAMSTER 或 Qwen。

模型预测相机内参/外参、深度及置信分数、直接三维点及置信分数。本次展示的点云采用官方演示默认做法：使用预测相机参数反投影预测深度。模型直接输出的点图也保留在 NPZ 中，不与反投影结果混合。

PLY 保留 VGGT 世界坐标，浮点坐标使用标准 float32 存储。原始反投影数组为 float64，在 NPZ 中保留；PLY 与其 float32 转换完全一致，最大舍入误差约 5.96e-8（模型单位）。RGB 来源为实际预处理后的图像；额外属性保存置信分数及像素 u/v 索引。没有人工修改形状、平滑、网格化、填洞或补全背面。

页面使用预测外参把点转换到输入相机坐标系，再用 `(x, -y, -z)` 转为 Three.js 的显示约定。相机外参约定为 OpenCV camera-from-world。显示变换不改变下载的 PLY。深度 PNG 仅用于预览，以第 2–98 百分位拉伸颜色；NPZ 中的深度未经归一化。

## 本次观察与限制

杯子的可见杯口和杯身、机械臂及桌面形成了可旋转查看的三维结构。斜视和侧视可见遮挡区域的缺面，以及轮廓附近拉伸和稀疏点。原图框选的杯子区域仍包含该矩形内的桌面等背景。

这是单图估计的可见表面，不是完整物体模型。没有真实深度、真实相机标定或三维真实值；绝对尺度未经校准，不能把输出坐标当作米，也没有到机器人基座坐标系的变换。

生成阶段实测约 0.201 秒，峰值 PyTorch allocated 显存约 5.32GiB；不含权重下载、加载、导出和浏览器渲染，不是吞吐率基准。

## 文件

| 文件 | 内容 |
|---|---|
| `input.png` / `model_input.png` | 原始图像 / 实际送入模型的缩放图像 |
| `predictions.npz` | 相机编码、深度、直接点图、置信度、反投影点、RGB、内外参 |
| `pointcloud.ply` | 完整有效彩色点云，可由通用点云工具打开 |
| `camera.json` | 预测内参、外参和图像尺寸 |
| `depth_preview.png` | 深度颜色预览 |
| `metadata.json` | 来源、版本、尺度说明、过滤规则、耗时和显存 |
| `validation.json` | 导出与重投影一致性检查，不是几何准确性评估 |
| `viewer_oblique.png` / `viewer_cup_region.png` | 浏览器实际渲染截图 |

## 复现

本机已有环境可直接执行：

```bash
ICRA_VGGT_OUTPUT=outputs/vggt_cup_single_repeat bash scripts/run_vggt.sh
```

新环境安装最小依赖后，指定 Python 路径（无需下载 HAMSTER）：

```bash
uv venv --python 3.12 .venv-geometry
uv pip install --python .venv-geometry/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-geometry/bin/python -r geometry/requirements.txt
ICRA_GEOMETRY_PYTHON=.venv-geometry/bin/python ICRA_VGGT_OUTPUT=outputs/vggt_cup_single_new bash scripts/run_vggt.sh
```

脚本会拒绝覆盖已有输出目录。当前入口为单图实验；其他图片可传 `--image ...`。权重下载目录可传 `--model-dir ...`。

检查已保存的结果，无需 GPU：

```bash
python -m geometry.validate results/vggt_cup_single
```

检查了所有 PLY 坐标、颜色、分数与 NPZ 对应关系。使用保存的相机参数投影回图像，最大数值误差为 0.000122 像素；这只是导出及变换的自洽性，不证明预测几何真实准确。浏览器检查覆盖加载、视角切换、框选、置信度过滤和恢复全点，页面无 JavaScript 异常。

Three.js 0.180.0 文件及 MIT 许可保存在 `vendor/three`，交互页面无需访问远程 CDN。
