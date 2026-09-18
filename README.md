# ICRA

单张图像与任务描述 → Qwen3.8-27B → 夹爪末端二维轨迹 JSON → 原图坐标叠加。

本实验输入为 `examples/long_task01_ep0001.png`，任务为“抓住桌面前景的灰色杯子，然后把它抬离桌面”。模型预测夹爪两指之间的中心点（TCP），不是杯子中心或腕部外壳。

已完成 BF16 思考/非思考两次实测，原始输出与轨迹图见 **[首次实验结果](results/README.md)**。流程可运行，但本样例存在末端定位与抓取点偏差；保留失败，不人工美化坐标。

## 实验约定

- 模型：`Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
- BF16，无量化；单张图片，单请求。使用 Transformers，SDPA attention。
- 提示词没有示范坐标。原图不添加人工框或标记后再送入模型。
- 默认启用 thinking，`reasoning_effort=medium`，greedy decoding（`do_sample=False`），seed 42。这是实验设置，不是官方采样推荐参数。
- 模型生成 8–12 个点，坐标为 `[0,1000]`，原点左上。转换为原图像素使用 `x*(W-1)/1000, y*(H-1)/1000`。
- 阶段为 start → approach → grasp → lift，包含对应夹爪状态。
- 保存原始回复和模型元数据，不补点、不截断越界点、不手工修正轨迹。致命坐标或结构错误不绘制；夹爪状态与提示词约定不一致时，记录严格校验失败，仍可展示未修改的轨迹。
- Pillow 只连接相邻模型点，不进行轨迹插值或平滑。`overlay.png` 与输入同尺寸；`trajectory.png` 是四倍最近邻放大和图例。
- 结果仅为模型的二维运动假设，未经深度、运动学、碰撞或抓取验证，不能直接作为机器人命令。

## 环境与运行

Python 3.12。Blackwell 显卡需使用支持其架构的 PyTorch，例如 CUDA 12.8 wheel。模型权重约 55.6GB，另需推理显存；本实验使用 96GB GPU。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_model.py --output /workspace/models/Qwen3.8-27B
.venv/bin/python run_inference.py \
  --model /workspace/models/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --image examples/long_task01_ep0001.png \
  --output-dir outputs/cup_lift_bf16
```

每次使用新的输出目录，避免覆盖已有实验。`--no-thinking` 可做非思考模式对照；`--task` 可更换任务。

## 结果文件

| 文件 | 内容 |
|---|---|
| `input.png` | 送入模型的图像 |
| `prompt.txt` / `formatted_prompt.txt` | 任务提示词和包含图像占位符的聊天模板 |
| `raw_response.txt` / `response.txt` | 原始 token 解码及去掉特殊 token 的文本 |
| `metadata.json` | 模型版本、输入 SHA256、生成设置、软件版本、耗时和显存 |
| `trajectory.json` | 经过格式校验的模型 JSON，坐标不修改 |
| `pixel_waypoints.json` | 原始坐标及其像素换算 |
| `overlay.png` / `trajectory.png` | 原尺寸叠加图 / 放大展示图 |
| `validation_error.txt` | 仅失败时产生 |
| `schema_validation.json` | 是否可绘制、是否严格遵循提示词；格式通过不代表物理正确 |

`outputs/` 默认忽略；选定的真实实验结果保存到 `results/`。模型权重、环境、凭据不入库。

重新绘图：

```bash
.venv/bin/python render.py --image examples/long_task01_ep0001.png \
  --trajectory results/cup_lift_bf16/trajectory.json --output-dir outputs/redraw
.venv/bin/python -m unittest discover -s tests -v
```

对保留的原始回复重新校验和绘图：`.venv/bin/python postprocess.py results/cup_lift_bf16`。不会调用模型或修改坐标。第一次推理曾因 `grasp/closed` 与提示词的 `grasp/close` 约定不同而被严格校验拦下；后处理显式区分“可绘制”与“严格符合提示词”，保留原错误记录。

## 查看逐点轨迹

服务器端运行 `bash scripts/serve_results.sh`，或安装 `scripts/icra-viewer.conf` 到 Supervisor 配置目录后执行 `supervisorctl reread && supervisorctl update`。服务仅监听 `127.0.0.1:18080`，避开实例的 Jupyter 端口。使用 SSH `-L 8080:localhost:18080` 转发后，在本地打开 `http://localhost:8080/viewer.html`。

页面默认读取 `results/cup_lift_bf16/`。可用 `?run=results/另一个实验目录` 选择其他结果。播放只表示点的顺序，不代表机械臂执行速度。

## 来源

- [Qwen3.8-27B 官方模型](https://huggingface.co/Qwen/Qwen3.8-27B)
- [固定版本配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)

原始任务图片由仓库所有者提供。
