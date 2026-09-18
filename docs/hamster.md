# HAMSTER 仿真复现可行性

核查日期：2026-09-18。本页区分高层 VLM 推理与完整仿真执行。

## 论文中的实验

[HAMSTER 论文](https://hamster-robot.github.io/paper.pdf) §5.2、Table 3、Appendix F 使用基于 RLBench 的 Colosseum。高层 VLM 预测二维末端路径，路径画在输入图像上，经过专门训练的底层 3D Diffuser Actor 输出机器人动作。

- 使用前视相机，评测 20 个任务中的 14 个。
- 每任务 100 条无视觉扰动训练示范；每种变化 25 个评测 episode。
- Table 3 五个随机种子的平均成功率：普通 3D-DA 为 0.35 ± 0.04，HAMSTER+3D-DA 为 0.46 ± 0.04。
- 上述是论文结果，并非本仓库运行结果。

## 已发布内容与缺口

[项目主页](https://hamster-robot.github.io/) 的 Code 链接指向 [liyi14/HAMSTER_beta](https://github.com/liyi14/HAMSTER_beta)。当前代码主要是高层 VLM 服务和 Gradio 客户端，权重为 [yili18/Hamster_dev](https://huggingface.co/yili18/Hamster_dev)，基于 VILA1.5-13B。

本次核查未找到完整的 HAMSTER 底层策略训练/评测代码或配套底层权重。[Issue #1](https://github.com/liyi14/HAMSTER_beta/issues/1) 也在请求底层策略代码。普通 3D-DA 权重不能直接视为经过路径条件训练的 HAMSTER 权重。

官方代码已下载到 `external/HAMSTER_beta`，外部代码不纳入本仓库提交。复现下载：

```bash
mkdir -p external
git clone https://github.com/liyi14/HAMSTER_beta.git external/HAMSTER_beta
git -C external/HAMSTER_beta checkout 526a37f59f97c445005fcdf28f2cfb81ea742e4b
```

模型仓库 revision：`794f1f925c87e861d2f562943e978cc11f8c344d`。

上游 README 的 VILA commit 存在不一致：依赖简介写 `da98f3b`，实际安装步骤写 `a5a380d6d09762d6f3fd0443aac6b475fba84f7e`。部署时需要隔离环境并验证依赖，不能直接覆盖现有 Qwen 环境。

## 可进行的试验

1. 高层轨迹测试：部署已发布 VLM，用已有杯子图和仿真截图预测点及夹爪开合，与 Qwen 输出比较。这个试验不需要底层权重，但无法证明闭环抓取成功。
2. 环境测试：安装 [Colosseum](https://github.com/robot-colosseum/robot-colosseum)，运行专家示范，导出前视图和真实末端投影轨迹，作为预测参照。专家执行结果不能当作 HAMSTER 执行结果。
3. 完整闭环：取得作者的路径条件底层策略，或另行实现并训练这部分，然后测量实际任务成功率。

仿真环境的[官方安装指南](https://robot-colosseum.readthedocs.io/en/latest/installation.html)依赖 CoppeliaSim 4.1、PyRep 和兼容版本的 RLBench。远程机器为 RTX PRO 6000 Blackwell 96GB；已实际完成下述高层 VLM 推理，峰值分配显存约 26.18GiB。尚未安装或运行仿真环境，也未验证其无显示器渲染。

## 高层 VLM 独立运行

```bash
bash scripts/setup_hamster.sh
PYTHONPATH="$PWD/external/VILA" .venv-hamster/bin/python -m hamster.run_inference \
  --image examples/long_task01_ep0001.png \
  --task 'Grasp the gray cup in the foreground on the table, then lift it off the table.' \
  --output-dir outputs/hamster_cup_lift_fp16
```

每次使用新输出目录。需要 `uv`、Git 和 NVIDIA GPU。下载路径默认为 `/workspace/models/Hamster_dev`；自定义路径可运行 `hamster/download.py --output ...`，推理时传 `--model-root ...`。

运行使用独立 `.venv-hamster`、PyTorch 2.11.0 CUDA 12.8 和 Transformers 4.37.2。与上游环境的差别：升级 PyTorch 以支持 Blackwell，使用原生 SDPA；将未使用的 Intern/Radio 视觉编码器和 DeepSpeed 训练组件改为延迟导入，补丁由 `hamster/patch_vila.py` 可重复应用。保留 HAMSTER 实际使用的 SigLIP、投影器和语言模型计算路径；不覆盖 Transformers 的训练专用文件。权重按上游 loader 转为 FP16，无量化。

`hamster/prompt.txt` 与官方 Gradio 的任务提示模板逐字核对一致，其中包含作者提供的示范坐标。使用 `vicuna_v1` 聊天格式并补上空 assistant 回复前缀；传 `--upstream-user-only` 可重现官方服务未补此前缀的格式。第一轮按官方服务原样运行时输出缺少列表括号及 `<ans>` 标签，失败记录保留，不人为补齐。采样为 greedy、seed 42、最大生成 1024 tokens、启用 KV cache。此处与 Qwen 的无示范坐标 JSON 提示不同，不能视为严格控制变量比较。

输出保留原始回复、token IDs、完整提示词、模型与代码 revision、输入图像 hash、推理耗时和显存。使用 `ast.literal_eval` 解析坐标及动作标签，不执行模型生成的 Python。开合事件绑定到前一轨迹点，不补初始夹爪状态或阶段。坐标按 `x*(W-1), y*(H-1)` 转为像素，与本仓库 Qwen 绘图一致；上游 Gradio 使用 `x*W, y*H`。不平滑、不补点、不人工调整。

重新解析绘图（不运行模型）：

```bash
.venv-hamster/bin/python -c 'from hamster.trajectory import postprocess; postprocess("outputs/hamster_cup_lift_fp16")'
```

远程长任务可使用 `scripts/icra-hamster-setup.conf` 和 `scripts/icra-hamster-inference.conf` 的 Supervisor 配置。日志保存在忽略的 `outputs/` 中。

## 本次杯子图片结果

- [标准 assistant 前缀结果](../results/hamster_cup_lift_fp16/)：4 个点，闭爪发生在第 0 点，开爪发生在第 3 点。原始坐标完整保留。生成耗时约 1.86 秒，峰值分配显存 26.18GiB。
- [官方 user-only 格式结果](../results/hamster_cup_lift_fp16_upstream/)：模型生成了数字和动作，但缺少列表与答案标签，格式验证失败，没有绘图。生成耗时约 2.85 秒。
- 两次运行都使用同一张输入图、同一任务、同一权重和 greedy 设置，唯一实验参数变化为 assistant 前缀。第一次失败由第二轮修复格式，但不代表任务计划已正确。
- 第二轮起点位于杯口内部附近，没有表达夹爪从当前位置接近杯子的过程；轨迹上移后向左，并在末尾松爪。不能认定满足抓住并抬起保持的任务。
- 仅进行了上述两次运行；没有根据结果修改坐标、补齐缺失格式或选择多次采样中最好的一条。单样本没有任务成功率或统计结论。
- `output_tokens` 是 VILA 返回序列长度，包含其返回的 BOS/EOS；耗时仅计生成阶段，不含模型加载，不作为跨模型速度基准。

启动现有结果服务后，访问 `/comparison.html` 可并排查看 HAMSTER 与已有 Qwen 两次结果。
