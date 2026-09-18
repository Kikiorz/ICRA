# HAMSTER 仿真复现可行性

核查日期：2026-09-18。当前仅完成资料核查和官方代码下载，尚未运行 HAMSTER 推理或仿真。

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

核查的模型仓库 revision：`794f1f925c87e861d2f562943e978cc11f8c344d`。尚未下载权重。

上游 README 的 VILA commit 存在不一致：依赖简介写 `da98f3b`，实际安装步骤写 `a5a380d6d09762d6f3fd0443aac6b475fba84f7e`。部署时需要隔离环境并验证依赖，不能直接覆盖现有 Qwen 环境。

## 可进行的试验

1. 高层轨迹测试：部署已发布 VLM，用已有杯子图和仿真截图预测点及夹爪开合，与 Qwen 输出比较。这个试验不需要底层权重，但无法证明闭环抓取成功。
2. 环境测试：安装 [Colosseum](https://github.com/robot-colosseum/robot-colosseum)，运行专家示范，导出前视图和真实末端投影轨迹，作为预测参照。专家执行结果不能当作 HAMSTER 执行结果。
3. 完整闭环：取得作者的路径条件底层策略，或另行实现并训练这部分，然后测量实际任务成功率。

仿真环境的[官方安装指南](https://robot-colosseum.readthedocs.io/en/latest/installation.html)依赖 CoppeliaSim 4.1、PyRep 和兼容版本的 RLBench。远程机器已核查为 RTX PRO 6000 Blackwell 96GB，GPU 空闲显存约 95GiB；尚未验证其无显示器渲染环境。容量足够尝试 13B VLM 推理，软件兼容性仍需实际验证。
