# HimLoco-AMP-Lab

[English](README.md)

这是一个面向 Unitree Go2 和 ChituMax 的 Isaac Lab 四足运动控制项目，结合 PPO、历史观测与
对抗运动先验（AMP），并提供训练、策略回放与导出、专家动作重放以及 MuJoCo
sim-to-sim 工具。

## 主要功能

- Unitree Go2 的 AMP 与纯 PPO 速度控制任务。
- ChituMax 的 RobotLoad AMP 训练（负载随机化、MGMS、RSI、对称损失），见 [训练说明](docs/chitu_max_amp.md)。
- 地形课程、域随机化与任务奖励。
- 六帧策略观测和 AMP 判别器。
- 用于验证本地专家轨迹的运动学重放工具。
- TorchScript/ONNX 导出及 MuJoCo sim-to-sim。
- 检查私有数据和生成文件的发布安全脚本。

## 安装

请先安装 Isaac Sim 和 Isaac Lab。当前配置面向开发时使用的 Isaac Sim 5.1、
Python 3.11 环境，其他版本可能需要适配。激活该环境后运行：

```bash
git clone <repository-url>
cd HimLoco-AMP-Lab
python -m pip install -e source
python scripts/list_envs.py
```

Go2 描述文件位于 `urdf/go2/`，请保持它与 `source/` 的相对位置。上游资源许可
保留在 `urdf/go2/LICENSE`。

## 任务与命令

| 任务 | 用途 |
| --- | --- |
| `Unitree-Go2-AMP` | AMP 训练 |
| `Unitree-Go2-AMP-Play` | AMP 策略回放与导出 |
| `Unitree-Go2-Velocity` | 纯 PPO 训练 |
| `Unitree-Go2-Velocity-Play` | 纯 PPO 策略回放与导出 |

```bash
# 训练
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-AMP --num_envs 4096 --headless
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --num_envs 4096 --headless

# 回放并导出
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-AMP-Play --num_envs 16
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play --num_envs 16
```

显存不足时降低 `--num_envs`。可使用 `--load_run` 和 `--checkpoint` 选择模型。
回放会将部署模型写入所选训练目录的 `exported/`；日志和模型已被 Git 忽略。

## 专家动作数据

本仓库有意**不发布专家轨迹**。请只使用获得授权的数据，并在本地放入
`datasets/motions/trot/*.json`。AMP 训练和 AMP 回放依赖这些文件，纯 PPO 不依赖。

帧结构由 `source/himloco_lab/rsl_rl/datasets/motion_loader.py` 定义，包括根节点
位姿、`xyzw` 顺序的四元数、关节位置、局部足端位置、机身线速度/角速度及关节速度。
AMP 关节顺序为：

```text
FL hip/thigh/calf, FR hip/thigh/calf,
RL hip/thigh/calf, RR hip/thigh/calf
```

训练前请检查动作重定向和关节顺序：

```bash
python scripts/replay_data.py --task Unitree-Go2-AMP-Play --max_steps 500
```

ChituMax 的动捕放在 `resources/robotload/datasets/chitu/mocap_motions/*.txt`，文件内容仍是上面的 JSON 帧。关节顺序是 LF/LH/RF/RH，每条腿依次为 HAA、HFE、KFE。回放会按训练配置做 Chitu 腿序重排，并用正运动学重算足端位置。训练前检查：

```bash
python scripts/replay_data.py --task ChituMax-AMP-Play --motion yaw_0p5
```

`--motion` 可以是文件名（不带 `.txt`）、完整路径，或从 0 开始的序号。不写则按文件名顺序播放目录里的全部轨迹。

## ChituMax ONNX 导出

不需要启动 Isaac，也不用改 `rl_deploy_python` 的 `chitu_max/trot`。脚本把估计器和策略包进与 `AMP_for_hardware` 相同的部署接口：输入仍是 47 维单帧，历史从旧到新排列，输出是 FR/FL/RR/RL 顺序的原始动作。

```bash
python scripts/export_policy_to_onnx.py \
  --checkpoint logs/himloco_rsl_rl/chitu_max_amp/<run>/model_26900.pt
```

默认写到该检查点旁边的 `exported/chitu_max_policy.onnx`，也可以用 `--output` 指定路径。图内部完成这些转换：丢掉步态相位，保留已经乘过 0.25 的角速度，把指令从部署的 1.5 倍还原成训练时的原始速度，并把关节从部署的每条腿 hip/thigh/calf 排成训练时的顺序：4 个 HAA、4 个 HFE、4 个 KFE，组内都是 LF、LH、RF、RH。动作再排回 FR/FL/RR/RL。部署侧继续使用 `obs_dim: 47`、`command_scale: [1.5, 1.5, 1.5]` 和 `scale_profile: amp`。这个 `0.18 / 0.3 / 0.3` 是按每条腿三个关节给的，和训练里按关节类型给的缩放一致，不要改成四个 0.18 再接八个 0.3。安装 `onnxruntime` 后，脚本会用随机输入对比 PyTorch。

## 当前 AMP 与控制参数

- 奖励混合：`0.9 * task_reward + 0.1 * style_reward`。
- AMP 风格奖励系数：`0.15`。
- 5 帧历史加当前帧：策略输入 `6 x 45 = 270` 维。
- 物理频率 200 Hz（`dt = 0.005 s`），策略频率 50 Hz（`decimation = 4`）。
- 关节目标：`默认关节位置 + 0.25 * action`。
- 默认增益：Kp 25、Kd 0.5。

算法配置位于 `source/himloco_lab/tasks/locomotion/agents/himloco_amp_rsl_rl_cfg.py`；
环境、奖励、速度指令、地形和随机化配置位于
`source/himloco_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`。

## MuJoCo Sim2Sim

```bash
python -m pip install mujoco pyyaml
python deploy/deploy_sim2sim/sim2sim.py
```

ChituMax 使用 `play.py` 导出的 `policy.pt`（45 维训练观测），不是给实机的 47 维 ONNX：

```bash
python deploy/deploy_sim2sim/sim2sim.py --config deploy/deploy_sim2sim/config/chitu_max.yaml
```

程序自动选择最新导出的 Go2 AMP 策略，也可以指定策略或执行无窗口测试：

```bash
python deploy/deploy_sim2sim/sim2sim.py --policy /path/to/policy.pt
python deploy/deploy_sim2sim/sim2sim.py --headless --no-joystick --no-real-time --duration 2
```

观测和关节顺序、控制方式、力矩限制及仿真器差异见
[sim-to-sim 说明](deploy/deploy_sim2sim/README.md)。sim-to-sim 能运行并不代表策略
可以安全部署到实体机器人。

## 目录结构

```text
source/himloco_lab/             Isaac Lab 扩展与学习代码
scripts/himloco_rsl_rl/         训练和策略回放入口
scripts/replay_data.py          专家动作重放
scripts/export_policy_to_onnx.py ChituMax checkpoint 导出为单个 ONNX
deploy/deploy_sim2sim/          MuJoCo sim-to-sim
urdf/go2/                       Go2 描述文件
datasets/                       本地专家数据（已忽略）
logs/ 和 outputs/              本地训练输出（已忽略）
```

## 发布安全

此工作树源自私有研究项目。不要强制添加专家数据、检查点、策略、日志或密钥。
发布前运行：

```bash
python scripts/check_publication.py
```

脚本会检查 Git 索引及所有可达提交。如果继承历史含有私有数据，应从经过审查、
没有旧历史的新仓库发布；同时仍需人工检查暂存差异和第三方资源许可证。

## 许可证

项目代码采用 [LICENSE](LICENSE)，Go2 资源保留 `urdf/go2/LICENSE` 中的许可证。
