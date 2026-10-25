# ChituMax AMP（RobotLoad 移植）

来源为 AMP_for_hardware 的 origin/RobotLoad，固定提交
b4ae8a0ddc86094b240f01f1bfc87084164e7d56。源仓库当前分支无需切换。

## 启动

### 独立 conda 环境

推荐固定 Python 3.11、Isaac Sim 5.1.0、Isaac Lab v2.3.2、
PyTorch 2.7.0 + CUDA 12.8。安装命令如下（需能访问 NVIDIA、PyPI 和 GitHub）：

```bash
conda env create -f environment.chitu_amp.yml
conda activate chitu_amp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com -c requirements.chitu_amp.constraints.txt
mkdir -p "$CONDA_PREFIX/opt"
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$CONDA_PREFIX/opt/IsaacLab"
git -C "$CONDA_PREFIX/opt/IsaacLab" apply "$PWD/patches/isaaclab-2.3.2-sim-5.1.patch"
python -m pip install --build-constraint requirements.chitu_amp.build-constraints.txt -c requirements.chitu_amp.constraints.txt \
  -e "$CONDA_PREFIX/opt/IsaacLab/source/isaaclab" \
  -e "$CONDA_PREFIX/opt/IsaacLab/source/isaaclab_assets" \
  -e "$CONDA_PREFIX/opt/IsaacLab/source/isaaclab_tasks" \
  -e "$CONDA_PREFIX/opt/IsaacLab/source/isaaclab_rl" \
  -e "$CONDA_PREFIX/opt/IsaacLab/source/isaaclab_mimic" -e source
python -m pip check
python scripts/check_chitu_amp.py
```

Isaac Sim 的精确依赖包含 NumPy 1.26.0 和 packaging 23.0，请保留约束文件，
避免单独升级导致不兼容。首次启动还需按 NVIDIA 提示接受软件许可。
构建约束固定 setuptools 80.9.0，兼容 flatdict 对 pkg_resources 的依赖；
`--build-constraint` 需要 pip 25.3 或更新版本。
另有一个显式环境兼容补丁：Isaac Lab v2.3.2 的 Starlette 0.49.1
与 Isaac Sim 5.1 的 FastAPI 0.115.7（要求 Starlette <0.46）冲突。
补丁只把 Lab 的安装依赖固定为 Sim 使用的 Starlette 0.45.3，
不改仿真代码；此环境面向本地 locomotion 训练，不用于部署 Web 服务。
上述依赖用于项目自带的 HimLoco RSL-RL，不额外安装其他 RL 框架。
Lab v2.3.2 还要求 URDF importer 2.4.31，而 Sim 5.1 默认附带 2.4.30。
首次加载 URDF 时需联网下载此扩展，并确保 Omniverse 的用户缓存目录可写。
本机 chitu_amp 环境已补齐 2.4.31，存放在 Isaac Sim 的 extscache 中。

完整物理仿真需要可访问的 NVIDIA GPU；安装 CUDA wheel 不会修复 WSL GPU
透传问题。WSL 下可检查 `/usr/lib/wsl/lib/nvidia-smi` 和 `/dev/dxg`，再执行：

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

### 训练与回放

使用本项目支持的 Isaac Sim / Isaac Lab 环境，先安装扩展：

```bash
python -m pip install -e source
python scripts/import_robotload_assets.py --source /path/to/AMP_for_hardware
python scripts/check_chitu_amp.py
python scripts/himloco_rsl_rl/train.py --task ChituMax-AMP --headless --num_envs 64 --max_iterations 2 agent.amp_num_preload_transitions=1024 agent.algorithm.amp_replay_buffer_size=4096
```

预检只依赖 NumPy、PyTorch，检查模型引用、62 个动作文件、腿序、速度差分、
FK、镜像映射、MGMS 和一次包含判别器及对称损失的 PPO 参数更新。
它不替代 Isaac Lab 的物理仿真冒烟测试或长期收敛验证。

冒烟测试成功后正式训练：

```bash
python scripts/himloco_rsl_rl/train.py --task ChituMax-AMP --headless
python scripts/himloco_rsl_rl/play.py --task ChituMax-AMP-Play --num_envs 16
```

默认 5480 环境、50 Hz 策略、200 Hz 物理、200 万预加载转换和 100 万 replay
容量；显存不足时降低环境数及上述 Hydra 参数。
日志位于 logs/himloco_rsl_rl/chitu_max_amp。
续训使用 --resume --load_run RUN_NAME --checkpoint model_N.pt。

## 数据与训练约定

- 原始 61 维数据腿序 FR/FL/RR/RL，加载后统一为 LF/LH/RF/RH。
  模拟器关节按名称解析，不依赖 PhysX 内部 DOF 顺序。
- 专家和策略共用 RobotLoad 的解析 FK；专家关节速度按 FrameDuration
  差分重算。判别器状态为 43 维，输入为前后两帧共 86 维。
- RSI 概率 0.85，重置关节位置、速度、根姿态及速度，并恢复专家速度指令。
  终止状态在 reset 前捕获，观测历史在 reset 时清空。
- 保留 RobotLoad 的 PD 300/14、HAA/HFE/KFE 动作缩放 0.18/0.3/0.3、
  关节限幅及 MGMS 摩擦。payload 不合并，按四档覆盖约 0–40 kg；
  最小 payload 质量 0.01 kg，避免零质量刚体。
- 迁移平地任务奖励（含站立、后退、原地旋转、外八约束和正值截断）。
  Isaac Lab RewardManager 会乘 dt，因此组合奖励先除 step_dt，
  保持 RobotLoad 的逐步奖励量级。
- AMP 系数 10、任务混合比例 0.28、对称 KL 系数 15。
  最小探索噪声按关节名称匹配，避免 Isaac Lab 关节顺序造成错配。

## 与原训练接口的区别

2026-10-25 本机验证：chitu_amp 的 pip check、62 组动作数据预检、
ChituMax 训练/回放任务注册通过；CPU 下 2 个环境、1 次迭代（48 步）
已完整运行并保存 checkpoint。当前沙箱会话无法访问 GPU，GPU 训练与收敛未验证。

策略继续使用 HimLoco 的 AMPActorCritic 和状态估计器，history_length=0；
policy 为 45 维（含上一动作），critic 为 51 维。并非原 Isaac Gym
42/48 维策略的二进制兼容移植，旧 checkpoint 不应直接续训。
部署需使用本次训练导出的配置及新策略，现有 Go2 sim2sim 配置不能直接使用。

负载、质量、质心、增益和外力随机化通过 Isaac Lab manager 执行；
接触、外力时序和求解器与 Isaac Gym 不完全相同，不能据此承诺相同收敛曲线。
Play 配置关闭 RSI、观测噪声和域随机化，保留 URDF 名义负载及 MGMS。

修改动作数据源时，同时修改 runner 的 amp_motion_files 与
env.events.reference_state.params.motion_files，保持 RSI 和判别器使用同一数据。
