# 右臂 LeRobot v2：使用说明

本分支基于上游 `c0960ab`，新增 LeRobot v2.0/v2.1 导入、绝对关节动作、末端旋转转换和动作桥导出。
支持本地 Parquet、逐 episode MP4、Parquet 中的编码图像 bytes/path。原始数据只读。
尚未访问你的真实数据或进行完整 GPU 训练、真机部署。

先请能访问数据的 AI 按 [核实与补齐说明](lerobot_data_handoff_zh.md) 完成两个 JSON 模板。
模板字段名是示例；null、unknown、空 evidence 有意阻止未经核实的转换。

## 算法边界

依据 [论文 §3.2–3.4、附录 D](https://arxiv.org/pdf/2610.02339)，IDM 提议短动作序列，Verifier 判断前缀可达性，选择有收益的桥。
桥训练使用真实源观测、原始动作前缀和桥动作，在桥结束处屏蔽后续 loss，并保留原始监督。
本分支沿用上游生成、验证和采样实现，未伪造桥中间图像或直接拼接目标轨迹。

绝对关节动作是本次工程扩展，论文实验效果不能直接视为此机器人上的效果。
末端路径使用共同 base_link 坐标系，不采用 UMI 的轨迹相对变换。新增候选预算/采样权重需在你的数据上调参。

| 输入 | Needlework 内部表示 |
| --- | --- |
| 绝对关节目标 | `[q0,...,qN-1,gripper]`；N 和顺序由 joint_names 指定 |
| 绝对末端目标 | `[x,y,z,row6D,gripper]`，共 10 维；位置 m |
| 关节/末端实测状态 | 独立 observation，不用 action 代替 |
| 夹爪 | 电机 rad；0 闭，应用张开值 1.6，SDK 上限 2.0，保留原值 |
| action 为实测状态或来源未知 | 拒绝导入为控制标签 |

输入列 6D：`[r00,r10,r20,r01,r11,r21]`；内部行 6D：`[r00,r01,r02,r10,r11,r12]`。
先按列 Gram-Schmidt 正交化，再转为内部行格式，不能只 reshape。导出末端桥时转回列格式。
输出对应正交化后的旋转，不保留原始非正交六元组。

不重新施加 0.1 平滑，不运行 FK/IK，不下发 MIT 指令。有真实目标关节时优先用 joint。
目标关节的 FK 不唯一决定关节解，不能认为末端训练加 IK 会复现原有 joint 指令。

## 1. 环境与盘点

完整训练按根目录 README / install_deps.sh 安装 Linux、Python 3.10、CUDA 和 DINOv3 权重。
在同一环境补充 I/O 依赖：

```bash
python -m pip install -r requirements-lerobot.txt
export NEEDLEWORK_ROOT=/absolute/path/to/needlework
source set_env.sh
python -m needlework.data.lerobot inspect \
  --root /absolute/path/to/raw_lerobot \
  --sample-episodes 3 --output /absolute/path/to/audit.json
```

无需安装最新 LeRobot。读取 v2 的路径模板、chunks_size、features 和 fps；v3 明确拒绝。
格式参照 [LeRobot v0.3.3 源码](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py)。
audit.json 提供 metadata、样本 schema、数值形状和范围，不能证明动作语义。输出必须是源目录之外的新文件。

## 2. 填写配置与 episode 标签

复制 `examples/lerobot/right_arm.template.json` 和 `episodes.template.json` 到自己的配置目录。

| 配置字段 | 填写要求 |
| --- | --- |
| action_mode | 导入表示 joint/ee，不是根据当前脚本默认值猜历史模式 |
| action_semantics | 经证实后填写 command_target |
| joint_names | 控制器实际顺序的全部关节名称；joint 必填，不能猜六轴/七轴 |
| task_index | 单一、语义一致的任务 ID |
| action_row_offset | observation 行 t 对应 action 行 t+offset，必须有日志依据 |
| timestamp_tolerance_s | 相邻时间间隔偏离 1/fps 的容差，须小于半帧 |
| video_tolerance_s | 视频 PTS 匹配误差容差，须小于一帧 |
| fields | 真实列名，支持 `{"column":"action","indices":[0,1,2]}` 显式切片 |
| cameras | 内部别名如 camera_front → 真实 feature key；目标视频时间为 observation_timestamp + timestamp_offset_s |
| gripper_range | 已核实的电机范围，如 [0,2.0]；越界报错，不裁剪 |
| evidence | action_source、time_alignment、control_and_sampling、coordinates_and_units、outcomes 的文件/日志依据 |

`offset=1` 配对 obs[0:N-1] 与 action[1:N]；`offset=-1` 配对 obs[1:N] 与 action[0:N-1]。
裁掉无对应标签的端点，不跨 episode 借动作。这只修复已知固定日志错位，不能修复变延迟、物理跟踪滞后或缺失指令。

ee 模式将 fields 中 action_joint/observation_joint 换成 action_ee/observation_ee，各取 xyz+列6D 共 9 维。
夹爪继续单独选择；原始字段有额外维度时用 indices。joint 与 ee 的采集模式不能在未知时混合。

episodes.json 只列已核实轨迹，每项包含 episode_index、布尔 success、action_semantics、历史 policy_mode 和 evidence。
policy_mode 允许 joint/ee/mixed，描述采集时配置；VR 经 IK 后可提供绝对关节标签。
不因接管自动判失败，不将人工片段剪成假成功 episode，不填猜测标签。

## 3. 转换

```bash
python -m needlework.data.lerobot convert \
  --root /absolute/path/to/raw_lerobot --task my_right_arm_task \
  --config /absolute/path/to/right_arm.json \
  --episodes /absolute/path/to/episodes.json \
  --output "$NEEDLEWORK_ROOT/data/lerobot/my_right_arm_task"
```

输出必须是不存在且与源数据独立的目录。生成 success.zarr、可选 failure.zarr、各自 sha256 标记和 import_manifest.json。
按成功/失败分组后各自按原 episode ID 排序；manifest 保存源文件哈希、配置、fps、原始 ID、裁剪起点和内部起点。
指纹包含转换依据，供特征缓存和 checkpoint 数据身份检查。不要手工修改转换结果或 manifest。

数值检查覆盖有限性、维度、索引、任务、帧率和夹爪范围。1.6 到已核实上限之间的值保留，并统计数量。
图像转 RGB、保持比例缩放、居中补黑到 224×224。MP4 按 PTS 匹配，错位或重复选帧时报错。
嵌入图像仅支持行对齐、时间偏移为 0。当前不验证未知的关节限位、碰撞、速度/加速度和控制器配置。

## 4. 特征、训练与桥

以下假设导入相机别名为 camera_front/camera_wrist。实际相机不同，所有训练与 stitching 命令一致覆盖，例如：
`'task.obs.cameras=[camera_front]'`。baseline 至少需要两个成功 episode；可靠校准需要更多独立且足够长的轨迹。

```bash
python -m needlework.build_features --domain lerobot --task my_right_arm_task \
  --poolings spatial_softmax_14 patch_grid_7 --device cuda:0
python -m needlework.train component=policy task=lerobot/joint \
  task.name=my_right_arm_task run.name=right_baseline
python -m needlework.train component=idm task=lerobot/joint \
  task.name=my_right_arm_task run.name=right_idm
python -m needlework.train component=verifier task=lerobot/joint \
  task.name=my_right_arm_task run.name=right_verifier
python -m needlework.stitch task=lerobot/joint \
  task.name=my_right_arm_task run.name=right_bridges \
  idm_checkpoint=/absolute/path/to/idm_run/checkpoints/last.ckpt \
  verifier_checkpoint=/absolute/path/to/verifier_run/checkpoints/last.ckpt 'gpus=[0]'
python -m needlework.train component=policy task=lerobot/joint \
  task.name=my_right_arm_task run.name=right_augmented \
  sampler=augmented sampler.bridges=/absolute/path/to/bridge_run/bridges.zarr
```

ee 路径统一改为 task=lerobot/ee。只允许 stride=1 和绝对动作；不能沿用 UMI stride=3 丢弃中间指令。
默认 24 行预测、2 行观测，留下 23 行可执行桥。只有成功数据时 stitching 传 `'stages=[1,2]'`。
先用 `task.epochs.idm=1` 等做 smoke test；短训练不能证明桥有效。正式模型需要从头训练，保持数据、相机、帧率、horizon 一致。
LeRobot 路径无 Robomimic 仿真评估；必须单独验证真实效果。平滑器/控制器有隐状态时，须核实当前观测是否足够，不承诺保持目标等于保持实测姿态。

## 5. 导出给外部训练器

```bash
python -m needlework.data.export_lerobot_bridges --task my_right_arm_task \
  --bridges /absolute/path/to/bridge_run/bridges.zarr \
  --output /absolute/path/to/right_bridge_supervision.npz
```

以 `np.load(path, allow_pickle=False)` 读取 packed actions、action_start/action_len、
source/target_episode_index、source/target_frame_index、stage、verifier_margin 和 metadata_json。
joint 保留顺序，ee 转回列 6D，夹爪保留 rad。元数据记录生成夹爪超出 SDK 区间的数量，不静默裁剪。

这是训练监督 sidecar，不是原生 LeRobot episode 或机器人执行指令。已有 Needlework augmented sampler 可直接训练桥。
外部 VLA/LeRobot trainer 尚未接入，需要实现源观测窗口、原始前缀、桥替换、桥后 loss mask 和原始样本保留。
不能复制目标图像补齐桥中间帧，也不能把目标 episode 尾部直接接在桥后。

## 验证

```bash
python -m pytest tests/test_lerobot.py -q
ruff check src tests
ruff format --check src tests
```

新增合成测试无需真实数据或 CUDA；上游真实数据/GPU 回归测试缺输入时跳过，不算通过。
真实数据适配、训练效果与部署仍须按 [交接说明](lerobot_data_handoff_zh.md) 验收。
