# 给能访问真实数据的 AI：核实与补齐任务

请结合本分支、[使用说明](lerobot_zh.md)、原始数据、采集代码历史版本和运行日志执行。
当前实现没有访问真实数据。每个结论附文件路径、commit/行号或日志时间；不能证明的保持未知，不能为通过校验虚构 evidence。

## 用户提供的源码事实

当前脚本控制右臂，支持 joint/ee，最终下发关节 MIT 位置、零目标速度、重力/摩擦补偿及 kp/kd。
joint/ee 分别为绝对关节/末端目标；VR 先用增量更新末端目标，再经 IK 下发绝对关节目标。
当前运动学为 base_link 下 tool_link，位置 m、关节 rad；旋转 6D 为第一列再第二列。
夹爪是电机位置，0 闭合、应用张开 1.6、SDK 上限 2.0。

传目标回调时 actions.arm_joints/actions.gripper 来自目标缓存，actions.arm_ee 是目标关节 FK；
未传回调则 actions 使用实测状态。当前平滑系数 0.1，Space 中止剩余 chunk，接管从实测初始化，恢复重置平滑器并重新推理。
这些是**当前代码语义**，不能直接证明某条历史 rollout 的采集事实。

## 未知项、检查方法和交付

| 未知项 | 核查依据 | 补齐结果 |
| --- | --- | --- |
| v2 子版本、列名、维度 | meta/info.json、Parquet schema、episodes.jsonl | 真实字段与 packed indices、相机 key，不能照抄模板 |
| 每个 episode 的 action 来源 | recorder 实际构造/调用点，是否传目标回调，历史 commit 和启动参数 | command_target / measured_state / unknown 分类；后两者不作为当前训练标签 |
| 缓存处于执行哪一层 | 原始模型 chunk、平滑后、限位后、IK 后、MIT send 前后 | 写清标签对应控制契约，排除未执行 chunk、IK 失败和旧缓存 |
| 观测与动作对齐 | 传感采集、图像曝光、缓存写入、send、recorder 各时间戳 | 证明 obs[t] 对应 action[t+offset]；变延迟不能用固定 offset 掩盖 |
| 数据/控制频率 | metadata fps、真实时间间隔、控制线程频率和周期内全部指令 | 明确一行表示哪个执行区间；缺中间指令时不能直接抽帧冒充完整序列 |
| 历史 joint/ee 模式 | CLI、配置快照、输出形状和回调类型 | 逐 episode policy_mode；当前默认 joint 不是历史证据 |
| 关节数、顺序、限制 | SDK、驱动、URDF、左右臂映射 | joint_names；后续需部署时另加关节/速度/加速度/碰撞检查 |
| FK、坐标、旋转 | 采集版本的 base/tool、工具偏移、标定；非零多轴旋转样本 | 确认 m/rad/列6D；单位旋转不足以判断行列顺序 |
| 夹爪 | SDK 版本、真实范围、Trigger 映射 | 确认 motor rad，分别理解 1.6 和 2.0，不缩放成宽度或概率 |
| 任务/环境一致性 | task_index、语义任务、相机/底座位置、工具、控制器参数 | 每次导入同任务、同标定、同执行契约的集合 |
| 成败标签 | 可信标注、终止原因、录像复核及人工接管后的最终结果 | 逐 episode 布尔标签和依据；接管/静止/结束都不是自动成功判据 |
| 接管边界 | Space/Squeeze/Trigger 事件、reset、推理重启、缓存更新 | 检查切换前后数秒，排除被取消 chunk、尚未跟随的夹爪假目标、陈旧缓存 |
| 相机同步 | MP4 PTS、帧数、帧率、起始偏移、坏帧漏帧 | 每相机 offset/tolerance；复核 RGB、方向、缩放补边 |
| 重复数据和划分 | 同源裁剪、重复录像/episode、跨文件重录 | 同源轨迹一起划分，避免泄漏，另留最终测试集 |

action 与 observation 很接近只能作为可疑线索，静止时两者本来就可能相等，不能凭相关性证明回调来源。
平滑器与高频控制器可能含未记录状态；若观测不充分，需要补上一时刻指令、速度或其他已核实状态，并扩展观测 schema/配置。
不要简单把历史参数 0.1 用于所有数据，也不要对已平滑目标再次平滑。

## 执行顺序

1. 运行 inspect，扫描全部 episode 的 schema、帧率和模式一致性；抽查成功、失败、接管、恢复、夹爪开合代表片段。
2. 完成 right_arm.json 和 episodes.json；逐条引用证据。独立审计表记录排除的原始 ID 与原因。
3. 若同一 episode 的动作来源按帧变化，不能整条简单标 command_target；先根据真实日志恢复连续可解释的数据。
   不要切碎轨迹后将所有碎片继承“成功”，这会改变成功后缀和步数收益的含义。
4. 若只有实测 action，检查能否从原始命令日志恢复。不能时报告无法恢复真实控制标签。
   q[t+1]、FK(q[t])、观测插值均非已下发指令；proxy 实验须另行命名与评估，不能关闭校验冒充真值。
5. 先转换小子集，手工核对目标/实测、原始索引、正负 offset 方向、旋转和夹爪；查看转换图像及 import_manifest。
6. 运行新增 CPU 测试，再在训练服务器构建真实 DINO 缓存，对三个模型分别做一轮 smoke test。
7. 固定训练/验证/独立测试划分后正式训练。检查 Verifier 各 horizon 的负样本数、假阳性率和阈值；数据不足时补数据，不能只降阈值。
8. 检查三类桥数量、步数收益、幅度/连续性、夹爪范围和可达性。Verifier 接受不等于真机安全证明。
9. 比较相同 split、种子、horizon、帧率和输入版本的 baseline/augmented。更改映射后重建缓存/模型/桥，不复用旧目录。

## 如需接入已有 LeRobot/VLA 训练器

本分支提供 export_lerobot_bridges；你的外部 trainer 尚未修改。
验证 metadata_json 与源哈希后，建立引用原数据的 Dataset wrapper：

- 保留原始训练样本，加入由真实 source 观测开始的桥窗口。
- source_frame_index 是原始**观测行**；提取原始动作前缀仍需加 action_row_offset。
- source 处替换为桥动作，桥结束后的所有动作维度 loss 置零。
- 不追加 target episode 的动作，不为桥内部虚构图像、时间戳或 proprio。
- joint 是 N+1 维；ee 是 xyz+列6D+motor-rad 夹爪。核对 trainer 的归一化、chunk 对齐和 mask。
- 若 trainer 强制每一步都有真实图像，必须改样本接口；不能复制 source/target 图像伪造完整 LeRobot episode。

## 最终交付

交付已填写配置、逐 episode 审计表、排除列表、采集版本与控制契约、转换报告、真实数据 smoke 日志和正式实验配置。
若改外部 trainer，给出提交及测试：桥后 loss=0、原始样本保留、没有目标尾部拼接、无跨 episode 索引错误。
仍缺失的证据单列，写清还需要哪份日志、历史源码或人工标注。

本仓库只提交通用代码、合成测试和模板。真实数据、视频、私有配置、权重与日志留在数据服务器，不写入公共模板。
