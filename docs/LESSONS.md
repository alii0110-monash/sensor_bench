# SensorBench 踩坑记录（LESSONS）

> 记录项目开发/实验过程中踩过的坑、根因与规避方法。**持续更新**——每次遇到新坑、或发现既有记录有误，追加/修订本文件。按类别组织，最新在上。

## 一、环境与运行时

### 1. 项目 python 环境识别（2026-08-16）
- **坑**：项目用 `/home/li/bin/pytest` → `/usr/bin/python3.12` 全局环境（torch 2.7.1+cu128, psutil, transformers 齐全）。不是系统 `python3`（miniconda base 无 torch）、不是 `.venv`（空）、不是 conda `mmfi` 环境（空目录）。
- **浪费**：逐个猜环境花了好几轮。`mmwave`/`rfgen` 各有部分依赖，都不全。
- **规避**：判断项目运行时环境用 `which pytest` + 看 pip 安装位置 + 跑 `python -c "import torch"` 实测，别逐个猜。项目脚本 `#!/usr/bin/env python` 不可靠（PATH 里 `python` 是 miniconda base）。

## 二、长任务运行

### 2. 误判训练卡死（2026-08-16，M6b 实验）
- **坑**：用 `timeout 115s` 跑训练命令，无 ep0 输出就断定"卡住"，误 kill 了 A2 变体、多次误报。
- **真相**：A 变体从启动到 ep0 需要 ~315s（含 CLIP 加载、数据集缓存初始化约 5 分钟）。`batch=32` 正常，只是初始化慢。
- **规避**：长任务判"卡死"三查——① 给足初始化时间（观察窗口 ≥ 5-10min）② 看 I/O 是否推进（`cat /proc/<pid>/io` 的 rchar 是否增长，注意要抓真 python pid 而非 bash wrapper）③ 看 GPU util。**"没有新日志"不等于卡住。**

### 3. 显存告警未按规则触发（2026-08-16）
- **坑**：batch=256 训练显存 15.8GB/16.3GB（≈97%），连续轮询却没按 AGENTS.md 要求"显存 ≥90% 立即报告"，被用户点名。
- **规避**：轮询长任务时**每轮**都要查 `nvidia-smi --query-gpu=utilization.gpu,memory.used`，>90% 立即报告给选项。16GB 卡跑 batch=256 本身不现实。

### 4. batch=256 + cache_size=4096 异常未定位根因（2026-08-16）
- **现象**：rchar 冻结、34min 无 ep0，GPU 100% 但显存带宽利用率 1-7%（卡死的典型特征），RSS ~19GB。
- **存疑点**：A 用默认 cache_size=256 正常；probe（batch=128 + cache=256）也正常。怀疑 cache_size=4096 导致 LazySplit 内存膨胀 + 慢速路径，但被终止未确认。**此坑未闭环，复测时须先小规模 probe 再全量。**
- **规避**：大 batch 改参数前，先用小实验（单 epoch 或部分数据）验证能跑通，再全量训练。别直接全量跑。

### 5. 后台进程管理（2026-08-16）
- **坑 a**：`pgrep -f "train_alignment.py --out-tag B"` 抓到 bash wrapper（rchar 15008）而非 python 子进程（rchar 19.6GB）。wrapper 的 I/O/CPU 无参考价值。
  - **规避**：匹配进程用 `ps aux | grep "python3 scripts/..."`（精确到 python 解释器）或 `nvidia-smi --query-compute-apps=pid`。
- **坑 b**：kill 只杀 wrapper 没杀 python 子进程（`-u -c` 内联脚本），GPU 残留 100%。
  - **规避**：清后台进程先 `nvidia-smi --query-compute-apps=pid` 确认 GPU 占用进程，再逐个 kill；kill 后复查 GPU util。

### 6. bash 工具 sleep 超时（2026-08-16）
- **坑**：`sleep 300` 超过 bash 工具 120s 超时，命令被杀，日志输出也丢了。
- **规避**：轮询拆成 ≤110s 的短 sleep；长等待用后台任务 + 短轮询，别前台长 sleep。

### 7. rchar 冻结被误读为卡住（2026-08-16）
- **坑**：rchar 停在 19.6GB 被当作"训练停滞"证据。实为**数据全部缓存完毕后的正常态**（cache 装满不再读盘），不代表训练停滞。
- **规避**：rchar 冻结要结合 GPU util/显存带宽一起看。GPU 真在算（util 高 + 显存带宽高）则正常；GPU 100% 但带宽 1-7% 才是卡死信号。

## 三、代码与设计

### 8. label-aware mask 掩掉对角线正样本（M6b spec 评审抓出）
- **坑**：`same = labels[:,None]==labels[None,:]` 对 `(i,i)` 必为 True，直接 mask 会连正样本一起 -inf → 整行 NaN。
- **规避**：`same[torch.arange(B), torch.arange(B)] = False` 显式保留对角线。

### 9. info_nce_loss 改动丢 F.normalize（M6b plan 评审抓出）
- **坑**：改造时把 `z/t = F.normalize(...)` 删了，静默改变 logits 尺度（影响温度语义），且连带影响 `train_projection.py:88` 调用方，违背"向后兼容"承诺。
- **规避**：改公共函数前先 grep 所有调用方；保留 normalize 等既有语义。

### 10. 测试数据与保底逻辑冲突（M6b plan 评审抓出）
- **坑**：label-aware 测试用 B=8 每类 2 样本 → `n_neg=6 < min_negatives=8` 保底触发 → 不 mask → 断言全挂。
- **规避**：测试数据要满足实现前提（每行可用负样本 ≥ min_negatives），用 B=16、每类 8 样本。

### 11. `_masked_logits` 归属不一致（M6b plan 评审抓出）
- **坑**：测试写 `info_nce_loss._masked_logits(...)`，实现却是模块级函数 → AttributeError。
- **规避**：测试直接 import 模块级辅助函数，别挂到函数对象上。

### 12. `_diagnose_label` 设备不匹配（M6b 实现，实测抓出）
- **坑**：`torch.tensor(labels)` 在 CPU，`above`（sim 比较）在 CUDA → RuntimeError。
- **规避**：`torch.tensor(labels, device=sim.device)`，比较张量同设备。

### 13. eval 加载 checkpoint 的 classification_head 键冲突（M6b spec 评审抓出）
- **坑**：C/E 变体训练会存 `classification_head.*` 键，eval 用 strict=True 加载炸 `unexpected keys`。
- **规避**：训练保存 checkpoint 时剔除 `classification_head.*` 键（只存 encoders+projection_head）。

## 四、数据与 loader

### 14. cache thrash / cache_size 与 batch 不匹配（M6b spec 评审抓出）
- **坑**：loader 默认 `cache_size=256`，batch=256 时缓存命中率塌陷，逐 epoch 从盘重读（v4 已有先例：25min/epoch）。
- **规避**：大 batch 时提 cache_size；但注意 cache_size 过大本身有内存膨胀风险（见坑 4，未闭环）。

### 15. held-out 语义失真（M5a 遗留，M6b spec 评审注明）
- **坑**：训练用全部 `ds.train`，held-out base 只从评测排除、未从训练排除。相对比较有效，但不能叫真 held-out。
- **规避**：对比实验对称即可；文档注明这是"评测排除集"而非训练 held-out。

## 五、流程与工具

### 16. `git commit` 自动生成身份（多次出现）
- **坑**：`Committer: alii0110-monash <li@li.localdomain>` 自动配置，非用户 git 身份。不影响内容但噪音大。
- **规避**：可提示用户 `git config --global user.name/email`，或用 `--author` 显式指定。

### 17. 工具输出被截断时用文件而非 tail（AGENTS 通用）
- **坑**：bash 工具输出超 51200B 截断后写入文件，直接 `tail` 会丢失上下文。
- **规避**：用 Read/Grep 读截断文件，别用 tail 截流。
## 一、环境与运行时

### 18. 并行跑两个 dataset probe 把硬盘 IO 打满（2026-08-17，M6c dataset_quality）
- **坑**：在两个 dataset (v1 + v2) 上同时 `setsid` 启 `run_dataset_quality.py`，两进程并发读各自的 ~16k pickle 文件 + 各自的迭代 probe → 整盘持续全速读，系统卡死（用户中断）。
- **根因**：两个 probe 各自跑 5 个 per-modality + 1 个 concat + JS 跨模态 = 每个 probe 多次遍历 train；两个并发 = ~10 个 dataset scan 同时打同一磁盘。
- **规避**：磁盘 I/O 密集任务**永远串行**，不要靠 `setsid` 并行省时间。RAM 充足 ≠ I/O 充足。
- **顺序恢复**：v1 串行完成（~2min）+ v2 串行完成（~2min），无 I/O 争用。两次共 ~5min 与并行目标相当，但不会卡死。

### 19. mmwave 数据语义错读（领域知识纠错，2026-08-18 → 2026-08-20 二次修正）
- **坑（2026-08-18 第一次发现）**：`extract_mmwave_features` 把 mmwave 当作稠密 64 range × 5 doppler 热力图处理，但实际上：
  - mmwave 是 **4D 雷达稀疏点云**，shape `(T, 64, 5)` = T 帧 × 最多 64 检测点/帧 × 5 属性/点
  - 每帧仅 12-22 个 active 点，其余**零填充**
- **坑（2026-08-20 二次发现，更严重）**：之前推断 5 列是 `[range, velocity, azimuth, SNR, elevation]`（球坐标），但 **2026-08-20 验证 MMFi 官方代码后确认** 5 列实际是**笛卡尔坐标 + 速度 + 强度**：
  - 真实列序（confirmed via `/home/li/datasets/MMFi_dataset/mmfi_omni/codecs/mmwave_vae.py`）：
    - `attr0=x`, `attr1=y`, `attr2=z`（Cartesian m，3D 几何坐标）
    - `attr3=doppler`（径向速度 m/s）
    - `attr4=intensity`（反射强度 / SNR）
  - 铁证：官方代码用前 3 列做 3D FPS 采样 + Chamfer Distance 损失 + per-frame 3D 中心化：
    - `mmwave_vae.py:65`   `farthest_point_sample(x[:, :, :3])`
    - `mmwave_vae.py:104`  `pred_xyz = pred_points[:, :, :3]; cdist(...)`
    - `train_mmwave_codec.py:44`  `points[:, :, :3] -= mean(axis=1)`
  - V5_architecture.md:80 也明确说 "mmwave 有 5 通道 (XYZ+速度)"
- **症状**：球坐标错读让 `extract_mmwave_features` 抽的特征方向反了——probe val acc 0.359（甚至比 raw 0.376 还低）。新 geom_v2（关注 XYZ 几何分布 + doppler/intensity 统计）→ **0.709**（+88.5% vs raw）
- **GUI 渲染也被波及**：`curation/gui/core/renderers.py::_mmwave_xyz` 沿用球→笛卡尔转换，把已经是笛卡尔的数据再做了一次"假球→笛卡尔"变换，3D 点云视觉错位
- **教训**：
  - **mmwave 真实独立判别力**（sparse point cloud 特征，正确笛卡尔几何特征后）= 0.71 probe val acc（之前 0.51 / 0.38 / 0.36 都是球坐标错读的产物）
  - mmwave + rgb 联合 0.51（错读）→ 实际更高（待 geom_v2 主流程复测）
  - **任何模态的"特征工程"前必须先核实数据语义**——直接读下游用库的官方代码（如 MMFi `mmwave_vae.py`）比凭 value range 推断可靠
  - **GUI 工具的可视化也要核对**——`renderers._mmwave_xyz` 的球→笛卡尔转换在没核对数据语义时被默认接受，造成视觉错位
  - **给后续 agent**：
    - 数据语义：`[x, y, z, doppler, intensity]`，Cartesian 坐标
    - 特征工程：`scripts/probe_mmwave_geom_v2.py::feat_geom_xyz_plus_signal`（134 维，val 0.71）
    - 待修：`extract_mmwave_features` 替换为 geom_v2；`_mmwave_xyz` 改为 `return pts[:, :3].copy()`
    - LESSONS 跟踪：见 STATUS.md 决策层 `[提议]`
