# fsblup 性能与调试验收

## 0.4.3 内部峰、搜索分辨率与工作区

**求解成本。** 双峰夹具上，普通随机 AI 默认起点的总 RHS/matvec 从 522/1705 变为 766/2403，EM 从 2628/8150 变为 6438/19745。新版两者各有 68 个 RHS 和 170 次 matvec 直接用于竞争比较，其余增量包含从较高盆地起点继续迭代。预热后 3 次配对墙钟中位数 AI 为 .0161 / .0158 s、EM 为 .0191 / .0262 s（旧 / 新）。在 64 和 512 个体固定 λ 的同路径 legacy 搜索中，两版分别完成相同的 17、18 点；512 个体用时中位数为 .0516 / .0536 s。[重复与离散程度](validation/v043/modes-medians.tsv) · [固定 λ 配对计量](validation/v043/paired-medians.tsv) · [逐 ID 与独立高峰对照](validation/v043/comparison-per-id.tsv)

**邻域与停止。** 初始 15 点保持原顺序，细化先访问四个对角方向，再访问负/正 α、负/正 β 轴向。在 simplex 上，反向对角构成和为 1 边界的切向，坐标方向允许在另外两条边上移动。全局去重后评估，容差内并列仍保留先访问者。默认只有步长不大于 1/32 且归一化改进小于 10⁻⁴ 才提前停止，最多 10 轮。缺失折的候选不能参加排名。legacy 复现旧几何和停止条件，grid 提供固定候选集合。对 1/32 单纯形的 561 个点独立直接 GLS 枚举后，边界最佳 (.125,0)、内部最佳 (.375,.25) 均由新版公开 fit 达到。四折拟合次数分别由 68→120、76→156；原 RMSE .03098 / .03156，新版 direct 小于 4×10⁻¹⁷、PCG 小于 2.5×10⁻¹³。[网格、成本与权重](validation/v043/comparison-grid.tsv)

**实际 RHS。** 训练、交叉预测与原始特征乘法在收到 RHS 后，为并存的输入、输出、层累加结果、系数和读/解码块计算可行宽度与行块，已驻留的训练特征先扣除。不能容纳一列时在创建系数矩阵前拒绝。80 行、60000 SNP、p=60、stream、probe-batch=1 的执行记录中，最初规划宽度为 1，实际请求 60；在 12 MiB 下实际最大批宽为 6，最大系数缓冲 2880000 B，面板保守估计 7420160 B。128 MiB 对照用 60 列、28800000 B 系数缓冲。两种预算均得到 Vu=Ve=0.5422733921646128，全部 u、yhat 差为 0，最大真实残差/容差均为 .83213348。2 MiB 预检失败。附件完整拟合使用放宽的 PCG 和随机精度参数；本例检验的是相同参数下的预算等价性。0.4.2 在 12 MiB 下也成功退出，但单个系数临时量已达 27.47 MiB。进程峰值 RSS 另含动态库、文件解析和分配器存留，数值不同于工作区上限。[分块与残差](validation/v043/comparison-workspace.tsv) · [预测差](validation/v043/comparison-workspace-pred.tsv)

**C 模式。** 512 个体系谱在不相关、全连通、单个断开的三种布局中，算子实际活动人数分别为 48、512、511。后两例 16 MiB 在活动块检查处拒绝，96 MiB 成功；511 人案的全群 nearPD 回退记录实际 dense、原因和已发现的活动人数。三例逐 ID 与完整 dense 直接构核的最大预测差小于 7×10⁻¹²。另测 1024 个体四代模拟系谱，64 个有基因型、24 个有组学，990 个活动个体，96 训练与 32 候选。16 MiB 拒绝；320 MiB 经全群 dense 回退，最大预测差 3.22×10⁻¹⁵。[512 个体认证](validation/v043/comparison-C.tsv) · [四代系谱](validation/v043/connected-summary.tsv)

### 固定 15 点的重试与留出准确度

同一模拟数据、相同种子与相同额外额度下比较两种策略。768 个体的小面板和 10000 个体的主面板各有 128 个参考，固定两折各 64 训练、64 验证。主面板另外 **9872 个候选**，32 个 SNP、8 个组学特征。根据模拟特征和独立抽取的效应生成真实 u，表型再加入独立高斯噪声，候选表型未传入 fit。使用 AI、4 个初始探针、基础上限 32、控制秩 16、MC 容差 .18、两个权重工作者、256 MiB，主面板重复三次。该模拟面板与 0.4.2 的主面板个体数相同，模拟特征和真值另行生成。

每个搜索批次先完成第一轮评估，再按固定顺序分配重试。rounds 按已评分折数从多到少稳定排序，每次只预留一个缺失折，成功折仅在同一权重、模型、训练和验证 ID、探针种子下复用；预算不足以完成的点保留失败原因和部分折分数，不产生均分。legacy 每次预留整组折，优先在当前候选上继续翻倍。`retry_reserved` 为已承诺额度，`retry_used` 为重试拟合使用的探针前缀数之和，`retry_unused` 为预留后未用的部分；已预留额度不再次分配。discarded_fit_probes 包括失败、被重算替代及最终不完整点的探针工作。RHS 和 matvec 单独计数。

| 主面板种子、额外预算 | 完整评分 legacy / rounds | 折尝试次数 | 实用探针 legacy / rounds | 选中 (α,β) |
|---|---:|---:|---:|---|
| 19，128 | 9 / 9 | 27 / 27 | 68 / 128 | 两者均 (.25,.75) |
| 19，512 | 10 / 11 | 31 / 31 | 296 / 368 | 两者均 (.25,.75) |
| 42，128 | 11 / 11 | 30 / 30 | 80 / 128 | 两者均 (.25,.75) |
| 42，512 | 11 / 11 | 34 / 34 | 352 / 512 | 两者均 (.25,.75) |

分轮策略在一组设置多完成 1 点，另三组持平。相同种子内上述策略及预算的候选预测一致；换种子后最大 |Δu|=.0477371、最大 |Δyhat|=.0480607。按 u 降序排名，完全相同的 u 按原表型 ID 行序破同分；前 k 取比例乘候选数向上取整。跨种子前 1%/5%/10% 名单重合率分别为 100%、99.7976%、99.4939%，次序相关 .999978。与独立留出的真实 u 对照，种子 19 的 Pearson r=.94254、RMSE=.39006；种子 42 为 .94182、.39644。768 个体面板中，种子 19、预算 512 时，策略从 legacy 换为 rounds 会改变权重与名单。覆盖率、重合率与留出准确性分别报告。这些模拟结果不能替代真实育种群体的独立留出验证。[主面板逐策略计量](validation/v043/medium-summary.tsv) · [三次中位数和 MAD](validation/v043/medium-medians.tsv) · [名单重合率](validation/v043/medium-overlap.tsv) · [768 个体面板](validation/v043/panel-summary.tsv)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 tools/v043_compare.py "$OLD_FSBLUP" "$RELEASE_FSBLUP" before after comparison
python3 tools/v043_connected.py "$RELEASE_FSBLUP" connected
python3 tools/v043_validation.py "$RELEASE_FSBLUP" split-check
for rep in 1 2 3; do python3 tools/v043_panel.py "$RELEASE_FSBLUP" "medium-$rep" --n 10000 --cap 32 --mc .18; done
sbatch -n1 -c16 --mem=68G validation/v043/wide.sh "$RELEASE_FSBLUP" runs/pcg/wide150/A-150000-50000-9 new-wide-run
```

## 0.4.2 选权、工作区与缓存

同一份 0.4.1 缓存、同一表型与外部折表用于配对测量。阶段字节统计在实际特征读调用处累加，工作线程各自标记当前阶段。`calibration_bytes` 包括全群准备或物化，`training_preload_bytes` 表示驻留训练行，`training_bytes`、`cross_bytes` 表示求解与预测。四项之和等于 `feature_bytes`；清单完整性校验的字节另列。

| 配置 | 0.4.1 | 0.4.2 |
|---|---:|---:|
| B，256 个体，8 个训练、4 个候选，128 SNP，α=0，β=.4 | 174720 B，0.0180 s | 10368 B，0.0137 s |
| 同上，全群增至 4096 | 2755200 B，0.0790 s | 10368 B，0.0364 s |
| 同上，4096 个体，α=.2 | 2755328 B，0.0780 s | 141696 B，0.0412 s |
| 40 个体、4096 SNP、32 个参考，32 MiB，weights/2 | 启动时拒绝 | 实际 2 个工作者，估计工作区合计 7.74 MiB，进程峰值 15872 KiB |

B 的给定训练与预测路径各有 19 次 matvec。α=0 时，全群仍以缓存中的对角做归一化；α=.2 时，为计算 B 的全群回归与标定读取 131328 B 特征。在两个 α 下训练读取均为 9728 B、交叉预测均为 640 B。4096 这一对旧缓存的 SHA-256 校验均另读 345867 B。256 个体的 dense/direct 给逐 ID 数值参照，重放另覆盖乱序、多个 RHS 与组学覆盖以外的目标。表内时间是预热后 3 次中位数。[配对原始计量](validation/v042/operations.tsv) · [中位数和 MAD](validation/v042/B-medians.tsv) · [32 MiB 工作区复现](validation/v042/after/planning.tsv)

B 的索引化实现使用 `G + 10⁻⁶ I_s + L Δ L′`。L 在组学子集 s 上为单位阵，其他行取原回归系数，Δ 为校准后替换块减去已扰动的 Gss。仅提取当前训练及目标 ID 的 L 行，训练乘法和交叉预测共享已校准的 Δ。C 的活动差值矩阵已由只读共享指针持有。对同一 144 个活动个体的多代模拟 C 模型，Massif 测得含分配器开销的堆峰值分别为 2264168 B 和 2264280 B；本轮保留其捕获方式。[C 分配剖析](validation/v042/C-heap.tsv)

**评分与预算的解释。** RMSE 排序比较原单位分差与 `1e-12 × scale`，细化停止采用 `1e-4 × scale`，全群回退复用同一比较函数。scale 为参考群 score-col 的 RMS；全零时回退到训练表型 RMS，两者都为零则取 1。平方和使用动态二进制幂缩放，阈值比较使用扩展精度，最终可表示的 RMSE 仍以 double 输出。cor、AUC 与 legacy 的比较尺度为 1。认证覆盖、被访问点、完整评分点和实际尝试、累计完成及最后一次评估保留的折数分别解释；失败的折不能贡献候选均分。重试按确定性批次顺序分配全局额度，每次重评该点全部可评分折，按新上限乘折数扣除额度。每个点的 `probe_attempts` 记录上限、原因、RHS、matvec 和各折使用的探针前缀数量；求解上限与内存预算仍生效。对于通过较高上限完成评分的获选点，全群估参沿用其上限。[旧阈值与新规则](validation/v042/legacy.tsv) · [单位和后端交叉验证](validation/v042/scale-grid.tsv)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun -c16 --cpu-bind=cores --mem=4G python3 tools/v042_bench.py "$OLD_FSBLUP" build/fsblup before/reproduce_operators/stream measured
srun -c4 --mem=4G python3 tools/v042_ops.py "$OLD_FSBLUP" build/fsblup before ops
srun -c4 --mem=4G python3 tools/v042_sensitivity.py build/fsblup runs/pcg/scale/A-10000-13-9 budget-replay --max-probes 64 --mc-tol .15 --trace-rank 24
sbatch -n1 -c16 --mem=68G validation/v042/wide-cv.sh "$RELEASE_FSBLUP" runs/pcg/wide150/A-150000-50000-9 new-wide-cv
```

### 同预算的线程对照

单节点分配 16 个逻辑 CPU，通过 taskset 为 1、2、4、8 个权重工作者各固定相应数量的不同物理核，BLAS 保持一线程。两版均使用相同缓存、固定两折、λ=1、stream、单纯形、1024 MiB 算法预算。新版显式使用 legacy 评分策略以配对同一搜索路径。每档预热一次，再交错测量三次；这是共享节点上的墙钟。下表每格依次为 0.4.1 / 0.4.2 秒。

| 模拟数据 | 1 工作者 | 2 工作者 | 4 工作者 | 8 工作者 |
|---|---:|---:|---:|---:|
| 40 个体，32 参考、8 候选，4096 SNP、2 组学特征 | .385 / .387 | .228 / .227 | .159 / .159 | .122 / .122 |
| 320 个体，256 参考、64 候选，256 SNP、16 组学特征 | 1.913 / 1.893 | 1.171 / 1.143 | .769 / .762 | .616 / .614 |

两组分别完整评分 16、24 点，累计 66/789、98/2043 个 RHS/matvec。对应的探针数为零。0.4.2 中等规模的 8 工作者实测 29284 KiB RSS、87601152 B 特征读取，3 次墙钟的 MAD 为 .00263 s。这组端到端收益主要来自多个候选并行；0.4.1 和 0.4.2 在相同并发下的耗时接近。本次规划修改使 32 MiB 流式用例可用，B 验证则直接测到按需读行缩短了端到端耗时。[含 RSS、MAD、分阶段字节和实际评分数的计量](validation/v042/thread-medians.tsv) · [旧新版逐点与逐 ID 差异](validation/v042/thread-equivalence.tsv)

### 探针上限、选权和预测

同一 10000 个体模拟缓存使用 128 个参考、9872 个候选及固定两折。在种子 42 和 19 下分别测试基础探针上限 8、16、32、64；各自比较额外预算 0 和 256，固定控制秩 24、MC 容差 .15、Lanczos 深度 32。全部试验均保留退出状态、折完成数和成本。限额 8 和 16 时出现搜索选出权重、最终全群拟合仍因精度不足而失败的实例；该情形保留失败日志及分项缓存计数。

| 探针上限 64 | 完整评分点 / 访问点 | 所选 (α, β) | 同种子增加预算后的变化 |
|---|---:|---|---|
| 种子 42，额外 0 | 42 / 43 | (.251953125, .126953125) | |
| 种子 42，额外 256 | 34 / 35 | (.1953125, .1328125) | 最大 Δu 绝对值 0.15434，候选最大秩差 561 |
| 种子 19，额外 0 | 22 / 23 | (.3125, .1875) | |
| 种子 19，额外 256 | 31 / 35 | (.2109375, .1328125) | 最大 Δu 绝对值 0.42053，候选最大秩差 994 |

重试允许先前失败的点参与本轮归并，下一轮生成的候选集合可以随之改变。两种种子都使用额外 256 时，共同完成评分的 31 点中最大权重秩差为 1，全部 ID 的最大 Δu 绝对值为 0.10989，候选最大秩差为 156。按 ID 的带符号差保存在原始记录包的 `final/ID-deltas.tsv.gz` 中。[预算扫表](validation/v042/probe-budgets.tsv) · [候选权重排序](validation/v042/candidate-ranks.tsv) · [跨种子预测与排名](validation/v042/probe-ranking.tsv)

### 发布二进制的 150000 个体复验

Slurm 作业 31651 直接执行打包安装后的 x86_64 Linux 程序，ELF SHA-256 为 `413ec36542dc70773ec182b74bbd7f3daebd01f9292c8bdd31d48b1bf6c7731c`。沿用 0.4.1 的 50000 SNP、9 个组学特征、相同的缓存清单、表型与外部折表。120000 参考、30000 候选，两折各 60000 训练和 60000 验证。节点分配 16 个逻辑 CPU 与 68 GiB，程序设置 8 个 BLAS 线程、resident、65536 MiB、随机 AI、探针种子 42、初始 4 个探针、基础上限 8、Lanczos 深度 18 和批宽 4。新版使用 relative RMSE，评分尺度为 1.57814427，另给整个搜索 32 个重试额度。[输入与程序指纹](validation/v042/wide-inputs.sha256) · [节点与亲和性](validation/v042/wide-environment.txt)

访问初始 15 点和细化 4 点。认证覆盖为 complete，10 点完成两折，9 点耗尽探针预算，无其他数值失败。总计尝试 30 折、成功 20 折。按原遍历顺序重试 (0,0)，为两折预留 16×2；该点首折在 8 个探针时用 48 次 matvec，上限提高至 16 后再用 72 次仍未通过精度要求。该点保留失败身份，均分为空。0.4.1 对照共尝试 29 折。两版都在有效候选中选中 (.5,.25)，0.4.2 的两折 RMSE 为 1.51193056 和 1.50686756，均分 1.50939906，搜索阶段 7822.15 s。细化改进低于停止阈值，选中点通过全群认证。[逐点分数、预算与成本](validation/v042/wide-search.tsv) · [覆盖和完成数](validation/v042/wide-tuning.tsv)

全群重新估参成功，状态为 approx-interior，AI 更新 7 次，Vu=2.0324701262、Ve=1.3858328276、λ=0.6818465914。与上次作业的 150000 个 ID、角色、u、yhat 逐行比较，全部一致，包含 30000 个候选。0.4.2 的端到端墙钟为 **8727.19 s（2:25:27）**，峰值 RSS **47998512 KiB（45.78 GiB）**。累计 210 个 RHS、13344 次按列计的 matvec，各次折尝试和最终拟合的探针前缀数之和为 172。特征读取 35037000000 B，包括训练预载 19599360000 B、交叉预测 15437640000 B；训练特征已驻留，训练乘法无额外文件读取。另读 1901519503 B 校验缓存。0.4.1 同一数据的墙钟为 9312.95 s，峰值 RSS 为 47992728 KiB；两次宽作业各测一次，节点负载、重试及评分规则均影响耗时，配对速度比较见上方三次重复表。[两版完整计量](validation/v042/wide.tsv) · [最终模型](validation/v042/wide-model.tsv) · [逐折求解成本](validation/v042/wide-fold-fits.tsv) · [逐 ID 差值汇总](validation/v042/wide-predictions.tsv)

原始记录压缩包包含冻结基线的五份重放、最终各轮输入输出、每次性能采样与 CPU 亲和性、探针预算面板、`final/ID-deltas.tsv.gz`、检测器日志、C 堆快照，以及宽搜索和参考版本的 15 万行预测。开发作业 31204 的中止日志单列在 debug 目录。上表仅计入正式程序从头完成的作业 31651。宽缓存由清单指纹定位，公开的模拟生成流程见 PCG.md。

```bash
python3 tools/v042_collect.py runs/v042/final summary --wide-reference runs/v041/wide-cv
python3 tools/v042_archive.py runs/v042 runs/v041/wide-cv dist/fsblup-0.4.2-validation.tar.gz
```

## 0.4.1 缓存、交叉预测与选权

以下为同一 Slurm 节点、同一输入缓存与 ID、单线程的一次配对实测。表内 RSS 为整个进程的峰值，包含动态库和分配器。内存预算约束算法工作区。完整表另列训练与候选数、保留 SNP、后端、搜索域、预算、RHS、matvec、特征与校验字节以及墙钟。[逐次实测](validation/v041/performance.tsv)

| 配置 | 0.4.0 | 0.4.1 |
|---|---|---|
| 160 个体，24 个参考、4 个候选；历史搜索域，固定两折，200 / 4 MiB | dense 选 (−.125,−.125)，operator 选 (0,0) | 两个后端均选 (−.125,−.125)，逐候选全群核按预算认证。200 / 4 MiB 墙钟 0.025 / 0.041 s |
| 同数据，0.8 MiB | — | 全群签名权重超预算，回退到 (0,0)，明确报告 partial。全群未能认证的候选保留原因 |
| 10 个体，8 个参考，追加 199999 个全缺失 SNP | 峰值 RSS 从 13312 增至 69520 KiB | 小表和大表均为 13312 KiB；预测逐字节一致。大表墙钟从 0.193 降至 0.126 s |
| 4096 个体、128 SNP，8 个参考、4 个候选，外部 λ=1 | 131584 B 特征，0.056 s | 384 B 特征，另读 450512 B 做完整性校验，0.052 s。逐 ID 与独立重建的 12 行稠密直接解最大差 2.78×10⁻¹⁶ |
| 模拟多代 C 模式，144 个体、96 SNP、96 个参考、48 个候选 | 0.043 s，89 matvec | 0.043 s，59 matvec。包含组学个体的活动连通集有 144 个个体。全群 K 经 2 轮投影，修正量相对加扰动输入为 28.9482% |

历史域共访问 23 组候选，两后端的逐折及均分最大差为 2.22×10⁻¹⁶。独立从原始 SNP、组学列重建的全群选中核最小特征值为 +0.19262324。[折分数和全群谱](validation/v041/domain.tsv)

规划提及的 +0.44259 和 10.42% 对应审查包中的输入，本次重建夹具保留权重分歧、正定性以及两轮投影现象。仓库 C 模式另一例修正量为 0.999665%，`pd.tsv`、`fusion.tsv` 与日志均标出全群 K 的投影动作。`cv_fold_fit_sum` 在 4 MiB 搜索中由 0 变为 0.00811 s。固定单纯形、串行和权重并行的折对照由 pcg_cli 验证。[预算不足时的逐点诊断](validation/v041/search.tsv) · [模拟系谱投影](validation/v041/connected-pd.tsv) · [交叉预测误差](validation/v041/cross-errors.tsv)

在另一个 1024 个体、多代共同祖先的模拟系谱上，用同一 0.4.1 可执行文件分别读 v2 与 v3。三次固定拟合的中位墙钟由 0.205 降至 0.0244 s，预测文件相同，均读 48 B 特征；每次多校验约 54.8 kB 拓扑数据。[拓扑缓存对照](validation/v041/topology.tsv)

缓存 v3 中 `topology.tsv` 按亲本先于后代的顺序保存 ID、双亲下标、D 及 Aii，加载时验证清单 SHA-256、顺序、数量和 D 与亲本对角的关系。v1/v2 重算拓扑。上述配对使用两版均可读取的同一 v2 缓存，校验字节计入完整清单中的数据文件。选权覆盖更完整时可能读取更多数据，4 MiB 搜索的耗时也相应增加。宽度实验、域对照及 1000 轮同进程回归的复现命令见[验证记录](VALIDATION.md)。

### 12 万参考、5 万 SNP 的独立两折作业

Slurm 作业 31078 使用现有 150000 个体、50000 SNP、9 个组学特征的模拟原始文件缓存。表型有 120000 个参考与 30000 个候选。两折各含 60000 个训练与 60000 个验证个体，按折独立估计方差。分配 16 个逻辑 CPU、68 GiB 内存，程序使用 8 个 BLAS 线程、一名选权工作者、65536 MiB 预算、resident 训练特征、PCG 与随机 AI。探针初值为 4、上限为 8，Lanczos 深度 18，Nyström 秩 16。输入清单、外部折表与长作业冻结的 0.4.1 二进制指纹随记录保存；之后的发布构建补充了诊断、工作区与显式极端初值检查。[输入和二进制指纹](validation/v041/wide-inputs.sha256)

初始网格 15 点，细化 4 点，搜索阶段 8355.63 s。19 点全部进入评估，10 点完成两折评分，另外 9 点在首折耗尽探针预算。程序记录了 29 次折拟合。单纯形请求域在该后端得到完整认证；可用于比较的候选数为 10。选中 (0.5, 0.25)，两折 RMSE 为 1.51193056、1.50686756，均分 1.50939906。细化改进小于 10⁻⁴ 后停止，全群核认证通过。[逐候选折分数与失败原因](validation/v041/wide-search.tsv) · [逐折求解诊断](validation/v041/wide-cv_pcg_summary.tsv)

全群模型重新使用 120000 个训练表型拟合，随机 AI 更新 7 次，返回 approx-interior，Vu=2.03247013、Ve=1.38583283、λ=0.68184659，30000 个候选全部输出。完整搜索加最终拟合的墙钟为 **9312.95 s**，峰值 RSS **47992728 KiB（45.77 GiB）**。累计 210 个求解 RHS、13272 次按列计的训练算子应用，特征读取 35037000000 B，另读 1901519503 B 校验缓存。折拟合累计 8349.60 s，全群拟合阶段 949.64 s。上述权重比较限于完成评分的 10 点。[完整计量](validation/v041/wide-summary.tsv) · [阶段耗时](validation/v041/wide-timing.tsv) · [最终模型](validation/v041/wide-model.tsv)

最终发布二进制用相同缓存、选中权重和 λ=0.68184659135829162 再执行一次固定比值 PCG 拟合。150000 个 ID 和角色逐行一致，u、yhat 最大绝对差分别为 1.7921×10⁻¹⁰、1.7919×10⁻¹⁰，墙钟 210.41 s。该项对照覆盖宽数据的发布版预测。官方示例的独立 GLS 对照见 VALIDATION.md。[逐 ID 差值汇总](validation/v041/release-wide-predictions.tsv) · [发布版重拟合](validation/v041/wide-release-model.tsv)

```bash
sbatch -n1 -c16 --mem=68G validation/v041/wide-cv.sh build/fsblup \
  runs/pcg/wide150/A-150000-50000-9 new-wide-cv
```

## 0.3.0 性能与调试验收

0.4.0 的 PCG、AI/EM、随机迹误差、特征缓存与 Slurm 大群体记录见 [PCG 开发与验收报告](PCG.md)。下文保留之前发布版的实测基线。

2026-11-26 在目标 Slurm 节点执行。公开起点为 `d3ba237ec5da3c19ac7fb7dd3684dcaf43fbe92c`，本地与远程 cpp 相同，工作树干净。校验附件清单后，仅应用一次通过 `git apply --check` 的主补丁（`fd818a7`）。全量对称 Gram 候选保存在独立试验提交 `71e7206`；生产实现根据实测尺寸选择计算路径。`7f85dd6` 引入权重并行和计算优化，`15aca56` 修复有限矩阵的极端值回环问题。[环境与提交记录](validation/perf/environment.txt) · [输入 SHA-256](validation/perf/inputs.sha256)

## 哪些改动节省了时间

官方输入为 A=1407、G=198、O=74，SNP 与表达量分别有 5000 个原始特征。下表为**单线程端到端墙钟秒**。两个 A/B 对照各自固定亲和性、文件与输出选项，预热一次，轮换运行五次取中位数。两批试验的主补丁时间分别列出，以便按批配对。节点使用共享 XFS，文本格式化和写入页缓存均计时。每次 CLI 关闭输出流、检查错误后发布结果目录。计时不含将页缓存强制刷到存储设备的耗时。

| 场景 | 公开 0.2 → 主补丁 | 主补丁 → 本轮 |
|---|---:|---:|
| 官方原始 build，写出 A/G/O | 1.3955 → 0.4213 | 0.4096 → 0.3742 |
| 官方固定 .2/.5，nearPD，写出 K_raw/K | 6.7693 → 4.7895 | 4.7899 → 4.4475 |
| 同一固定拟合，仅输出预测与诊断 | 4.1632 → 4.1647 | 4.1756 → 3.8200 |

主补丁将矩阵文本写出改为每行复用缓冲、17 位有效数字的 `to_chars`。这项收益集中在写大矩阵的任务。本轮 nearPD 融合时将保留的谱分量按列乘以特征值平方根，再做对称 rank-k 重构；连续保留的特征向量列只复制一次。Dykstra 更新、一次 jitter、严格相对谱阈值、无穷范数收敛、最终特征值下限、对角缩放与 Cholesky 检查沿用 0.2 的规则。配对试验中官方写核的峰值从 221468 降到 206352 KiB。[公开版对照](validation/perf/baseline-medians.tsv) · [构核补丁试验](validation/perf/build-variants-medians.tsv) · [本轮配对计时](validation/perf/final-paired-medians.tsv)

**原始 G/O。** G 连续扫描每列并计数，只遍历一次来写入中心化剂量；非整数均值插补仍不计入整数基因型频数。O 使用连续列视图，顺序处理缺失、过滤与标准化。两层均按输入列次序串行压缩，保留计数由单个线程更新。输入 198×5000 的官方 G 采用 DSYRK，74×5000 的 O 保留 DGEMM 路径。切换条件为至少 96 个样本且至少 32 个保留特征，是本版经验阈值。ltrace 核验主补丁依次调用 DGEMM/DGEMM，独立试验为 DSYRK/DSYRK，本轮为 DSYRK/DGEMM。[实际调用](validation/perf/blas-calls.tsv) 独立试验的官方 G 阶段从 129.0 降到 125.8 ms，O 阶段从 68.9 增到 69.7 ms，均含文本读入。下面单独测 LP64 BLAS 乘法；DSYRK 包含补全另一三角，分块宽度为 256。每种形状五次中位数，单位为 ms。[逐次 Gram 计时与矩阵误差](validation/perf/gram.tsv)

| 样本×特征 | DGEMM | 整块 DSYRK | 分块 DSYRK |
|---|---:|---:|---:|
| 24×17 | 0.0010 | 0.0030 | 0.0030 |
| 74×5000 | 3.017 | 2.542 | 2.610 |
| 198×5000 | 17.762 | 12.180 | 12.183 |
| 513×1025 | 23.343 | 13.863 | 14.238 |

整块 rank-k 在较大形状节省乘法时间，分块方案保留在基准工具中。官方 build 的总时间还包含读入、系谱递推和文本写出。生产目标使用通用 x86-64 编译选项、OpenBLAS 自带 CPU 调度与编译器常规向量化。GCC 报告列循环使用 16 字节向量，存在别名检查与标量余部。97×33、9 行与边界过滤用例覆盖非整倍数输入；另外构建关闭编译器向量化的二进制，并在同一台机器上指定 OpenBLAS NEHALEM 后端完成测试。未使用全局 fast-math。[编译器报告](validation/perf/vectorized.txt)

## 线程扩展

Xeon Silver 4216，2 个插槽，每个 16 个物理核，SMT2。一次 Slurm 分配包含 16 个逻辑 CPU，工具读取 sysfs 的插槽与 core_id，仅使用每个物理核的第一个可用逻辑 CPU。每个线程数始终绑定相同的 CPU 前缀。五次正式运行按轮换、反序交错安排，前置的第 0 轮计为预热。下表为中位数秒；逐配置的实测范围、峰值 RSS（KiB）、输出字节数、CPU 列表、实际 OpenBLAS 设置及权重工作者数均已记录。小程序的启动、共享库和分配器也计入进程 RSS。[逐配置完整结果](validation/perf/final-suite-medians.tsv) · [逐次结果](validation/perf/final-suite-runs.tsv)

| 工作负载 | 并行层 | 1 | 2 | 4 | 8 |
|---|---|---:|---:|---:|---:|
| 完整原始 build | BLAS | 0.3743 | 0.3730 | 0.3750 | 0.4058 |
| 同尺寸仅需 jitter，写核 | BLAS | 1.0572 | 1.0278 | 1.0251 | 1.0068 |
| 官方 .2/.5 实际投影，写核 | BLAS | 4.5244 | 3.1100 | 2.4091 | 2.1886 |
| 官方 .2/.5 实际投影，关闭写核 | BLAS | 3.8677 | 2.4705 | 1.7561 | 1.5083 |
| 30 个体，自动选权 | BLAS | 0.0257 | 0.0256 | 0.0256 | 0.2287 |
| 30 个体，自动选权 | 权重 | 0.0257 | 0.0172 | 0.0119 | 0.0109 |
| 192 个体，自动选权 | BLAS | 0.5171 | 0.5135 | 0.5117 | 0.9281 |
| 192 个体，自动选权 | 权重 | 0.5167 | 0.3257 | 0.2189 | 0.1933 |

**阶段计时。** 单线程官方 build 的 SNP 读取、预处理、Gram 分别为 69.8、5.89、11.89 ms；表达量分别为 64.1、1.61、1.71 ms。系谱阶段包含系谱文本读取和 A 递推。官方投影并写核时，1/8 线程的输出文件总量中位数分别为 82388839/82388080 B，峰值 RSS 为 206008/218604 KiB。单线程读核 0.326 s、nearPD 特征分解 2.881 s、重构 0.392 s、写核 0.657 s、最终 REML/BLUP 0.0075 s。8 线程对应的特征分解和重构为 0.890 和 0.091 s。显式整矩阵及保留列复制计时分别约 0.0180 和 0.0198 s，此项是其它计时的子集。nearPD_eigen 包含最终报告所需的最小特征值分解。cv_fold_fit_sum 等字段是各候选的耗时之和，并行时可大于 search 阶段墙钟。perf 用户态 cycles 抽样 382 次，最高频符号为 OpenBLAS 的 dgemm_kernel_SKYLAKEX，约 41%；LAPACK 分解内部也会调用该内核。[采样](validation/perf/profile-top.txt) · [计数器](validation/perf/debug/perf-stat.txt)

**并发与内存。** 默认 `--threads` 继续控制 OpenBLAS。显式 `--parallel weights` 在进入搜索前将 BLAS 固定为 1；主线程参与计算。初始网格和每一轮细化分别去重、分配独立结果槽、等待全部任务，然后按串行次序归并。并列容差 1e−12、停止阈值 1e−4，与全群回退顺序沿用基线。C 模式在构造 FusionPlan 时完成条件块分解并保留准备错误；工作线程通过 const 接口共享计划，各候选独享 K、谱分解、求解器和诊断数据。失败原因保存在该候选自己的槽位。

外层并行的稠密工作区估算中，共享部分为完整三层输入、每个参考个体平方 64 字节、设计矩阵及 1 MiB 预留；每个工作者按每个参考个体平方 192 字节、设计空间及 9 MiB BLAS/线程缓冲预留。候选总数最多 15，细化轮最多 4。22 MiB 回归用例将 8 工作者请求降为 2，5 MiB 用例明确失败。估算用于选择并发数，Slurm 限制整个进程。中等数据固定种子 4127，A/G/O=192/128/64，177 个参考，三种覆盖组各 59 个，固定三折。8 个权重工作者的峰值 RSS 为 53444 KiB，单工作者为 21304 KiB。该路径为显式选项。8 线程内层 BLAS 在小样本上离散度较大，30 个体范围为 0.0260–0.2471 s，中等数据为 0.5101–3.8978 s。在另一组固定 CPU 上做同条件交错复测，30 个体 8 线程内层 BLAS 的主补丁中位数为 0.185 s（0.054–0.247），本轮为 0.227 s（0.168–0.245）。两者均在搜索与最终融合阶段出现明显波动；12 次小样本拟合的 8 线程 cycles 抽样中，6.71% 直接落入 OpenBLAS 的 exec_blas_async_wait，大部分其它热点也位于 BLAS 库内。该现象在主补丁上也可复现。外层调参让单次小矩阵操作固定使用 BLAS=1。[小样本复测](validation/perf/check8-nested.tsv) · [中等规模复测](validation/perf/check8-medium.tsv)

**完整官方 15 点固定权重。** 每轮启动 15 个独立 fit，使用相同的 178/20 遮蔽文件，关闭写核。表中时间为一整轮各进程墙钟之和，峰值 RSS 为轮内最大值。四档线程的所有权重均完成两次迭代的实际投影。该表衡量固定权重的全群构核与拟合开销；自动选权的折表与结果见上面的嵌套搜索。

| BLAS 线程 | 中位 s | 五次范围 s | 加速比 | 峰值 RSS 中位 KiB | 输出字节中位数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 57.969 | 57.799–58.039 | 1.00 | 206224 | 183200 |
| 2 | 37.019 | 36.829–37.130 | 1.57 | 208744 | 183114 |
| 4 | 26.350 | 26.277–28.863 | 2.20 | 210956 | 183098 |
| 8 | 23.860 | 22.573–26.289 | 2.43 | 219624 | 183257 |

[逐权重、逐轮分阶段数据](validation/perf/grid-weight-runs.tsv) · [逐点修正幅度](validation/perf/grid-audit.tsv) · [网格汇总](validation/perf/grid-medians.tsv)

## 严格调试和数值回归

| 复现输入 | 根因与处理 | 回归 |
|---|---|---|
| 4×4 核，忽略 SIGXFSZ，RLIMIT_FSIZE=96 B | 公开版在最终缓冲写失败时返回 0，原本 347 B 的 K.tsv 只保留 96 B，最后一行截断。主补丁显式 close、传播文件名和错误；发布前移除失败临时目录 | output_failure 分别限制元数据、矩阵与预测文件，并验证现有空目录 |
| 相同文件限额，SIGXFSZ 保持 Linux 默认处置 | 额外实测发现，已应用主补丁时进程仍可因信号 25 被终止，留下一个临时目录。CLI 入口现在让文件限额作为 EFBIG 进入受检写出路径；同用例退出 1，保留的临时目录数为 0 | output_failure 同时覆盖继承忽略和默认信号处置 |
| 性状乘 1e200 / 1e−200 | 公开版成功写出 NaN 方差 / 零方差和负无穷目标值。主补丁在稳定尺度拟合，恢复单位时检验表示范围，两次复现均返回 1 | solver_range 验证可表示尺度及固定/自动拟合失败路径 |
| 核心收到空折列表 | 公开版在读取 folds.front() 时产生 UBSan 空引用。主补丁先拒绝空集合 | core 检查候选记录的原因 |
| 最小非零 double、接近最大值的方阵、负零和含引号的超长 ID | stod 的 ERANGE 会拒绝可表示的次正规数；对相等的大数先相加再除以二会溢出。补偿解析保留非零可表示的次正规数，极大矩阵采用安全对称平均 | io_limits 检查 12000 字符以上的带转义 ID、符号位、原始核和重读结果 |
| 多线程首次构核、单任务失败、整轮失败、争用输出目录 | C 计划改为预构造；每轮等待工作者结束，按预先分配的槽位归并，成功目录整体发布 | concurrent_replay、repeated_lifetime、numerical_cli 覆盖预算降级、并列、异常展开、全群回退和临时目录回收 |

Release、`-O1 -g` 的 ASan+UBSan、独立 TSan 均运行全部 11 项 CTest。ASan 开启 LSan，UBSan 禁止恢复，TSan 遇错停止。阳性泄漏对照故意遗失 79 B，LSan 和 Memcheck 分别以 1、99 退出；竞态对照由 TSan 报告两个工作线程中对同一内存的冲突访问并以 66 退出。对应阴性对照均为 0。系统 OpenBLAS/LAPACK 是发布库，TSan 观测范围包括被插桩的项目代码与其可见同步操作。[控制对照](validation/perf/debug/controls.tsv) · [Release](validation/perf/debug/Release.txt) · [ASan+UBSan](validation/perf/debug/ASan-UBSan.txt) · [TSan](validation/perf/debug/TSan.txt)

Memcheck 开启完整泄漏、全部泄漏类型、来源跟踪以及错误退出码 99。对 raw build、SPD fuse、固定 fit、实际 nearPD、strict 拒绝、外层搜索、短写和同进程混合循环逐项核对应用退出码。8 条路径 errors、definitely/indirectly/possibly lost、still reachable 全部为 0。应用拒绝与短写的预期退出码为 1。另将全部 11 项 CTest 调用的原生程序逐个置于 Memcheck 下，11/11 通过。首轮 32 B 短写测试触发了 Valgrind 自己的 vgdb 共享文件限制（48 B），关闭调试器通信 `--vgdb=no` 后按原测试限额重跑整套通过。[Memcheck 全套记录](validation/perf/debug/Memcheck-CTest.txt) [Memcheck 各项字节数](validation/perf/debug/mem-final.tsv)

单进程预热 6 轮，再运行 1000 轮原始 G/O、投影、BLUP、成功/失败搜索和异常写文件。每 10 轮记录 glibc 活跃分配量（uordblks+hblkhd）、smaps_rollup 的 RSS/PSS、FD 和任务数。该次运行中活跃分配从 117184 B 到 121680 B，最后 200 轮各采样点相同；RSS 从 18432 到 18456 KiB，最后 200 轮 PSS 为 12146 KiB。每次取样均为 4 FD、1 个线程。进程正常退出后 Memcheck 报告所有堆块释放。RSS/PSS 含共享库映射及分配器保留页。异常退出测试覆盖 C++ 异常和 I/O 错误；操作系统强行终止进程时可能留下带 PID 的临时目录。[生命周期记录](validation/perf/repeated.tsv)

**独立数值对照。** R 4.4.1、Matrix 1.7-0、rrBLUP 4.6.3 重新构造完整官方核及 178/20 的固定遮蔽参考。1、2、4、8 线程逐一核验。四档中最大的修正 K 元素差为 5.98e−14、相对 Frobenius 差 5.24e−14；u/yhat 最大差分别为 5.92e−6 / 5.66e−6。同 R 的 λ 下显式 GLS 最大差 <3e−13。Vu、Ve、λ 和截距最大差分别为 3.22e−5、2.46e−5、4.04e−6、2.74e−7。四档均投影两次迭代后收敛，以 S=K_raw+1e−6 I 为基准的相对修正约为 8.55089%。20 个候选按 u 排序与 R 完全一致。

原始 A/G/O 对归档数据按两轴 ID 比较的最大绝对差分别为 5.11e−15、1.33e−14、8.44e−15。嵌套 30 个体的两种并行层、全部线程数均选 (.75,.25)，18 个候选的三折与均值对独立 R 的最大差为 3.26e−8。外层工作者与 BLAS=1 串行路径的输出核、预测、模型、折表和搜索轨迹逐字节相同。小矩阵测试含 FPKM 恰好 95%、常数列、50% 缺失、非整数插补与 p/q 计数、A22/A44 奇异、重根及谱阈值处的特征值。[逐线程验证](validation/perf/thread-validation.tsv) · [1 线程官方详细对照](validation/perf/official-1.json) · [8 线程](validation/perf/official-8.json)

## 复现

在 cpp 目录按 README 安装独立 Armadillo 头文件，使用带 LAPACK 的 LP64 OpenBLAS。下面假设 `ARM_INC` 指向包含 armadillo 文件的目录，`d` 指向经 SHA-256 核验的官方解包目录。工作目录使用新名字。日常 CTest 仅需 Python 标准库，额外的性能数据生成和独立 R 比较使用 NumPy/pandas。

```bash
export ARM_INC="$PWD/.deps/usr/include" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
d=example-input
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DARMADILLO_INCLUDE_DIR="$ARM_INC" -DBUILD_TESTING=ON
srun -n1 -c4 --mem=6G cmake --build build -j4
srun -n1 -c4 --mem=3G ctest --test-dir build --output-on-failure
Rscript --vanilla tests/nearpd_generate.R tests/nearpd_cases.tsv
Rscript --vanilla tools/nearpd_official.R "$d" reference
srun -n1 -c16 --cpu-bind=cores --mem=6G python3 tools/nearpd_bench.py build/fsblup "$d" reference/masked.tsv bench-suite --scenario suite --parallel both --threads 1,2,4,8
srun -n1 -c16 --cpu-bind=cores --mem=6G python3 tools/nearpd_bench.py build/fsblup "$d" reference/masked.tsv bench-grid --scenario grid --threads 1,2,4,8
srun -n1 -c16 --cpu-bind=cores --mem=5G python3 tools/verify_threads.py build/fsblup "$d" reference verify
# 配对比较可在同一 bench 命令追加 --variant patch=/absolute/path/to/patched/fsblup
g++ -O3 -std=c++17 -I/usr/include/openblas tools/gram_bench.cpp -lopenblasp -o /tmp/fsblup-gram-bench
srun -n1 -c1 --cpu-bind=cores /tmp/fsblup-gram-bench
```

```bash
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 TSAN_OPTIONS=halt_on_error=1
for tool in san tsan mem; do
  cxx=clang++
  case "$tool" in
    san) flags="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all" ;;
    tsan) flags="-O1 -g -fno-omit-frame-pointer -fsanitize=thread" ;;
    mem) cxx=g++; flags="-O1 -g -gdwarf-4 -fno-omit-frame-pointer" ;;
  esac
  cmake -S . -B "build-$tool" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="$cxx" \
    "-DCMAKE_CXX_FLAGS_DEBUG=$flags" -DARMADILLO_INCLUDE_DIR="$ARM_INC" -DBUILD_TESTING=ON
  srun -n1 -c4 --mem=8G cmake --build "build-$tool" -j4
  srun -n1 -c4 --mem=4G ctest --test-dir "build-$tool" --output-on-failure
done
srun -n1 -c4 --mem=2G tools/debug/controls.sh detector-controls
srun -n1 -c4 --mem=4G python3 tools/debug/memcheck.py build-mem memcheck-cases
srun -n1 -c4 --mem=4G python3 tools/debug/memcheck_ctest.py build-mem memcheck-ctest
srun -n1 -c4 --mem=2G build/repeat_test 1000
```

在目标 CentOS Stream 9（glibc 2.34）打包。用 Red Hat 官方 UBI 9 Minimal 的只读根文件系统隔离宿主 R、库和网络；根文件系统 glibc 同为 2.34。冒烟测试覆盖 build、固定 fit、折表搜索、真实不定核的 nearPD 解析解，以及 4 个权重工作者的嵌套拟合和 .75/.25 权重。共享宿主内核。发布包还在该根文件系统内按完整官方 1407×1407 输入执行固定拟合，再在外部按 ID 和 R 对照，修正 K 最大差 5.98e−14，u/yhat 最大差 4.99e−6 / 4.78e−6。[干净环境数值记录](validation/perf/clean-vs-R.json) 文件大小信号策略在计时器启动前设置。解包命令和镜像层内容摘要固定如下。

```bash
layer=42dd82cfea990337e85629f4800f64d42e5fca052db254cd3386d99051476958
curl -fL "https://registry.access.redhat.com/v2/ubi9/ubi-minimal/blobs/sha256:$layer" -o ubi9-layer.tar.gz
echo "$layer  ubi9-layer.tar.gz" | sha256sum -c -
mkdir ubi9-root
tar -xzf ubi9-layer.tar.gz -C ubi9-root --no-same-owner --exclude='dev/*'
tools/package.sh build dist
srun -n1 -c8 --cpu-bind=cores --mem=2G tools/clean-smoke.sh ubi9-root \
  dist/fsblup-0.3.0-linux-x86_64-glibc2.34.tar.gz clean-0.3.0 "$d" reference/masked.tsv
python3 tools/nearpd_compare.py reference clean-0.3.0/official
```
