# 0.4.0 求解器、缓存与验收

本次起点为 `02491e68c90582ae02f7b407ca06544cf0bb7e4cd`。开始时本地与远程 cpp 相同，工作树干净。所有生产拟合均由 C++17 完成，R 和 Python 科学计算包用于参考与基准。目标机为 CentOS Stream 9 x86_64/glibc 2.34，GCC 11.5.0、Armadillo 11.4.2、LP64 OpenBLAS 0.3.29，双插槽 Xeon Silver 4216，每插槽 16 个物理核、SMT2。

## 五个可运行的入口

| 阶段 | 使用方式 | 验收对象 |
|---|---|---|
| 基线和日志 | 原版命令，附加 --log-level / --log-file | 先运行原有 11 项 CTest，冻结 A/B/C 输入顺序、参考/全群核、PD、折、搜索、模型与预测 |
| 线性系统 | fit --solver pcg --varcomp fixed --lambda .7 | 密集 A/B/C、多个固定效应、真实 RHS 残差和独立 GLS |
| 方差分量 | fit --varcomp ai-reml 或 em-reml | 同一核心用于折内和最终拟合，解析 Hadamard 夹具、显式 P、多个起点、KKT 与官方 178/20 |
| 特征算子 | prepare 后 fit --cache，显式 --trace hutchinson | 稀疏系谱、阻塞 G/O、条件块认证、共同随机探针、配对误差及 PCG 终检 |
| 交付 | CTest、Slurm 规模验证、无 R 根文件系统中的发布包 | 来源与命令见本文末尾，数值和性能分别记录 |

0.3.0 与显式 **direct + profile-reml** 在相同输入和单线程下分别复跑。在 A/B/C 小例上，共同 model 字段与固定效应、预测、折表、搜索轨迹、ID、修正决策、K_raw、实际 K、选中权重的参考群 K 均相同。三个搜索分别评估 17、19、18 个坐标。旧软件的 profile 检验继续显式指定该组合；另有新默认 AI 路径和随机路径的独立检验。[基线指纹](validation/pcg/frozen.sha256) · [复验清单](validation/pcg/compatibility.tsv)

## 模型和数值约定

训练集只包含本次训练的有表型个体。g=Vu、e=Ve，V=gK+eI。在每个训练系统中求解 Vh=y、VW=X，小系统 C=X′W 使用 Cholesky，并检验条件数。固定效应 b 解 Cb=X′h，r=h−Wb。所有目标按表型顺序输出 u=gK_*T r 和 yhat=Xb+u。P 的应用使用同一个 W 和 C 的分解。固定 λ 则在 H=K+λI 尺度拟合，u=K_*T r、g=(y−Xb)′r/(n−p)、e=λg，绕过残差谱与 logdet。响应按二的幂缩放并移去截距偏移，输出时恢复效应和方差单位。

默认 PCG 容差为 rtol=1e−11、atol=1e−13，最大 2000 步。收敛时重新计算 b−Ax，要求其范数不超过 atol+rtol‖b‖；诊断中的比值 1 表示恰好达到门槛。绝对容差作用于实际求解的 RHS 单位。最终还检查 Vr−(y−Xb) 和 X′r，必要时重新求解去固定效应的 RHS，再作小系统校正。零 RHS 单独处理；非正曲率、非有限量、停滞和未达到残差目标均可使候选失败。Nyström 预条件器按训练顺序取等距地标，通过小矩阵谱构造低秩因子，并对带正对角余量的矩阵使用 Woodbury 求解。它只改变预条件步骤。[CG 的 SPD 适用条件](https://petsc.org/release/manualpages/KSP/KSPCG/)

**精确 AI 与 EM。** 对每个权重和训练掩码只分解一次误差对比谱。设 κ 是 Q₂′KQ₂ 的特征值，z 是旋转后的对比，d=gκ+e，D 的两个谱分量为 κ、1。精确迹为 Σκ/d 和 Σ1/d。误差对比对数似然 ℓc=−Σ(log d+z²/d)/2，省略正态常数；全设计形式的无常数受限对数似然为 ℓc−log|X′X|/2。精确 score 为 Σ Dᵢ(z²/d²−1/d)/2，平均信息为 Σ DᵢDⱼz²/(2d³)。这与 r′Dᵢr、(Dᵢr)′P(Dⱼr) 的公式等价，精确迭代中直接复用谱；线性求解器负责最终 GLS/BLUP。旧 profile_objective 保留自由度尺度；精确最优规模下等于 −2ℓc−(n−p)。fixed 和随机估计的该字段为 NA。

AI 解二维平均信息系统，必要时在非负象限的边上重算自由方向，然后按实际似然回溯。纯 EM 同时使用同一个旧 θ 的两个 score，更新 θᵢ+2θᵢ²sᵢ/n，n 是训练观测数。精确 EM 要求似然在数值精度内单调。默认正初值为训练性状样本方差各半，AI/EM 的默认预算为 200/20000 次。停止结合按响应及核尺度归一的 score 与似然变化。解析计算可行边界的最优尺度，核验原参数的一侧 KKT，并比较可见内部候选的似然。精确 e=0 需要训练 K 通过 Cholesky；g=0 时只保留固定效应。误差对比谱平坦时记录不可辨识并取兼容分解 λ=1。这些局部迭代与有限网格检查不给出任意似然面的全局最优证明。[AI-REML 方法](https://doi.org/10.2307/2533274) · [方差分量 REML](https://pmc.ncbi.nlm.nih.gov/articles/PMC3014363/)

**随机估计。** 探针分量为按 seed、probe 编号和字符串 ID 哈希派生的 ±1。每个训练核和 θ 共享 W、C 和 r。一个探针 RHS 提供两种导数的二次型，平均时把固定效应的低秩随机项替换为其精确迹。小训练集另使用已知 tr K 的零均值线性控制变量。可选的 --trace-rank 使用独立导频向量得到控制子空间，解析减去近似低秩模型的对数和导数二次型波动，再加其已知期望。不同权重用相同基础探针，探针工作区属于单一候选和训练掩码。批内各列独立迭代，共享一次特征块读取。精度不足时按倍数增加探针；达到上限仍不能确认即失格。检查共同探针的配对似然差标准误、尺度化 score 与方差比灵敏度。随机残差方差下限为训练 OLS 残差均方的 1e−9，边界遵守单侧条件。g=0 的 inward score 可从 K 的对角与固定效应空间精确核验。

SLQ 分别计算深度 m 与最多 2m 的准则变化。--probe-solver pcg 在各轮直接求解探针；--probe-solver lanczos 复用同一训练算子在标量移位下的三对角谱测度，对共同近似似然求导。单列设计还可复用设计及响应误差对比测度；多列设计单独求 W、C。拟接受收敛时，以实际 PCG 探针求解重新检查 score，超出门槛即拒绝。默认 full 重正交；大群体另测显式 local 短递推，保留深度翻倍检查与终检。显式 --slq-batch 可将 local 递推的多个探针合并为一次流式矩阵乘，每列独立处理突破和终止。在同一个 θ 上增加探针时可复用 W、C、响应投影和已解探针的两个迹样本。发生负 Ritz 谱、无效曲率、求积误差或 Monte Carlo 精度超预算都报告真实原因。没有跨训练核复用 Krylov 空间。[随机 Lanczos 方差估计](https://pmc.ncbi.nlm.nih.gov/articles/PMC6668092/)

## 缓存与覆盖

prepare 将原文件指纹、拓扑排序的系谱、两层字符串 ID、保留列、均值插补、位点 p/q、中心和缩放写入分版本缓存。清单包含每个数据成员的 SHA-256；fit 打开时重新验证。默认保存归一化行主序 f64；--snp-format 2bit 对二倍体 SNP 使用四种码，缺失剂量在读取时由原全集统计量还原。组学保持 f64，FPKM 的 95% 门槛、缺失与常数列规则沿用原实现。按临时目录完整写出再发布，损坏缓存可从原文件重新 prepare 到新目录。构建目标为小端 x86_64。

系谱通过两次亲本递推实现 A=(I−T)⁻¹D(I−T)⁻ᵀ。Dᵢ=1−(A_ss+A_dd)/4，未知亲本项为零；已知双亲时 Aᵢᵢ=1+A_sd/2。A_sd 按共同祖先路径权重与已有 D 的乘积求和。训练子集可剪枝至其祖先并继续使用原 D。高度近交的深系谱可能有较多路径工作量，准备过程没有全群 N² 系谱数组。G/O 使用 W(W′v) 与已保存的全集标准化信息；非零训练行及目标行按 ID 读取，多个 RHS 使用同一个解码块。auto 可保留小训练特征，stream 逐块扫描，resident 在同覆盖模式下显式预留训练特征空间，仍保留目标预测的流式路径。[系谱关系间接计算](https://doi.org/10.1186/1297-9686-34-4-409)

| 输入/后端 | 拟合及认证范围 | 已测范围 |
|---|---|---|
| 外部 A/G/O TSV、CSV | 沿用 A/B/C 整核 strict 或 nearPD，最终用实际使用的核。direct、PCG、精确和随机估计都可在预算内使用 | 官方 C 的 1407 阶整核、178 训练/20 留出 |
| 原始缓存 A，同覆盖 | 单纯形内的非负 PSD 组合，统一加一次 jitter。fixed PCG 与随机估计无需全群稠密矩阵 | 150000 个体；特征宽度、训练规模、估计方式分别见下表 |
| 原始缓存 B | 对所选群体求 Gss+εI、O 的仿射校准、T、L 和平均对角尺度。处理 β=0。Schur 余项和 T 的正性、β∈[0,1] 给出可用的协方差证书 | 小型独立乘积、对角、任意训练/目标行及 β=0 对照 |
| 原始缓存 C | 汇集与任一实测 G 个体同一系谱连通分量的活动组，在预算内处理原始 A22/A44 和公式 3 六块。其余组与所有测量个体的系谱协方差为零，保留原 A。支持活动组含多个由 G/O 相连的系谱分量 | 150000 系谱 + 72 个基因型 + 36 个组学的模拟；小例逐块对照 |
| 负权重、未获证书、需要投影或 --write-k | 预算内可用 dense 后端按原全群作用域处理。部分活动组需要 nearPD 时不会只投影该组；大型单连通 C 的条件空间超预算时在分配前终止。--write-k 会预检完整文件的磁盘空间 | 官方 C 整群 nearPD；字节级兼容与预算失败用例 |

固定权重先按完整支持集合建核，再限制到训练/目标。搜索先选参考群、限制原始层支持、按每组权重建参考核，各可评分折用自己的 y 和 X 重估；最后检查全群并重拟合。operator 无法认证的候选会写原因；不应把 PCG 本身作为正定证书。已有随机折、时间截断、full_status 和全群回退顺序保留。真正独立留出个体在拟合性状中填 NA。

## 数值和规模实测

4×4 Sylvester Hadamard 无 jitter 数学测试直接得到内部最优 g=2、e=1、λ=.5。不同正初值下 PCG/direct 和纯 AI/EM 均通过。有限差分 score 与显式 P 的平均信息矩阵分别按 1e−9 与 1e−10 检验。另测 profile 参考、零方差两端、平坦谱、非正曲率、零 RHS、近病态设计、PCG 上限和单步 EM 误报；经 CLI 的用例计入 jitter。响应乘以 2 的 ±500 次幂后，fixed、AI、EM 的方差及效应恢复单位后与原值相同；超出 double 方差范围的 ±665 次幂用例均失败并清理输出。[单位变换记录](validation/pcg/units.tsv) 30 个体既有三折夹具上，精确 AI 和纯 EM 均选 (.75,.25)，合格点的单折 RMSE 相对既有 R/GLS 参考的最大差分别为 2.12e−7、2.34e−7。[每点折分数与有效性](validation/pcg/nested-search.tsv) 原始 A/B/C 包含独立换序、多个 RHS 的任意向量、精确对角与 train/cross 对照；全同胞交配产生的近交对角为 1.25。

CLI 的近边界夹具预设 g=.001、e=1，AI 从两个正初值得到 λ≈1000，profile 参考约为 1000.0019；纯 EM 达到 20000 轮预算并报告失败。确定零方差的两端另测解析 KKT。近边界的方差比相对变化与目标值变化应结合解释。[多起点与边界记录](validation/pcg/boundaries.tsv)

官方完整示例固定 α=.2、β=.5，按原表型从首行每十行遮蔽一个 age。R 4.4.1 / Matrix 1.7-0 / rrBLUP 4.6.3 独立重建 C 的原始核再投影。固定 λ=4.39429236354486 时，PCG 对独立 GLS 的最大候选 u/yhat 差为 3.48e−11 / 3.35e−11。精确 AI 得 Vu=41.02977139、Ve=180.29668994、λ=4.394289410；纯 EM 得 41.02977360、180.29668893、4.394289149。两者对 R 最大留出 u 差分别为 4.33e−6、4.71e−6；R 的 λ 为 4.394292362。默认容差下分别约 9、403 个更新。上述全部配置的 20 个候选排名与 R 一致。从官方原始数据经 2-bit prepare、dense C 完整融合到默认预测也通过独立数值门槛。[官方逐配置误差和 RHS](validation/pcg/official.tsv) · [原始数据端到端](validation/pcg/official-raw.json)

随机版使用两种训练量 32/96、三个响应种子 7/31/73，每个数据有 8 个候选，三个独立 probe seed 3/19/43。用随机对比符号及规定的对比幅度构造可识别内部点。以各自精确估计为参考，两种方法各 18 次，失败数均为 0。两者的 |Δlog λ| 第 95 百分位约 0.01053，候选效应 RMSE/精确候选效应标准差第 95 百分位约 0.00377，开发门槛为 0.05 与 0.01。该批次为受控内部点。精度超预算、边界和近并列点单独记录。另以随机二倍体基因型的 Gram 矩阵实测，同样的 6 个响应数据集和 3 个探针种子。控制秩为 12，探针上限 4096 时，两个估计器分别有 9/18 次因随机精度不足被拒绝。保持原误差门槛，将探针上限增至 32768 后均为 0/18 失败，第 95 百分位分别约为 0.01647 与 0.00507。[4096 探针预算](validation/pcg/gram-4096.tsv) · [扩大预算后的复验](validation/pcg/gram-32768.tsv)

调试中修复了 EM 在改变探针数后沿用旧精度指标，以及投影 Lanczos 超出 n−p 后出现虚假负 Ritz 值的问题。重放测试核对串行与权重并行的决策，并单独改变验证真值核验训练估计。[逐种子记录](validation/pcg/stochastic.tsv) · [直接探针 PCG](validation/pcg/stochastic-pcg.tsv) · [移位测度与 PCG 终检](validation/pcg/stochastic-lanczos.tsv)

**规模实验均为模拟原始文件**。生成器依次写出带表头的系谱、SNP、表达量和固定折表。常规宽度为 13 SNP + 9 个组学特征；宽数据实际写出每个个体的 50000 个 SNP。全群预测数包含参考和候选。单性状、一个截距。以下为单次 Slurm 墙钟，含缓存完整性检查、求解和结果发布。单线程以及 1/2/4/8 档的详细记录含 RSS、字节、RHS、matvec、折数、候选与迭代。不同阶段版本的诊断列中，空白表示当时未采集。[规模实测表](validation/pcg/scale.tsv) · [真实文件宽度实验](validation/pcg/wide.tsv)

| N_A / N_G / N_O | SNP / 组学 | 训练 / 全部输出 | 方法与配置 | 秒 | 峰值 RSS KiB |
|---|---:|---:|---|---:|---:|
| 150000 / 150000 / 150000 | 13 / 9 | 120000 / 150000 | 外部 λ=1.5，1 线程 | 3.03 | 158628 |
| 同上 | 13 / 9 | 120000 / 150000 | 随机 AI / EM，1 线程，复用移位测度 | 18.78 / 18.78 | 226080 / 224924 |
| 同上 | 13 / 9 | 120000 / 150000 | 二折选权并重拟合，8 工作者，AI / EM | 100.75 / 103.86 | 822968 / 824816 |
| 150000 / 72 / 36 | 13 / 9 | 120000 / 150000 | 同一随机二折搜索，AI / EM | 18.87 / 18.45 | 792012 / 807116 |
| 10000 / 10000 / 10000 | 50000 / 9 | 8000 / 10000 | f64、stream、固定 λ，1 线程 | 180.63 | 154868 |
| 150000 / 150000 / 150000 | 50000 / 9 | 0 / 准备全群 | prepare 2bit | 526.90 | 108892 |
| 同上 | 50000 / 9 | 128 / 150000 | 定比、早期两遍流式预测；优化后只读非零训练行 | 91.87 / 36.57 | 424136 / 见宽表 |
| 同上 | 50000 / 9 | 128 / 150000 | 精确 AI / EM | 36.68 / 36.51 | 见宽表 |
| 同上 | 50000 / 9 | 128 / 150000 | 精确二折搜索，AI / EM | 40.66 / 64.17 | 471368 / 464700 |
| 同上 | 50000 / 9 | 120000 / 150000 | 8 个 BLAS 线程，训练特征 resident，固定 λ | 299.40 | 47843084 |

大训练集的同覆盖双折搜索也在每一档工作者数完成。表中每格为 AI / EM 秒数，使用 13 SNP + 9 组学，最大数据集的 12 万个参考分成两折，每折实际训练 6 万。全群候选选择只使用成功完成全部可评分折的点；例如 15 万个体的 8 工作者 AI 搜索评估 18 个权重、包含最终拟合在内共尝试 29 次训练，累计 97631 次训练算子应用。试验跨 Slurm 分配，共享节点上壁钟可能随并发数出现回退。

| 全群规模 | 1 工作者 | 2 工作者 | 4 工作者 | 8 工作者 |
|---|---:|---:|---:|---:|
| 10000 | 22.30 / 34.95 | 13.56 / 21.27 | 9.53 / 13.77 | 7.10 / 10.47 |
| 50000 | 76.32 / 83.63 | 46.70 / 51.52 | 43.66 / 44.79 | 24.57 / 26.04 |
| 150000 | 276.25 / 242.88 | 346.23 / 254.46 | 232.28 / 180.57 | 100.75 / 103.86 |

真实宽度的 SNP 文本为 15001690651 B，2-bit G.bin 为 1875000000 B。一次 150000 阶 f64 方阵需要约 167.6 GiB。resident 用 double 保存本折的训练特征（120000×50000 约 44.7 GiB），这次测得进程峰值约 45.6 GiB，缓存读取约 5.28 GB，2 个 RHS、87 次训练算子乘，最大真实残差/门槛为 0.937。相同训练规模在 2 GiB stream、4 线程下 300 s 超时。低维同覆盖大训练集的 16 列 Nyström 在单线程固定 λ 下把 103 次应用变为 143 次（计入建造），时间由 3.03 增到 4.23 s、峰值由 158628 增到 204432 KiB，因此保留为可选项。宽输入、128 个参考的 CV 中，纯 EM 的默认上限未能完成选定权重的全群重拟合；独立增大预算后以 31479 轮收敛。低秩大训练集的部分候选因随机精度不足失格；增加控制变量后完成搜索。在 10000 个体的另一组固定权重实验中，控制秩 24、128 个探针的上限有 8 次精度不足，失败行亦随规模表交付。外层最高 15 个兼容点同时受线程和内存预算限制；当前局部细化每轮最多 4 点。

在这份 **150000×50000 SNP、120000 个参考、30000 个候选** 的同一输入上，随机 AI 和纯 EM 的完整固定权重拟合也已实测。使用 8 线程、resident、16 列 Nyström、4 个共同探针和批宽 4 的 local Lanczos，最终用真实 PCG 探针检查。AI 在 7 个更新后得到 Vu=0.97426027、Ve=1.61946229、λ=1.66224811；EM 在 328 个更新后得到 0.97426042、1.61946220、1.66224776。墙钟分别为 **977.57 s / 857.40 s**，RSS 为 47897056 / 47868576 KiB；分别有 10 / 8 个实际求解 RHS 和 621 / 552 次按列计的训练算子应用。两端均为 approx-interior。30000 个候选中两种方法的最大效应差为 2.08e−7，最大秩差为 1。该对照衡量相同随机准则下两种估计器的一致性。单探针依次构建 SLQ 的 AI 配置在 1200 s 作业预算内未完成。两次试验的批宽和最大探针预算不同。[最宽数据诊断](validation/pcg/wide-panel.tsv)

完整宽数据的权重搜索已验证到 128 个参考和 15 万个输出；120000 个参考的多权重搜索已验证到 13 个 SNP 和 9 个组学特征。120000 参考与 50000 SNP 同时启用全部权重和折的完整搜索仍未验收。可在下面的 wide 命令中将 methods 改为 cv-ai-reml,cv-em-reml，使用 --trace hutchinson --probe-solver lanczos --slq-reorth local --slq-depth 18 --slq-batch 4，并为整个搜索单独配置 Slurm 内存和墙钟。

10k 同覆盖固定 λ 的本地文件页冷提示与复读试验：stream 中位约 1.18 s、逻辑特征读取约 246 MB；允许训练特征驻留时约 0.21 s、约 4.93 MB，另有每次约 2.51 MB 的完整性验证读取。仅对试验私有文件使用 POSIX_FADV_DONTNEED 提示，系统实际物理磁盘缓存不受本程序控制。各阶段计时和解码字节详见原始数据。24 个体拟合中 off/info/debug 五次进程墙钟中位约 9.53/9.48/9.71 ms，分别写入约 0/594/4598 B 日志，预测文件逐字节相同。[I/O 试验](validation/pcg/cache-io.tsv) · [日志试验](validation/pcg/logging.tsv)

## 诊断、限制与重做命令

`pcg_summary.tsv` 给出训练数、真实残差比、投影方程和正交性比值、累计 RHS 和 matvec、估计器、初值、预条件器及建造时间、探针数和种子、迹标准误、配对似然差标准误、深度差、log λ 的 Monte Carlo 标准误与各阶段时间。batch matvec 按参与的列数计，SLQ 计入总数。reml_trace 按实际输出 stride 保留迭代、原单位 g/e、score、lik、步长、回溯、平均信息秩、条件和终止原因。cv_ 前缀用于参考群所有候选；仅合格候选可排名。内存紧时可以放大保留轨迹的步距，同时保留首末轮和失败原因。cache_usage 列出逻辑读取调用、校验与特征字节、解码时间。新选项和直接示例见 README。

全部 14 项 CTest 分别在 Release、独立 ASan+UBSan/LSan、独立 TSan 和 Valgrind Memcheck 下通过。另以阳性和阴性对照验证检测器。包括必需日志打开和 /dev/full flush 失败、cache 损坏、文件大小限额、预算不足、PCG 上限、AI 零回溯预算、纯 EM 上限、线程与文件描述符回收。同进程预热后运行 1000 轮混合成功和失败路径；起末采样均为 4 个 FD、1 个线程，最后 200 轮的活跃分配保持稳定。OpenBLAS 等系统库使用发行版二进制，TSan 的插桩覆盖项目自身与可见同步操作。[测试记录](validation/pcg/debug) · [同进程采样](validation/pcg/lifetime.tsv)

在 cpp/ 目录执行。ARM_INC 是独立 Armadillo 头文件目录。需要 LP64 OpenBLAS/LAPACK 与 OpenSSL 的 libcrypto 开发文件。spdlog 1.15.3 及其内置 fmt 11.2.0 由本地 third_party 归档离线提取，SHA-256 为 15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67，配置时校验。许可随源码和发布包交付。同步 logger 默认 stderr，外部文件日志以追加方式写入，输出目录中的日志随该目录一起发布。文件 sink 的错误处理器保存故障；写入、flush 和日志与结果路径冲突都在发布前核对。[spdlog 错误处理](https://github.com/gabime/spdlog/wiki/Error-handling)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DARMADILLO_INCLUDE_DIR="$ARM_INC" -DBUILD_TESTING=ON
srun -n1 -c4 --mem=6G cmake --build build -j4
srun -n1 -c4 --mem=4G ctest --test-dir build --output-on-failure
Rscript --vanilla tools/nearpd_official.R example-input reference
srun -n1 -c1 --mem=3G python3 tools/pcg_official.py build/fsblup example-input reference check-official
srun -n1 -c1 --mem=3G python3 tools/pcg_accuracy.py build/fsblup stochastic-check lanczos
srun -n1 -c1 --mem=3G python3 tools/pcg_accuracy.py build/fsblup gram-check lanczos --family genotypes --trace-rank 12 --max-probes 32768
srun -n1 -c1 --mem=2G python3 tools/pcg_baseline.py /path/to/0.3.0/fsblup build/fsblup tests compat
srun -n1 -c16 --cpu-bind=cores --mem=6G python3 tools/pcg_scale.py build/fsblup scale --tag fixed --threads 1,2,4,8
srun -n1 -c16 --cpu-bind=cores --mem=6G python3 tools/pcg_scale.py build/fsblup scale --tag small-search --cohorts small --methods ai-reml,em-reml,cv-ai-reml,cv-em-reml
srun -n1 -c16 --cpu-bind=cores --mem=12G python3 tools/pcg_scale.py build/fsblup scale --tag approx-search --cohorts large --methods cv-ai-reml,cv-em-reml --trace hutchinson --trace-rank 24 --max-probes 64 --slq-depth 40 --parallel weights --threads 8 --memory-mb 8192 --timeout 1200
srun -n1 -c16 --cpu-bind=cores --mem=68G python3 tools/pcg_scale.py build/fsblup wide --sizes 150000 --markers 50000 --modes A --cohorts large --snp-format 2bit --storage resident --memory-mb 65536 --threads 8 --tag wide-fixed --timeout 3600
srun -n1 -c16 --cpu-bind=cores --mem=68G python3 tools/pcg_scale.py build/fsblup wide --sizes 150000 --markers 50000 --modes A --cohorts large --snp-format 2bit --storage resident --memory-mb 65536 --threads 8 --methods ai-reml,em-reml --trace hutchinson --probe-solver lanczos --slq-depth 18 --slq-batch 4 --probes 4 --max-probes 8 --prec nystrom --tag wide-panel --timeout 3600
python3 tools/pcg_collect.py consolidated.tsv scale/fixed.tsv scale/small-search.tsv scale/approx-search.tsv
```

生成目录与输出标签应唯一。独立 debug 构建和检测器命令沿用 PERFORMANCE.md 中的 san/tsan/mem 三组参数；对本版全部 CTest 使用同一命令。最终包在无 R 的只读 UBI 9 Minimal 根文件系统（glibc 2.34、网络隔离）完成 build、2-bit prepare、固定与自动权重、实际 nearPD、缓存 EM、随机 AI 及全尺寸官方 1407 阶拟合。发布包随附 libcrypto 和 OpenBLAS 等非 glibc 运行库，另行用关闭网络的 CMake 构建验证本地 spdlog 归档。[干净根文件系统结果](validation/pcg/debug/clean-linux.tsv) · [官方独立误差](validation/pcg/clean-official.json) 以下是带符号的最小 PCG 上限重现和目标环境复验。UBI 根文件系统层的固定摘要见 PERFORMANCE.md。

```bash
d=tests/data
gdb -batch -ex 'catch throw' -ex run -ex 'bt 8' --args build-mem/fsblup fit \
  --A "$d/A.tsv" --G "$d/G.tsv" --O "$d/O.tsv" --pheno "$d/pheno.tsv" \
  --trait trait --alpha .2 --beta .5 --varcomp fixed --lambda .7 --pcg-maxiter 1 --out debug-failure
srun -n1 -c4 --mem=3G tools/debug/controls.sh controls
srun -n1 -c4 --mem=4G python3 tools/debug/memcheck_ctest.py build-mem memcheck-ctest
srun -n1 -c4 --mem=2G build/repeat_test 1000
tools/package.sh build dist
srun -n1 -c8 --mem=4G tools/clean-smoke.sh ubi9-root dist/fsblup-0.4.0-linux-x86_64-glibc2.34.tar.gz clean-0.4 example-input reference/masked.tsv
python3 tools/nearpd_compare.py reference clean-0.4/official
```
