# 数值与发布验收记录

## 0.4.1 数值回归

2026-11-27 开工时本地与远程 cpp 均为 `f21497f82830e01a465fa6b539b89f825a63284a`，工作树干净。旧版 14 项 CTest 全部通过。工作区没有 `FSBLUP_0.4.0_cpp_audit_and_replay.zip`，解析例按规划给定谱重建，其余使用仓库原始数据或种子固定的模拟数据。[旧版基线](validation/v041/baseline.log)

| 复验 | 0.4.0 现象与原因 | 0.4.1 实测 |
|---|---|---|
| 8 个参考，1 个候选，表型增加 10⁶x | 用原始表型尺度初始化，绝对梯度过小，两种方法第 0 轮均返回 Vu=Ve=571428571432 | AI 的 Vu/Ve 为 1.99999999996 / 1.00000000006，EM 为 1.99999999910 / 1.00000000244。两者 ℓ=−8.90701477885 |
| 4 个参考，请求 Vu=Ve=10¹² | 两种方法原地收敛，ℓ=−43.14713 | 保持初始比值，实际均从 1.65555556 起步，AI 4 次、EM 1962 次更新，ℓ=−3.95263739 |
| 平移、缩放、调换固定列顺序 | 协变量加 20000 或乘 10⁸使 GLS 法方程报病态 | 按训练折平衡设计并还原系数，检查了训练和候选逐 ID 的 u/yhat、表型增加 Xc 后的系数和平移量 |
| 双峰 | 单一起点停在 λ≈0.74738943、ℓ=−8.50633779 | 校验剖面局部区间与单侧 KKT 边界，AI/EM 最终 λ≈0.005105209、ℓ=−7.41302154006。选择后的更新分别为 3/50 |
| 2-bit 元数据、稀疏预测、不同预算的选权 | 整表加载、全群特征回读、证书限制改变可评分点 | 逐行解码字典、按输出 ID 做交叉乘法、参考与全群逐点认证，见性能表 |
| 固定权重 PCG 超限 | 工作目录清理时丢失内部日志和轨迹 | 测试核对退出码、stderr、目录内日志、失败摘要及成功模型文件缺席。磁盘短写和 /dev/full flush 另有测试 |
| 大尺度组学与小差值 | 未缩放平方和可溢出；在大偏移上直接更新均值会损失小差值 | build/prepare 共用偏移、二进制缩放与标准化。24 个体的 2i 和 10²⁰⁰i 核差 <10⁻¹²，预测差 <10⁻¹⁰。三组值的 SD 与 √(64/23) 相差 <10⁻¹⁴ |

解析例使用不含 −df·log(2π)/2 的正交误差对比似然，输入核在 CLI 扰动前减去 10⁻⁶I。EM 保留自身的方差更新式和迭代上限。精确内点同时检查无量纲 score、似然变化及 |yᵀPy/df−1|。对数比值的有限网格用于发现可分辨的竞争峰，不能证明任意连续谱上已找到全部峰。随机迹在两种训练规模、三组数据种子、三个探针种子和两种估计器的 36 项试验中全部完成，最大 |log λ 误差| 为 0.01053，最大相对预测 RMSE 为 0.00766。[旧版逐例输出](validation/v041/before.tsv) · [修复结果](validation/v041/after.tsv) · [随机迹](validation/v041/samples.tsv)

官方模式 C 用 R 4.4.1、Matrix 1.7-0 和 rrBLUP 4.6.3 重新计算，顺序与遮蔽保持一致。1407 个体核在一次扰动后投影 2 轮，投影量相对扰动后输入为 8.5508902542%。C++ 修正核相对 R 修正核的最大元素差为 5.98×10⁻¹⁴，相对 Frobenius 差为 5.23×10⁻¹⁴。178 个训练和 20 个留出个体的全体最大 u/yhat 差分别为 4.33×10⁻⁶ / 4.14×10⁻⁶（AI）和 4.71×10⁻⁶ / 4.50×10⁻⁶（EM）。20 个候选的排序全部一致。独立 GLS 使用相同 C++ λ 时最大差为 5.46×10⁻¹¹。[矩阵和分量误差](validation/v041/official.json) · [五种方法](validation/v041/official.tsv)

在 cpp 目录运行以下命令。旧版可从指定基线单独构建，复验脚本使用新目录。官方例重建需要上述 R 包，运行交付的 fsblup 二进制只需 Linux。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DARMADILLO_INCLUDE_DIR="$PWD/.deps/include"
srun -n1 -c4 --mem=8G cmake --build build -j4
srun -n1 -c4 --mem=6G ctest --test-dir build --output-on-failure
srun -n1 -c4 --mem=6G bash validation/v041/replay.sh build/fsblup /path/to/0.4.0/fsblup new-results /path/to/official-example
srun -n1 -c4 --mem=4G env OPENBLAS_NUM_THREADS=1 build/repeat_test 1000
```

0.4.0 的 PCG、AI/EM、随机迹误差、特征缓存与 Slurm 大群体记录见 [PCG 开发与验收报告](PCG.md)。下文保留之前发布版的实测基线。

0.3.0 已以 d3ba237 为基线复验官方 178/20 模型、候选排名和嵌套逐折评分，并完成并行、内存和 glibc 2.34 验收。[0.3.0 性能与严格调试报告](PERFORMANCE.md)

2026-11-26，基线 `babc8ea221664412de7deca3cdb72833d61f018a`。开发开始时上游 main 与该提交相同。主实现和参考来自仓库 `R/FSBLUP.R`，用户输入类型依据 README.md。DESCRIPTION 声明 GPL (>=3)。`renv.lock` 固定 R 4.4.1、rrBLUP 4.6.3、AGHmatrix 2.1.4、Matrix 1.7-0。已在独立目录构建 R 4.4.1 与 Matrix 1.7-0，生成脚本启动时检查四个实际版本。`R CMD build` 和 `R CMD INSTALL` 均已通过。科学来源为论文公式 3、7、Materials and methods 及补充推导。[基线源码](https://github.com/Xueyh1/FSBLUP/blob/babc8ea221664412de7deca3cdb72833d61f018a/R/FSBLUP.R) · [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12888590/) · [Additional file 3](https://pmc-oa-opendata.s3.amazonaws.com/PMC12888590.1/13059_2026_3931_MOESM3_ESM.pdf) · [论文归档](https://doi.org/10.5281/zenodo.17760713)

## 覆盖与数值定义

**全覆盖。** 三个集合相同，按 A 的 ID 排列。原始融合矩阵为 (1−α−β)A+βG+αO。先检验实质非对称，容差为 1e-10 × max(1,max|Kij|)。容差内两轴取平均。默认 strict 对融合原矩阵先试 Cholesky；失败时计算最小特征值，低于 −64 ε n max(1,max|λᵢ|) 即判无效。允许合法半正定与舍入级负特征值，保留 `raw` 并在交给求解器的 K 上统一加 1e-6 I，无投影。`fusion.tsv` 记录分块与全局扰动。R 的 `adj_pos`（206–215 行）先加扰动再尝试 Cholesky，还可走 nearPD。

**组学子集 B。** 对应 `BB000`（256–291 行）。S 为有 O 的个体，U 为其余个体。默认通过 Cholesky 的块用 B=G_SS+1e-6 I；nearPD 策略触发投影时用修正后的 B。令 d 为平均对角线，m 为全矩阵元素均值。斜率 w=(m(B)−d(B))/(m(O)−d(O))，截距 v=d(O)−w d(B)。校准量 wO+v 的截距作用于所有元素。T=α(wO+v)+(1−α)B。用三角求解得到 L=G_US B⁻¹。填 H_SS=T、H_US=LT、H_UU=G_UU+L(T−B)L′，另一块按转置填。K_raw=(1−β)A+βH/d(H)。α=0 时直接取 T=B。固定 α=.2、β=.5 的 `B_base/B_cal/B_T/B_H/B_K` 记录逐步参考。子集核夹具 Ob 在原始 O 子块上加单位阵以构造可识别、稳定的校准用例。测试还对参考子集重算 B_Kref，验证它与全体先融合再截取确有差异。

**严格嵌套 C。** 公式 3 的第 3 组为空。1=A\\G，2=G\\O，4=O，全用原始 A 的相应行列。令 C=αO₄₄+βG₄₄+(1−α−β)A₄₄，Rᵢⱼ=Aᵢⱼ Aⱼⱼ⁻¹，用分解求解。以下六块与其转置完全定义生产矩阵，融合后做同样的协方差有效性检查。

| 块 | K_raw |
|---|---|
| 11 | A₁₁+R₁₄(C−A₄₄)R₁₄′ |
| 12 | R₁₂G₂₂ |
| 14 | R₁₄C |
| 22 | G₂₂ |
| 24 | R₂₄C |
| 44 | C |

G₂₂ 与 G₄₄ 来自原始基因组矩阵，O₄₄ 来自原始组学。公式 3 的这些块没有 BB000 的仿射校准或平均对角线归一化。α=0 消去共同观测块中的 O，组 2 保留 G₂₂。R 脚本独立逐元素构建 6 个体参考。固定 A/G/ID，改变 O 的一个对角和一个非对角元素，核的 11/14/24/44 块按公式响应，12 与 22 保持原值，α=0 时两次结果相同。`C_legacy.tsv` 保存旧 `CC000` 的输出，生成时严格验证相同扰动使旧版输出逐元素完全相同。R 第 315、317 行覆盖 M3，第 336 行再读取覆盖结果。

覆盖由完整输入的集合决定。组 1 为空且有组 2 时采用已验证的 B 规则。仅组 4 时采用全覆盖规则。G=O 且为 A 的真子集、O 不属于 G 的个体、A 外个体、空层均不开放融合。调参限制到参考群后若丢失区分原覆盖所需的组，报告无法辨识权重。固定权重产生实质不定协方差时返回错误。自动调参依参考群评分和既定并列规则逐个检查全体融合，直至找到可行权重或已评估的合格点耗尽。后一种情形返回错误，并保留各点的全群诊断。可另行准备已核实的共同覆盖矩阵开展同组分析，这会改变分析群体。

**原始核。** A 按亲本拓扑序递推。G 遵循 AGHmatrix 2.1.4 的 diploid VanRaden 默认分支与 Mcheck。保留缺失比例 ≤0.5 的位点，在位点内用观察剂量均值插补。插补后 p=(2n₂+n₁)/(2n)，q=(2n₀+n₁)/(2n)，整数基因型按完全相等计数。非整数插补值进入中心化剂量，不计入这些基因型计数，因此 q 可与 1−p 不同。分母为 2Σpq。所有位点均被过滤或分母为零时显式报错。O 先补零，FPKM 再删低于 0.1 的样本占比至少 95% 的列，去常数列，用样本标准差标准化，每层全部样本共同参与，O=WW′/保留列数。`Omatrix`（684–705 行）中负空索引和单列降维在 C++ 端按实际保留集合处理。测试覆盖零列需过滤、恰好 95%、仅余一列、全部丢弃。

**REML 与权重。** 每个权重和训练掩码独立拟合。在 X_T 的正交补 Q 中，令 d 为 Q′K_TT Q 的特征值，z 为按其特征向量变换的 Q′y。目标为 df log[Σ z²/(d+λ)/df]+Σ log(d+λ)，df=n_T−rank(X_T)。这是省去与 λ 无关常数的负二倍限制对数似然。对 λ∈[1e−9,1e9] 在对数轴取 64 个区间，细化采样到的局部极小点并比较两端；整段不可辨时报告状态并取 λ=1。最后用 Cholesky 白化求 GLS，u=K_*T (K_TT+λI)⁻¹(y_T−X_T b)，yhat=Xb+u。固定 λ 的核心入口也用于直接代数验收。

初始网格每轴 0、.25、.5、.75、1，单纯形内共 15 点。先按 α 再按 β 升序，差 ≤1e−12 保留最先评估点。每次从目前最佳点评估 (α±h,β±h) 四角，初始 h=.125，每轮减半，默认允许越出单纯形，总轴向位移小于 .25。`--strict-simplex` 全程限制。去重，始终保留最好合格点，改善小于 1e−4 或达到 10 轮停止。手算测试在 (.375,.375) 的平方距离上达到零。R 的 GridSearch（59 行起）走整个方格，Bisection（119 行起）步长和保优策略不同，顶层 constrain_weights 没有透传；这套确定性局部细化也是本项目对论文算法的具体取舍。

旧 `cv.cal_ab` 的 K 按表型对齐被注释（约 450 行），`tv.cal_ab` 在 529 行附近显式对齐；最终调用在 830 行遗漏 X。行 465/539/830 的 `tryCatch(error={...})` 在此 R 版本会先求值处理器表达式，对成功输入也会多求解一次、额外加扰动。脚本记录实际两次调用并验证两次所用 K 一致，存为 `legacy_solver_K.tsv`。固定 K、截距、共同 ID 顺序用于该历史对照。修复后的固定效应使用明确传 X 的 mixed.solve 参考。另用 R 原入口执行 25 个初始点及最多两轮细化，保存每个已评估点的分数和最终 u（legacy_search、legacy_search_u）。R 用按重复编号设种子的有放回折标签；跨语言比较共享提交的折表。R 用 which.max 选择 RMSE，旧 AUC 对并列依赖顺序；C++ 分别采用最小化与平均秩。

## 数值验收

在 cpp 目录执行 `ctest --test-dir build -V`。日常测试直接读取 `tests/data`。重新生成时在上述锁定版本的库路径下执行 `OPENBLAS_NUM_THREADS=1 Rscript --vanilla tests/generate.R`。脚本另存 LL 及其省略常数后的目标用于核对。以下“最大误差”为绝对值，混合判据为 atol+rtol×|参考值|。

| 输入与比较层级 | 最大误差或结果 | 坐标 / 判据 |
|---|---:|---|
| 24 个体原始 A/G/O | 0 / 4.89e−15 / 5.11e−15 | G 020,020；O 013,013；atol 1e−8，rtol 1e−7 |
| 全覆盖 α=.2 β=.5 原始 / 最终 K | 6.88e−15 / 7.11e−15 | 008,008；三个端点最大 2.23e−16 |
| B 最终 K 与 R CombineMat | 1.82e−12 | 008,007；atol 1e−8，rtol 1e−7 |
| C 原矩阵与公式 3 的独立参考 | 3.33e−15 | 001,001；另测组学扰动与零权重 |
| λ=.7，截距、年龄与 herd，b/u 对显式 GLS | u 3.67e−15 | 017；b 和 u 绝对容差 1e−8 |
| 20 个参考、4 个 NA，REML 的 u/yhat | 3.26e−6 | 016/u；atol=rtol=1e−5 |
| 同例 Vu、Ve、λ 对 mixed.solve | 1.74e−6 / 1.61e−6 / 6.64e−6 | R 的 λ=1.3847765965；目标值差 <1e−8 |
| 固定 4 折、15 个候选、60 个 RMSE | 7.80e−5 | α=.5 β=0，折 2；对 rrBLUP 绝对容差 1e−4 |
| 同折独立 Cholesky 限制似然核查 | RMSE 差 <1e−8 | 直接解 λ=1e−9，RMSE 1.2405674795 |
| 固定折表端到端搜索 | α=0，β=.25，调参 RMSE=1.4289268061 | 次优合格点 1.4509591279，随机折导出后回读轨迹完全一致 |

最大逐折差来自边界。rrBLUP 在 λ 原尺度用 R optimize 默认精度，得到 λ=6.8200e−5、目标 8.446344705；独立 R Cholesky 参考在下界的目标为 8.446198844。测试分别比较软件基线和直接解，避免将边界优化精度当作 ID/评分误差。额外覆盖包括两轴及表型独立置换、前导零、秩退化、AUC 并列与单类折、非法覆盖、半正定端点、不定核、内存限额、时间验证和无有效分数。固定真值的不可评分折对所有权重共同排除。

## nearPD 0.2.0 验收

2026-11-26 在本机 Slurm 节点复验。初始 HEAD 和远程 cpp 均为 `cff637f2da8c50f7d072dddca10344cf0bb7e4cd`，工作树干净。附件 SHA-256 为 `1718d693cb3a738b1c2873779585a674db6324534337bb043298125d1b4d55c9`，校验包内清单，执行 `git apply --check` 后仅应用 from_cff637f。下列结果来自本机重新生成的参考和实际运行。

### 实现与覆盖

CRAN Matrix 1.7-0 的 R/nearPD.R 使用 Dykstra 残差 R=Y−D，在 PSD 投影后以 X−R 更新 D。C++ 按严格大于 eig.tol×最大特征值筛选谱分量；以 ‖Y−X‖∞ / ‖Y‖∞ 为相对变化，矩阵无穷范数按最大绝对行和计算，该比值不超过 conv.tol 时收敛。阈值分别为 1e−6、1e−7，最多 100 次。最后以 posd.tol=1e−8 乘投影后最大特征值的绝对值设下限，按 PSD 投影后的对角元素（至少为该下限）对结果做对角缩放。CLI 采用协方差选项 corr=FALSE、keepDiag=FALSE。移植文件保留原作者、GPL-2.0-or-later 许可和源码位置。[Matrix 1.7-0 原始实现](https://github.com/cran/Matrix/blob/1.7-0/R/nearPD.R)

默认 strict 检查原核。显式 nearPD 仅在加过一次 1e−6 I 的块 S 仍无法 Cholesky 时投影。结果必须有限、收敛且再次通过 Cholesky。B 中参与传播的 Gss 和最终 K 分别处理。C 的 A22/A44 使用原始系谱条件块，分解失败报告块名。参考和全群各自持有与 ID 顺序绑定的融合计划，C 的索引、条件回归系数及与权重无关的 12 块可跨权重复用；最终选择的全群核直接交给 fit。每个权重在参考群中只融合一次，各折分别在自己的训练集估计方差比。

Release 用 GCC 11.5，ASan+UBSan 用 Clang 22.1.8，开启 leak detection 和遇错停止。两种构建的 6 项离线 CTest 均通过。12 个核心用例以本机 Matrix 1.7-0 重建，涵盖迭代上限、非对称、缩放、谱阈值与对角约束。CLI 用例还检查无需投影的单次扰动、无正谱、全群才需投影的选权、B 的 Gss、C 的奇异条件块和按 ID 对齐的独立 GLS。

### 官方完整示例

样本集合为 A=1407、G=198、O=74。R 4.4.1、Matrix 1.7-0、rrBLUP 4.6.3；R 与 C++ 使用 OpenBLAS 0.3.29 单线程。R 按生产模式 C 的六个分块独立重建原始核，先加 1e−6 I，再调用 Matrix::nearPD。预测比较使用输出核中相同的 198 个表型 ID，与上游旧 CC000 的算法区别见前文。

固定 α=.2、β=.5 时，原始最小特征值 −4.02844463312647，修正后为 2.45676193e−7。投影收敛于第 2 次迭代。以 S 为基准，‖P−S‖F 为 6.682462755887，相对修正为 **8.550890254%**，最大元素改变量 0.226128530225。计入对角扰动、直接与 K_raw 比较时，相对量为 8.550896333%，最大元素改变量 0.226129530225。[分块诊断](validation/nearpd/official-pd.tsv)

| 独立 R 与 C++ 的比较量 | 实测差异 |
|---|---:|
| 原始 K 最大绝对差 | 6.66e−16 |
| 修正 K 最大逐元素绝对差 | 5.94e−14（X805，X810） |
| 修正 K 相对 Frobenius 差，以 R 修正 K 为分母 | 5.24e−14 |
| 全部 198 个体 max abs Δu / Δyhat | 4.868e−6 / 4.654e−6 |
| 20 候选 max abs Δu / Δyhat | 4.868e−6 / 4.643e−6 |
| 固定 R 的 λ，在 C++ 核独立解 GLS，对 R 的 u / yhat | 2.86e−13 / 2.87e−13 |
| 固定 C++ 的 λ，独立 GLS 对 CLI 的 u/yhat，二者最大值 | 1.51e−14 |

| 参数 | C++ | R | 绝对差 |
|---|---:|---:|---:|
| Vu | 41.02977435 | 41.02974791 | 2.6434e−5 |
| Ve | 180.29668768 | 180.29670786 | 2.0180e−5 |
| λ | 4.394289039 | 4.394292362 | 3.3229e−6 |
| 截距 | 1.679963962 | 1.679963737 | 2.2533e−7 |

按原表型顺序从第 1 行开始每 10 行遮蔽 age，保留 178 个参考和 20 个候选。另将这 20 个体的 yhat 与被遮蔽的原始 age 比较，RMSE=11.56783、Pearson=0.19580；这是该次固定留出的描述性表现。[完整数值对照](validation/nearpd/official-comparison.json)

**15 个固定权重。** 在完整官方矩阵上单独拟合初始单纯形各点，15/15 均投影且在第 2 次迭代收敛。相对修正量范围为 8.55294%（.25,.5）至 8.83149%（0,0）。每个点的原始/修正最小特征值、绝对/相对 Frobenius 量及最大元素变化均已保存。原始 age 自动选权时参考群 A=198、G=198、O=74，缺少仅有系谱的组，因而报告覆盖退化。固定 .2/.5 的 strict 路径也已单独复验，拒绝最小特征值约 −4.028445 的完整核。[逐点修正表](validation/nearpd/official-grid.tsv) · [官方 age 调参诊断](validation/nearpd/official-age-auto.txt)

**可识别嵌套搜索。** 另一套 30 个体的合成输入保留 27 个参考，其中仅系谱、有系谱和基因型、三层共同观测三组分别为 7、8、12 个体。固定 3 折，三个源核的两轴分别换序。R 对参考子集重建 K、投影并用显式 GLS 剖面似然逐折优化；另在全群重建选定核。C++ 和 R 均选出 (.75,.25)。C++ 三折 RMSE 约 1.178610、1.387564、1.269387，均值 **1.278520593**。15 个初始点及 3 个细化点的各折及均值对 R 的最大差 1.58e−8，选定全群 K 最大差 1.51e−14。score 列均为参考群调参分数。这个结果与官方全群固定权重的修正表分别报告。[逐折搜索与全群检查](validation/nearpd/nested-search.tsv) · [独立 R 参考](tests/nested/scores.tsv)

### 性能复核

Slurm 单任务，Intel Xeon Silver 4216，LP64 OpenBLAS pthread 0.3.29，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 --threads 1`。每个场景先预热一次，取后三次的中位数。表中阶段及总量为 fsblup 记录的墙钟秒，总量含其余元数据输出，RSS 单位为 KiB。完整快速路径使用相同 ID 和维度，G/O 替换为原始 A 的相应主子矩阵。投影路径使用原始官方 G/O。15 权重为 15 个独立固定 fit 的阶段时间之和，各批取最大进程 RSS；此处仅输出预测和诊断。

| 版本与场景 | 读取 s | 搜索 s | 最终融合 s | 写核 s | REML/BLUP s | 总 s | 峰值 KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 补丁原版，同维度仅扰动 | 0.32922 | — | 0.12988 | 2.5441 | 0.0069538 | 3.0063 | 80440 |
| 补丁原版，完整官方，固定 .2/.5 | 0.32885 | — | 3.8797 | 2.5967 | 0.0073802 | 6.8125 | 218756 |
| 补丁原版，完整官方，15 个固定权重 | 4.8585 | — | 57.564 | — | 0.11441 | 62.579 | 218996 |
| 补丁原版，30 个体，嵌套搜索 | 0.00088554 | 0.024174 | 0.0012522 | 0.0015468 | 0.00034784 | 0.028627 | 11456 |
| 复用后，同维度仅扰动 | 0.33133 | — | 0.13172 | 2.5562 | 0.0069599 | 3.0298 | 82784 |
| 复用后，完整官方，固定 .2/.5 | 0.33082 | — | 3.8956 | 2.6354 | 0.0074046 | 6.8705 | 221320 |
| 复用后，完整官方，15 个固定权重 | 4.938 | — | 58.046 | — | 0.1148 | 63.144 | 221308 |
| 复用后，30 个体，嵌套搜索 | 0.0008757 | 0.021279 | 0.0012568 | 0.0017488 | 0.00034737 | 0.026037 | 11452 |

复用仅发生在同一个进程、同一 ID 集合。30 个体搜索阶段的中位数从 0.02417 s 到 0.02128 s。完整官方单次 fit 的时间主要在谱分解和 TSV 输出。保留条件块使固定 .2/.5 的峰值 RSS 增加 2,564 KiB；单次总时间中位数由 6.81 s 变为 6.87 s。计时来自共享计算节点。复用前后使用相同输入、相同折表和独立 R 参考；官方 .2/.5 的 K_raw、K、模型和预测文件逐字节相同。已通过 Cholesky 的块跳过特征分解，参考群候选跳过仅用于最终诊断的修正后最小特征值求解。[预热与各次实测](validation/nearpd/timings.tsv)

### 复现命令

在 cpp 目录使用 README 的 Release 构建命令。下面的 R 解释器须能加载指定版本的 Matrix 与 rrBLUP。官方对照脚本另外使用 NumPy 和 pandas。运行目录须为新目录。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ \
  -DARMADILLO_INCLUDE_DIR="$PWD/.deps/usr/include" -DBUILD_TESTING=ON \
  '-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all'
srun -n1 -c4 --mem=8G cmake --build build-san -j4
srun -n1 -c1 --mem=3G ctest --test-dir build-san --output-on-failure
srun -n1 -c1 --mem=2G ctest --test-dir build --output-on-failure

Rscript --vanilla tests/nearpd_generate.R tests/nearpd_cases.tsv
Rscript --vanilla tests/nearpd_nested.R tests/nested
d=example-input
srun -n1 -c1 --mem=3G Rscript --vanilla tools/nearpd_official.R "$d" nearpd-R
args=(--A "$d/test_matA.csv" --G "$d/test_matG.csv" --O "$d/test_matO.csv" --id id --trait age)
srun -n1 -c1 --mem=3G build/fsblup fit "${args[@]}" --pheno nearpd-R/masked.tsv \
  --alpha .2 --beta .5 --pd-policy nearPD --threads 1 --write-k --out nearpd-fixed
python3 tools/nearpd_compare.py nearpd-R nearpd-fixed
for scenario in official grid nested; do
  srun -n1 -c1 --mem=3G python3 tools/nearpd_bench.py build/fsblup "$d" nearpd-R/masked.tsv \
    "bench-$scenario" --scenario "$scenario" --nested tests/nested
done
# 此诊断命令按预期以非零状态退出，并写出 failure.txt。
build/fsblup fit "${args[@]}" --pheno "$d/test_pheno.csv" --pd-policy nearPD --out age-auto
```

## 0.1.0 基线的实际数据与性能

上游 example.zip 的 SHA-256 为 `c43dde9f91297b346928865e1d88fbdc731e623542b03b2e0f25e8b0a0bdafd6`。实际数据行×数值/特征列（均另有 ID）：A 1407×1407，G 198×198，O 74×74，组学 74×5000，SNP 198×5000，系谱 1407×2，表型 198×3（id、age、bf、czs）。矩阵两轴实际都使用 X 前缀字符串。完整原始文件重新构核后，按两轴 ID 对齐归档矩阵，A/G/O 的最大绝对差分别为 5.11e−15（X1393,X1393）、1.32e−14（X1328,X1313）、8.44e−15（X1292,X1292）。[示例数据](https://github.com/Xueyh1/FSBLUP/releases/download/example/example.zip)

2026-11-26 Slurm 单任务、1 CPU，Intel Xeon Silver 4216，CentOS Stream 9，GCC 11.5，OpenBLAS 0.3.29。构核用原始完整文件。固定权重对比用 74 个共同观测 ID 的相同预构 TSV 和同一性状 age，按组学文件顺序每隔 10 行遮蔽一个，66 个参考、8 个候选。R 与 C++ 都限制为 1 线程，R 用此机串行 OpenBLAS，C++ 用 pthread 构建。时间为单次计时；小数据的初始化与文本 I/O 占比较大。

| 运行 | 读取或构核 s | 融合 s | REML/BLUP s | 搜索 s | 墙钟 s | 峰值 RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| C++ 完整原始 A/G/O | .019 / .243 / .174 | — | — | — | 1.617（写入 1.180） | 33772 |
| R 同输入固定 .2/.5 | .031 | .371 | .006 | — | .409 | 268844 |
| C++ 同输入固定 .2/.5 | .021 | .00062 | .00200 | — | .0245 | 11696 |
| C++ 同输入默认 5 折×2 | .021 | .00023 | .00152 | .163 | .186 | 11808 |

R 的 .409 s 从重读与 C++ 相同的 TSV 算起，进程总耗时 3.18 s 含载包、准备子集与写输入，RSS 为进程峰值。C++ 命令的外部耗时分别 .03 s 与 .19 s。固定权重的 u 对 R 最大差 6.05e−12（X1305），λ 在上界。完整嵌套归档矩阵在 .2/.5 时最小特征值约 −4.028445。对同一例，全部 15 个初始点的最小特征值落在 −4.095 至 −3.986，strict 策略拒绝不定协方差。默认 age 调参还会因参考群中缺少仅有系谱的个体而提前终止。74 个体结果对应三层共同覆盖的子群。

```bash
# 取上游示例并解包到 example-input，然后在 cpp 目录执行
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun -n1 -c1 --mem=3G build/fsblup build --ped example-input/test_pedi.csv \
  --snp example-input/test_snp.csv --omics example-input/test_omic.csv \
  --omics-type fpkm --out example-kernels
srun -n1 -c1 --mem=3G /usr/bin/time -v Rscript --vanilla tools/benchmark.R example-input bench
args=(--A bench/A.tsv --G bench/G.tsv --O bench/O.tsv --pheno bench/pheno.tsv --id id --trait age --threads 1)
srun -n1 -c1 --mem=3G /usr/bin/time -v build/fsblup fit "${args[@]}" --alpha .2 --beta .5 --out bench/fixed
srun -n1 -c1 --mem=3G /usr/bin/time -v build/fsblup fit "${args[@]}" --folds 5 --repeats 2 --seed 42 --out bench/auto
```

## 0.2.0 发布复验

构建与打包命令见 README。解包后 `bin/fsblup` 通过相对 RPATH 找到 `lib/` 下的 libopenblasp.so.0、libstdc++.so.6、libgcc_s.so.1、libgfortran.so.5、libquadmath.so.0。系统提供动态加载器、libc 和 libm。包中附有 `linked-libraries.txt`。使用 Ubuntu Base 22.04.5 amd64 的独立只读根文件系统，Bubblewrap 隔离宿主库、R、环境变量与网络，运行 build、固定权重 fit、带搜索的 fit 和真实不定的二个体 nearPD fuse。脚本检查 K 的四个元素与二阶解析解，并检查 pd.tsv 的投影动作、两次迭代、收敛标记与正的最小特征值。根文件系统的 glibc 是 2.35，发布基线主机的 glibc 是 2.34。

```bash
curl -fLO https://cdimage.ubuntu.com/ubuntu-base/releases/22.04/release/ubuntu-base-22.04.5-base-amd64.tar.gz
echo '242cd8898b33ea806ef5f13b1076ed7c76f9f989d18384452f7166692438ff1a  ubuntu-base-22.04.5-base-amd64.tar.gz' | sha256sum -c -
mkdir ubuntu-base
tar -xzf ubuntu-base-22.04.5-base-amd64.tar.gz -C ubuntu-base --exclude='dev/*'
srun -n1 -c1 --mem=2G tools/clean-smoke.sh ubuntu-base dist/fsblup-0.2.0-linux-x86_64-glibc2.34.tar.gz clean-results
```

此测试在隔离环境内逐 ID 核验全部 24 行固定权重 u/yhat 与 R 参考的绝对误差 ≤1e−5，并检验候选数为 4。Slurm 共享宿主内核。当前支持密集单核单性状。任意非嵌套覆盖、多组学任意多核、重复记录、PLINK 原生格式、APY 和矩阵自由解法属于后续范围。

### 0.4.1 运行时与发布复验

150000 个体、50000 SNP 的两折作业和发布版同 λ 预测校验已完成，逐候选结果含探针预算耗尽的 9 个点，见[宽数据实测](PERFORMANCE.md)及[原始折分数](validation/v041/wide-search.tsv)。

Release、Clang ASan+UBSan（开启 LSan）及独立 TSan 以及 Valgrind Memcheck 均通过 16 项 CTest。独立 Memcheck 寿命进程退出时直接、间接和可能丢失字节均为 0，仍可达 165818 B。检测器阳性对照分别命中 79 字节泄漏与数据竞争，阴性对照正常退出。寿命测试在同一进程反复加载缓存、执行真实的并行折拟合、PCG、精确 AI/EM、投影和错误返回。预热后 1000 轮起末均为 4 个 FD、1 个线程，RSS 从 21332 到 23432 KiB，最后 200 轮各采样点的活跃分配均为 367648 B。TSan 首次启动工作线程会创建一个运行时线程，基线采样前先启动并 join 一个空工作线程，使其在整个测试中受到同一计数规则的检查。[寿命采样](validation/v041/lifetime.tsv) · [阳性及阴性对照](validation/v041/detectors.tsv) · [组学列统计](validation/v041/omics.tsv)

CentOS Stream 9 x86_64 构建的程序已随所需动态库打包，并在无 R 的 UBI 9 Minimal 独立根文件系统中运行。复验包括解析 nearPD、4 工作者嵌套选权、v3 缓存 prepare/fit 和官方模式 C。官方例的 K、u、yhat 在隔离环境输出后重新与 R 比较。[隔离环境官方误差](validation/v041/clean-official.json)

```bash
bash tools/package.sh build dist
srun -n1 -c4 --mem=5G bash tools/clean-smoke.sh "$UBI_ROOT" \
  dist/fsblup-0.4.1-linux-x86_64-glibc2.34.tar.gz new-ubi-results "$OFFICIAL_EXAMPLE" "$R_REFERENCE/masked.tsv"
```
