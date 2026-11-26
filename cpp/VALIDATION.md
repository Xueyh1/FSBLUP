# 数值与发布验收记录

2026-11-26，基线 `babc8ea221664412de7deca3cdb72833d61f018a`。开发开始时上游 main 与该提交相同。主实现和参考来自仓库 `R/FSBLUP.R`，用户输入类型依据 README.md。DESCRIPTION 声明 GPL (>=3)。`renv.lock` 固定 R 4.4.1、rrBLUP 4.6.3、AGHmatrix 2.1.4、Matrix 1.7-0。已在独立目录构建 R 4.4.1 与 Matrix 1.7-0，生成脚本启动时检查四个实际版本。`R CMD build` 和 `R CMD INSTALL` 均已通过。科学来源为论文公式 3、7、Materials and methods 及补充推导。[基线源码](https://github.com/Xueyh1/FSBLUP/blob/babc8ea221664412de7deca3cdb72833d61f018a/R/FSBLUP.R) · [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12888590/) · [Additional file 3](https://pmc-oa-opendata.s3.amazonaws.com/PMC12888590.1/13059_2026_3931_MOESM3_ESM.pdf) · [论文归档](https://doi.org/10.5281/zenodo.17760713)

## 覆盖与数值定义

**全覆盖。** 三个集合相同，按 A 的 ID 排列。原始融合矩阵为 (1−α−β)A+βG+αO。先检验实质非对称，容差为 1e-10 × max(1,max|Kij|)。容差内两轴取平均。融合原矩阵先试 Cholesky；失败时计算最小特征值，低于 −64 ε n max(1,max|λᵢ|) 即判无效。允许合法半正定与舍入级负特征值，保留 `raw` 并在交给求解器的 K 上统一加 1e-6 I，无投影。`fusion.tsv` 记录分块与全局扰动。R 的 `adj_pos`（206–215 行）先加扰动再尝试 Cholesky，还可走 nearPD。

**组学子集 B。** 对应 `BB000`（256–291 行）。S 为有 O 的个体，U 为其余个体。用 B=G_SS+1e-6 I。令 d 为平均对角线，m 为全矩阵元素均值。斜率 w=(m(B)−d(B))/(m(O)−d(O))，截距 v=d(O)−w d(B)。校准量 wO+v 的截距作用于所有元素。T=α(wO+v)+(1−α)B。用三角求解得到 L=G_US B⁻¹。填 H_SS=T、H_US=LT、H_UU=G_UU+L(T−B)L′，另一块按转置填。K_raw=(1−β)A+βH/d(H)。α=0 时直接取 T=B。固定 α=.2、β=.5 的 `B_base/B_cal/B_T/B_H/B_K` 记录逐步参考。子集核夹具 Ob 在原始 O 子块上加单位阵以构造可识别、稳定的校准用例。测试还对参考子集重算 B_Kref，验证它与全体先融合再截取确有差异。

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

覆盖由完整输入的集合决定。组 1 为空且有组 2 时采用已验证的 B 规则。仅组 4 时采用全覆盖规则。G=O 且为 A 的真子集、O 不属于 G 的个体、A 外个体、空层均不开放融合。调参限制到参考群后若丢失区分原覆盖所需的组，报告无法辨识权重。实质不定的候选直接失格；最终全体融合不定也返回错误。可另行准备已核实的共同覆盖矩阵开展同组分析，这会改变分析群体。

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

## 实际数据与性能

上游 example.zip 的 SHA-256 为 `c43dde9f91297b346928865e1d88fbdc731e623542b03b2e0f25e8b0a0bdafd6`。实际数据行×数值/特征列（均另有 ID）：A 1407×1407，G 198×198，O 74×74，组学 74×5000，SNP 198×5000，系谱 1407×2，表型 198×3（id、age、bf、czs）。矩阵两轴实际都使用 X 前缀字符串。完整原始文件重新构核后，按两轴 ID 对齐归档矩阵，A/G/O 的最大绝对差分别为 5.11e−15（X1393,X1393）、1.32e−14（X1328,X1313）、8.44e−15（X1292,X1292）。[示例数据](https://github.com/Xueyh1/FSBLUP/releases/download/example/example.zip)

2026-11-26 Slurm 单任务、1 CPU，Intel Xeon Silver 4216，CentOS Stream 9，GCC 11.5，OpenBLAS 0.3.29。构核用原始完整文件。固定权重对比用 74 个共同观测 ID 的相同预构 TSV 和同一性状 age，按组学文件顺序每隔 10 行遮蔽一个，66 个参考、8 个候选。R 与 C++ 都限制为 1 线程，R 用此机串行 OpenBLAS，C++ 用 pthread 构建。时间为单次计时；小数据的初始化与文本 I/O 占比较大。

| 运行 | 读取或构核 s | 融合 s | REML/BLUP s | 搜索 s | 墙钟 s | 峰值 RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| C++ 完整原始 A/G/O | .019 / .243 / .174 | — | — | — | 1.617（写入 1.180） | 33772 |
| R 同输入固定 .2/.5 | .031 | .371 | .006 | — | .409 | 268844 |
| C++ 同输入固定 .2/.5 | .021 | .00062 | .00200 | — | .0245 | 11696 |
| C++ 同输入默认 5 折×2 | .021 | .00023 | .00152 | .163 | .186 | 11808 |

R 的 .409 s 从重读与 C++ 相同的 TSV 算起，进程总耗时 3.18 s 含载包、准备子集与写输入，RSS 为进程峰值。C++ 命令的外部耗时分别 .03 s 与 .19 s。固定权重的 u 对 R 最大差 6.05e−12（X1305），λ 在上界。完整嵌套归档矩阵在 .2/.5 时最小特征值约 −4.028445，程序拒绝拟合；74 个体结果属于另一个明确限定的覆盖群体。

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

## 发布复验

构建与打包命令见 README。解包后 `bin/fsblup` 通过相对 RPATH 找到 `lib/` 下的 libopenblasp.so.0、libstdc++.so.6、libgcc_s.so.1、libgfortran.so.5、libquadmath.so.0。系统提供动态加载器、libc 和 libm。包中附有 `linked-libraries.txt`。使用 Ubuntu Base 22.04.5 amd64 的独立只读根文件系统，Bubblewrap 隔离宿主库、R、环境变量与网络，运行 build、固定权重 fit 与带搜索的 fit。根文件系统的 glibc 是 2.35，发布基线主机的 glibc 是 2.34。

```bash
curl -fLO https://cdimage.ubuntu.com/ubuntu-base/releases/22.04/release/ubuntu-base-22.04.5-base-amd64.tar.gz
echo '242cd8898b33ea806ef5f13b1076ed7c76f9f989d18384452f7166692438ff1a  ubuntu-base-22.04.5-base-amd64.tar.gz' | sha256sum -c -
mkdir ubuntu-base
tar -xzf ubuntu-base-22.04.5-base-amd64.tar.gz -C ubuntu-base --exclude='dev/*'
srun -n1 -c1 --mem=2G tools/clean-smoke.sh ubuntu-base dist/fsblup-0.1.0-linux-x86_64-glibc2.34.tar.gz clean-results
```

此测试在隔离环境内逐 ID 核验全部 24 行固定权重 u/yhat 与 R 参考的绝对误差 ≤1e−5，并检验候选数为 4。Slurm 共享宿主内核。当前支持密集单核单性状。任意非嵌套覆盖、多组学任意多核、重复记录、PLINK 原生格式、APY 和矩阵自由解法属于后续范围。
