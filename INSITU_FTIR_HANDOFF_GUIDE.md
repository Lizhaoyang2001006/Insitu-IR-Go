# 原位红外（In-Situ FTIR）CO₂–胺液解吸数据处理规范与完整数学模型说明

> **文档说明**：本文档完整披露 $\text{CO}_2$ 富液（以单乙醇胺 MEA 水溶液体系为例）程序升温与恒温催化解吸原位红外（In-Situ FTIR）光谱的标准化预处理、气相水汽正交投影去噪、共享物理基底多峰去卷积定量及表观再生速率回归的**全部数学模型与算法实现细节**。所有方程、窗口边界、滤波器阶数、约束上下界及输出字段均与仓库 `scripts/` 源代码 **1:1 严格一致**。本仓库仅公开通用算法代码与数学方法，不包含任何实验原始光谱或实验图谱文件。

---

## 一、输入数据规格与命名约定

管线支持任意组数的对照样与催化剂样品（例如 `Blank`、`Catalyst-A`、`Catalyst-B`），自动识别 `.csv`（按 `utf-8-sig → utf-8 → gb18030` 探测编码）及 `.xlsx` 文件，兼容 1 行常规表头与多行仪器导出表头（如 OMNIC 4 行表头）：

* **程序升温段（`Ramp`，如 `40 → 120 °C`）**：文件名含 `Temp` / `Temperature` / `Ramp` / `升温`（例如 `Blank-Temperature.csv`、`CatalystA-Temperature.csv`），首列为波数 $\nu$ ($\text{cm}^{-1}$)，后续各列头包含温度数值；
* **恒温保持段（`Hold`，如 `120 °C` 恒温）**：文件名含 `120` / `Time` / `Hold` / `Isothermal` / `恒温`（例如 `Blank-120-Time.csv`、`CatalystA-120-Time.csv`），首列为波数 $\nu$ ($\text{cm}^{-1}$)，后续各列头包含采集时间戳（$\text{min}$）。

脚本 `infer_canonical_key()` 通过正则自动剥离模式后缀并提取样品前缀 `<Sample>_<Hold|Ramp>`，自动将 `Blank` 排在首位作为对照基准。

---

## 二、CO₂–MEA 水相体系红外特征峰归属与光谱窗口物理依据

| 波数区间 ($\text{cm}^{-1}$) | 对应物种 | 振动模式 | 解吸再生演化方向 | 物理机制与定量防呆原则 |
|:---:|:---|:---|:---:|:---|
| **1634–1642** | $\text{MEAH}^+$ + 液态 $\text{H}_2\text{O}(l)$ | $\delta_{\text{as}}(\text{NH}_3^+)$ / $\delta(\text{H–O–H})$ | **↓ 减弱** | 水相体系中二者强重叠，水分挥发同样引起该带下降，**不可单独作为催化脱质子化判据** |
| **1596–1600** | **`*MEA`**（再生游离/吸附态胺） | $-\text{CH}_2\text{–NH}_2$ 剪式弯曲 $\delta(\text{NH}_2)$ | **↑ 增强** | 受 `1566` 邻峰叠加与 `1638` 水峰右翼下陷影响，实测差谱表观正峰顶常移至 `1588–1592 cm⁻¹` |
| **1566–1568** | **$\text{MEACOO}^-$**（氨基甲酸盐） | $\nu_{\text{as}}(\text{COO}^-)$ | **相对减弱 / 净负峰** | 单纯失水浓缩时呈微幅上升；高活性催化分解速率超越浓缩效应时呈现**净负峰** |
| **1487** | **$\text{MEACOO}^-$**（叠加 $\text{MEAH}^+$） | $\nu_{\text{s}}(\text{COO}^-)$ / $\delta_{\text{s}}(\text{NH}_3^+)$ | **↓ 减弱** | 对称羧酸根伸缩振动记作 $\nu_{\text{s}}(\text{COO}^-)$ |
| **1386 / 1324** | $\text{HCO}_3^-$ / $\text{MEACOO}^-$ | $\nu_{\text{s}}(\text{COO}^-)$ / $\nu(\text{N–COO}^-)$ | 低温清晰，高温易受本底包络掩盖 | 仅作低温定性参考，不纳入高温动力学回归 |
| **1069 / 1016** | $\text{MEA} / \text{MEAH}^+$ | $\nu(\text{C–O})$ / $\nu(\text{C–N})$ | 低温特征锐峰 | $\nu(\text{C–O})$ 波数高于 $\nu(\text{C–N})$ |
| *1745–1760* | 2-噁唑烷酮（OZD，热降解副产物） | 环状羰基 $\nu(\text{C=O})$ | 监控有无生成 | 用于核验高温恒温窗口内是否发生胺热降解副反应 |
| *1000–1450* | 高温池体/液膜非催化宽包络 | 背景热漂移包络（含 `~1432` 带） | 随升温同步抬升 | 不做强行扣空白；诊断与分峰窗口收紧至 **`1475.0 – 1750.0 cm⁻¹`** 以切净 `1432` 高波数侧肩溢入 |
| *<1000 & >1800* | 窗片截止区 / 气相 $\text{CO}_2(g)$（`2280–2420`） | 检测器截止饱和 / 光路吹扫失配 | 非定量区 | Step 1A 统一截断剔除 |

---

## 三、完整处理流程与数学模型披露（Step 0 → Step 4）

设单个数据集中第 $j$ 条原始光谱在离散波数网格 $\{\nu_i\}$ 上的吸光度为 $S_j(\nu_i)$。

---

### 3.1 Step 0：四重自动化数据质量审计门禁（`scripts/step1A_baseline_truncate.py`）

1. **门禁 0.1（波数轴有限性与严格单调性）**：
   $$\forall i,\;\; |\nu_i| < \infty \quad \land \quad \Delta\nu_i = \nu_{i+1} - \nu_i > 0$$
   若存在任何 `NaN`/`Inf` 或非严格单调递增点，立即抛出 `[AUDIT FAIL]` 终止流程。
2. **门禁 0.2（误贴波数列自动识别与剔除）**：
   对任意数据列 $j \ge 1$，若满足：
   $$\min_i S_j(\nu_i) > 900.0 \quad \land \quad \max_i S_j(\nu_i) > 3500.0$$
   自动判定为拼表时误粘贴的波数列，予以剔除并记录于 `Step1A_Audit_Baseline_Offsets.csv`。
3. **门禁 0.3（截断域吸光度有限性与全文件重复谱排查）**：
   核查截断区间 $\Omega_{\text{1A}}$ 内所有数据点满足 $|S_j(\nu_i)| < \infty$；并将当前谱图与同文件内**所有前序谱图** $S_m\ (m < j)$ 做数值比对（`np.allclose(..., rtol=1e-5, atol=1e-8)`），若发现重复列则在 `Step1A_Manifest.csv` 的 `DuplicateDataWarnings` 中告警，同时对同名列头自动添加 `_dup{j}` 后缀消歧。
4. **门禁 0.4（升温末态与恒温首态连续性审计）**：
   若同一样品同时存在升温段 `<Sample>_Ramp` 与恒温段 `<Sample>_Hold`，在催化诊断窗 $\Omega_{\text{diag}} = [1475.0,\, 1750.0]\text{ cm}^{-1}$ 自动计算升温终点谱图 $S_{\text{Ramp,end,1A}}$ 与恒温起点谱图 $S_{\text{Hold},0,\text{1A}}$（$t = 0\text{ min}$）之间的均方根偏差：
   $$\operatorname{RMS}_{\text{Ramp(end)}\to\text{Hold}(t_0)} = \sqrt{\frac{1}{|\Omega_{\text{diag}}|} \sum_{\nu_i \in [1475.0,\, 1750.0]} \left(S_{\text{Hold},0,\text{1A}}(\nu_i) - S_{\text{Ramp,end,1A}}(\nu_i)\right)^2}$$
   结果写入 `Step1A_Manifest.csv` 的 `Continuity_RMS_1475_1750` 列，用于核验恒温首帧是否真实对齐目标温度到达时刻。

---

### 3.2 Step 1A：统一波数截断与无吸收锚定窗中位数水平基线归零（`scripts/step1A_baseline_truncate.py`）

1. **统一波数截断域 $\Omega_{\text{1A}}$**：
   $$\Omega_{\text{1A}} = \{\nu_i \mid \nu_{\min} \le \nu_i \le \nu_{\max}\}, \qquad \nu_{\min} = 999.9\text{ cm}^{-1},\;\; \nu_{\max} = 1800.3\text{ cm}^{-1}$$
   （可通过 `--wn-min` / `--wn-max` 配置，波数输出四舍五入至 5 位小数。）
2. **中位数水平基线平移归零**：
   选取无液相特征吸收的平直锚定窗 $\Omega_{\text{bl}} = [1750.0,\, 1800.0]\text{ cm}^{-1}$（可通过 `--bl-min` / `--bl-max` 配置），取每条光谱在该窗口内的吸光度**中位数** $b_j$ 做刚性水平平移：
   $$b_j = \operatorname{median}_{\nu_i \in [1750.0,\, 1800.0]} S_j(\nu_i), \qquad S_{j,\text{1A}}(\nu_i) = S_j(\nu_i) - b_j \quad (\forall \nu_i \in \Omega_{\text{1A}})$$
   * **数学稳健性说明**：中位数算子具备 $50\%$ 崩溃点（Breakdown Point），可有效免疫 `1750–1800 cm⁻¹` 内残余尖锐气相水汽转动线引起的单侧均值偏移；严禁采用跨 `1455–1750 cm⁻¹` 的两点倾斜拉线基线，以免 `1432 cm⁻¹` 高温包络抬升导致 `1487 / 1568 cm⁻¹` 吸收峰被人为压低失真。校正后吸光度保留 7 位小数，中位数与算术均值同步存档于 `Step1A_Audit_Baseline_Offsets.csv`。

---

### 3.3 Step 2：带皮尔逊相关门控的气相水汽 $\text{H}_2\text{O}(g)$ 正交投影扣减（`scripts/step2_step3_wvclean_and_ratios.py`）

针对气相水汽弯曲振动 $\nu_2$ 转动精细结构（窄线宽 `1–3 cm⁻¹`，分布于 `1350–1800 cm⁻¹`）与凝聚相胺/氨基甲酸盐宽带（$\text{FWHM} > 30\text{ cm}^{-1}$）之间的**频域尺度分离**特性，建立四步带门控正交投影算法：

1. **高水汽模板源数据集自动遴选**：
   若未显式指定 `--template-key`，对每个数据集 $D$（含 $N_D$ 条光谱）取末尾 $M_D = \min(8, N_D)$ 条光谱的均值谱 $\bar{S}_D(\nu)$，利用 **55 点 3 阶 Savitzky–Golay 平滑算子** $\operatorname{SG}_{55,3}$ 提取高通残差，并在无液相吸收的纯气相水汽锚定窗 $\Omega_{\text{anc}} = [1700.0,\, 1800.0]\text{ cm}^{-1}$ 上选取高频总体标准差（`ddof=0`）最大的数据集 $D^*$：
   $$D^* = \arg\max_{D} \operatorname{std}_{\nu_i \in [1700.0,\, 1800.0]}\!\left(\bar{S}_D(\nu_i) - \operatorname{SG}_{55,3}(\bar{S}_D)(\nu_i)\right)$$

2. **带线性渐变窗（Linear Taper）的锚定区单位 $L_2$ 范数模板 $W(\nu)$ 构造**：
   记高通残差 $H(\nu) = \bar{S}_{D^*}(\nu) - \operatorname{SG}_{55,3}(\bar{S}_{D^*})(\nu)$。对 `<1350.0 cm⁻¹` 严格置零，在 `[1350.0, 1365.0] cm⁻¹` 施加宽度 $15.0\text{ cm}^{-1}$ 的线性渐变窗以消除截断不连续阶跃：
   $$W_{\text{raw}}(\nu) = \begin{cases} 0, & \nu < 1350.0\text{ cm}^{-1} \\[4pt] \dfrac{\nu - 1350.0}{15.0}\, H(\nu), & 1350.0 \le \nu \le 1365.0\text{ cm}^{-1} \\[6pt] H(\nu), & 1365.0 < \nu \le 1800.3\text{ cm}^{-1} \end{cases}$$
   随后在纯气相锚定窗 $\Omega_{\text{anc}} = [1700.0,\, 1800.0]\text{ cm}^{-1}$ 上归一化为单位 $L_2$ 范数：
   $$W(\nu) = \frac{W_{\text{raw}}(\nu)}{\|W_{\text{raw}}\|_{2,\Omega_{\text{anc}}}}, \qquad \|W_{\text{raw}}\|_{2,\Omega_{\text{anc}}} = \sqrt{\sum_{\nu_i \in \Omega_{\text{anc}}} W_{\text{raw}}(\nu_i)^2}, \qquad \sum_{\nu_i \in \Omega_{\text{anc}}} W(\nu_i)^2 = 1$$

3. **锚定窗高频投影与皮尔逊相关门控（Correlation Gate）**：
   对待处理光谱 $S_{j,\text{1A}}(\nu)$，提取同阶高通分量 $S_{j,\text{hp}}(\nu) = S_{j,\text{1A}}(\nu) - \operatorname{SG}_{55,3}(S_{j,\text{1A}})(\nu)$，在 $\Omega_{\text{anc}} = [1700.0,\, 1800.0]\text{ cm}^{-1}$ 上计算：
   * **皮尔逊相关系数 $r_j$**：
     $$r_j = \frac{\sum_{\nu_i \in \Omega_{\text{anc}}} \left(S_{j,\text{hp}}(\nu_i) - \overline{S_{j,\text{hp}}}\right)\left(W(\nu_i) - \overline{W}\right)}{\sqrt{\sum_{\nu_i \in \Omega_{\text{anc}}} \left(S_{j,\text{hp}}(\nu_i) - \overline{S_{j,\text{hp}}}\right)^2}\,\sqrt{\sum_{\nu_i \in \Omega_{\text{anc}}} \left(W(\nu_i) - \overline{W}\right)^2}}$$
   * **过原点最小二乘正交投影系数 $\beta_{j,\text{raw}}$**（因 $\sum_{\Omega_{\text{anc}}} W(\nu_i)^2 = 1$，内积精确等于最小二乘投影斜率）：
     $$\beta_{j,\text{raw}} = \langle S_{j,\text{hp}}, W\rangle_{\Omega_{\text{anc}}} = \sum_{\nu_i \in \Omega_{\text{anc}}} S_{j,\text{hp}}(\nu_i)\, W(\nu_i)$$
   * **硬阈值相关门控**（默认 $r_{\text{gate}} = 0.50$，可通过 `--corr-gate` 调整）：
     $$\beta_{j,\text{applied}} = \begin{cases} \beta_{j,\text{raw}}, & |r_j| \ge r_{\text{gate}} \\ 0, & |r_j| < r_{\text{gate}} \end{cases}$$
   * **有符号投影系数的物理含义**：允许 $\beta_{j,\text{applied}}$ 取正值或负值，分别对应样品采集时光路环境湿度相对背景光谱的同向或反向湿度差；当 $|r_j| < r_{\text{gate}}$ 时强制置零，确保对不含气相水汽精细结构的低噪光谱零扰动。

4. **全谱扣减、残余基线重归零与高频纹波评估**：
   扣减水汽模板后，在 $\Omega_{\text{bl}} = [1750.0,\, 1800.0]\text{ cm}^{-1}$ 执行二次中位数微调重归零（`Post_BL_Rezero`）：
   $$\tilde{S}_j(\nu) = S_{j,\text{1A}}(\nu) - \beta_{j,\text{applied}} \cdot W(\nu)$$
   $$\delta b_j = \operatorname{median}_{\nu_i \in [1750.0,\, 1800.0]} \tilde{S}_j(\nu_i), \qquad S_{j,\text{clean}}(\nu_i) = \tilde{S}_j(\nu_i) - \delta b_j$$
   在催化核心评估窗 $\Omega_{\text{eval}} = [1480.0,\, 1720.0]\text{ cm}^{-1}$ 上，以 **41 点 3 阶 Savitzky–Golay 平滑曲线** $\operatorname{SG}_{41,3}$ 为基准定量评估去水汽前后的高频均方根纹波降幅：
   $$\operatorname{HF\_RMS}(S) = \operatorname{std}_{\nu_i \in [1480.0,\, 1720.0]}\!\left(S(\nu_i) - \operatorname{SG}_{41,3}(S)(\nu_i)\right)$$
   $$\text{HF\_RMS\_Reduction\_Pct} = \left(1 - \frac{\operatorname{HF\_RMS}(S_{j,\text{clean}})}{\operatorname{HF\_RMS}(S_{j,\text{1A}})}\right) \times 100\%$$

---

### 3.4 Step 3：抗蒸发内标比值与三指标交叉验证定量（`scripts/step2_step3_wvclean_and_ratios.py`）

为消除高温吹扫过程中水分挥发导致的液膜厚度变化与溶质浓缩效应，采用同谱氨基甲酸盐特征带作为内标分母；同时针对 `1600 cm⁻¹` 紧邻 `1638 cm⁻¹` 液态水/$\text{MEAH}^+$ 宽峰右翼易受失水塌陷牵引的光谱重叠问题，同步输出三套互补指标：

#### 指标 1（主推，`decon`）：共享 6 峰物理高斯基底箱式约束最小二乘去卷积 $I_{\text{decon}}(1596)/I_{\text{decon}}(1566)$
在干净催化诊断窗 $\Omega_{\text{fit}} = [1475.0,\, 1750.0]\text{ cm}^{-1}$ 上，对所有样品、所有温度与时间帧统一施加**固定中心 $\mu_k$ 与固定宽度 $\sigma_k$ 的 6 高斯物理基函数**（半高全宽 $\text{FWHM}_k = 2\sqrt{2\ln 2}\,\sigma_k \approx 2.3548\,\sigma_k$）及有界平直微调项 $c_0$，从数学上根除自由分峰的参数共线性与峰位漂移：

| 基函数序号 $k$ | 固定中心 $\mu_k$ ($\text{cm}^{-1}$) | 固定标准差 $\sigma_k$ ($\text{cm}^{-1}$) | $\text{FWHM}_k$ ($\text{cm}^{-1}$) | 对应物理振动归属 |
|:---:|:---:|:---:|:---:|:---|
| $1$ | `1638.0` | `24.0` | `56.5` | $\text{H}_2\text{O}(l)\ \delta(\text{H–O–H}) + \text{MEAH}^+\ \delta_{\text{as}}(\text{NH}_3^+)$ |
| $2$ | `1596.0` | `15.0` | `35.3` | 再生游离/吸附态 **`*MEA`** $\delta(\text{NH}_2)$（分子项） |
| $3$ | `1566.0` | `15.0` | `35.3` | **$\text{MEACOO}^-$** $\nu_{\text{as}}(\text{COO}^-)$（分母内标项） |
| $4$ | `1528.0` | `18.0` | `42.4` | 氢键耦合/过渡重叠肩带 |
| $5$ | `1487.0` | `15.0` | `35.3` | $\text{MEACOO}^-\ \nu_{\text{s}}(\text{COO}^-) / \text{MEAH}^+\ \delta_{\text{s}}(\text{NH}_3^+)$ |
| $6$ | `1438.0` | `18.0` | `42.4` | `1432 cm⁻¹` 高温背景包络高波数侧翼补偿 |

* **光谱线性组合模型与箱式约束最小二乘优化问题**（通过信赖域反射算法 `scipy.optimize.least_squares` 求解）：
  $$\hat{S}_j(\nu; \mathbf{c}, c_0) = c_0 + \sum_{k=1}^{6} c_k \exp\!\left(-\frac{(\nu - \mu_k)^2}{2\sigma_k^2}\right)$$
  $$(\hat{\mathbf{c}}, \hat{c}_0) = \arg\min_{\mathbf{c},\, c_0} \sum_{\nu_i \in \Omega_{\text{fit}}} \left(\hat{S}_j(\nu_i; \mathbf{c}, c_0) - S_{j,\text{clean}}(\nu_i)\right)^2 \quad \text{s.t.} \quad c_k \ge 0\;\;(k=1,\dots,6),\quad -0.015 \le c_0 \le +0.015$$
* **解析高斯峰面积与去卷积再生比值**：
  $$I_k = \int_{-\infty}^{+\infty} \hat{c}_k \exp\!\left(-\frac{(\nu - \mu_k)^2}{2\sigma_k^2}\right) d\nu = \sqrt{2\pi}\,\hat{c}_k\sigma_k$$
  $$R_{\text{decon},j} = \frac{I_2}{I_3} = \frac{I_{\text{decon}}(1596)}{I_{\text{decon}}(1566)} = \frac{\hat{c}_2\sigma_2}{\hat{c}_3\sigma_3} = \frac{\hat{c}_2}{\hat{c}_3} \quad (\text{因 } \sigma_2 = \sigma_3 = 15.0\text{ cm}^{-1})$$
* **分峰拟合优度 $R^2_{\text{fit}}$ 与拟合残差均方根误差 $\operatorname{RMSE}_{\text{fit}}$**：
  定义逐点拟合残差 $\varepsilon_j(\nu_i) = S_{j,\text{clean}}(\nu_i) - \hat{S}_j(\nu_i)$：
  $$R^2_{\text{fit},j} = 1 - \frac{\sum_{\nu_i \in \Omega_{\text{fit}}} \varepsilon_j(\nu_i)^2}{\sum_{\nu_i \in \Omega_{\text{fit}}} \left(S_{j,\text{clean}}(\nu_i) - \overline{S_{j,\text{clean}}}\right)^2}, \qquad \operatorname{RMSE}_{\text{fit},j} = \sqrt{\frac{1}{|\Omega_{\text{fit}}|} \sum_{\nu_i \in \Omega_{\text{fit}}} \varepsilon_j(\nu_i)^2}$$
  脚本自动生成对齐恒温终点时刻的分峰重构与底部残差验证图 `Fig_SI_Step3_Deconvolution_Verification.png/.pdf`。

#### 指标 2（零参数实测峰顶敏感性比值，`r1590`）与指标 3（传统文献峰位比值，`r1600`）
定义以波数 $\nu_0$ 为中心、半宽 $h = 2.0\text{ cm}^{-1}$（即 $[\nu_0 - 2.0,\, \nu_0 + 2.0]\text{ cm}^{-1}$）的窄带算术平均吸光度：
$$A_j(\nu_0) = \frac{1}{|I(\nu_0)|} \sum_{\nu_i \in [\nu_0 - 2.0,\, \nu_0 + 2.0]} S_{j,\text{clean}}(\nu_i)$$
同步计算零分峰参数比值及辅助验证比值：
$$R_{1590,j} = \frac{A_j(1590)}{A_j(1568)}, \qquad R_{1600,j} = \frac{A_j(1600)}{A_j(1568)}, \qquad R_{1487,j} = \frac{A_j(1487)}{A_j(1568)}, \qquad R_{1634,j} = \frac{A_j(1634)}{A_j(1568)}$$
对任一主指标 $R \in \{R_{\text{decon}}, R_{1590}, R_{1600}\}$ 定义相对于初始帧（恒温 $t = 0\text{ min}$ 或升温初始温度）的净增量：
$$\Delta R_j = R_j - R_0$$

---

### 3.5 Step 4：二维差谱矩阵、统一对称色标与 OLS 表观再生速率回归（`scripts/step4_isothermal_fig4_template.py`）

1. **恒温相对时间轴对齐与共同窗口截取**：
   以各样品恒温段首条光谱的时间戳 $t_{\text{raw},0}$ 为零点建立相对保持时间轴，并截取共同时间窗口 $[0,\, t_{\max}]$（默认 `--max-hold-min 10.6`）：
   $$t_j = t_{\text{raw},j} - t_{\text{raw},0}, \qquad 0 \le t_j \le t_{\max}$$
2. **诊断窗二维差谱矩阵**（$\Omega_{\text{diag}} = [1475.0,\, 1750.0]\text{ cm}^{-1}$）：
   * 恒温段差谱矩阵：$\Delta S_{\text{iso}}(\nu_i, t_j) = S_{\text{clean}}(\nu_i, t_j) - S_{\text{clean}}(\nu_i, t_0)$
   * 升温段差谱矩阵：$\Delta S_{\text{ramp}}(\nu_i, T_j) = S_{\text{clean}}(\nu_i, T_j) - S_{\text{clean}}(\nu_i, T_0)$
3. **跨样品共享对称发散色标（`RdBu_r`）自动定标**：
   取全部参比样品 $|\Delta S|$ 的 $99.5\%$ 分位数最大值加 $0.0005\text{ Abs}$ 裕量并保留 3 位小数，强制所有子图共享同一对称色标区间 $[-v_{\max},\, +v_{\max}]$：
   $$v_{\max} = \operatorname{round}\!\left(\max_{s} P_{99.5}\!\left(|\Delta S_s|\right) + 0.0005,\; 3\right)$$
4. **浓度无关表观光谱 MEA 再生速率常数 $k$ 的普通最小二乘（OLS）线性回归**（`scipy.stats.linregress`）：
   在共同恒温窗口 $[0,\, t_{\max}]$ 内对 $(t_j, R_j)_{j=0}^{n-1}$ 拟合线性动力学方程 $R(t_j) = k \cdot t_j + R_{\text{int}} + e_j$（注：$\Delta R_j = R_j - R_0$ 仅平移截距，斜率 $k$、标准误 $\operatorname{SE}(k)$、决定系数 $R^2$ 与 $p\text{-value}$ 在 D1 堆叠图和 D2 净增量叠加图中严格恒等）：
   $$k = \frac{\sum_{j=0}^{n-1}(t_j - \bar{t})(R_j - \bar{R})}{\sum_{j=0}^{n-1}(t_j - \bar{t})^2}, \qquad \operatorname{SE}(k) = \sqrt{\frac{\frac{1}{n-2}\sum_{j=0}^{n-1}\hat{e}_j^2}{\sum_{j=0}^{n-1}(t_j - \bar{t})^2}}, \qquad R^2 = \frac{\left[\sum_{j=0}^{n-1}(t_j - \bar{t})(R_j - \bar{R})\right]^2}{\sum_{j=0}^{n-1}(t_j - \bar{t})^2 \sum_{j=0}^{n-1}(R_j - \bar{R})^2}$$
   双侧 $t$ 检验统计量 $t_{\text{stat}} = k / \operatorname{SE}(k)$（自由度 $\text{df} = n - 2$）用于给出显著性水平 $p\text{-value}$。

---

## 四、代码与数学方程逐项映射表（Code-to-Math Traceability）

| 算法模块 | 数学量 / 约束条件 | 对应源码脚本与符号 | 默认参数值 |
|:---|:---|:---|:---|
| 波数截断域 | $\Omega_{\text{1A}} = [\nu_{\min}, \nu_{\max}]$ | `step1A_baseline_truncate.py` (`WN_MIN, WN_MAX`) | `[999.9, 1800.3]` cm⁻¹ |
| 水平基线锚定窗 | $b_j = \operatorname{median}_{\Omega_{\text{bl}}} S_j(\nu)$ | `step1A_baseline_truncate.py` (`BL_MIN, BL_MAX`) | `[1750.0, 1800.0]` cm⁻¹ |
| 误贴波数列门禁 | $\min > 900 \land \max > 3500$ | `step1A_baseline_truncate.py` (`Gate 0.2`) | 自动剔除并记录日志 |
| 升温—恒温衔接审计 | $\operatorname{RMS}_{\text{Ramp(end)}\to\text{Hold}(t_0)}$ | `step1A_baseline_truncate.py` (`Gate 0.4`) | `[1475.0, 1750.0]` cm⁻¹ |
| 水汽高频分离滤波器 | $\operatorname{SG}_{55,3}$ | `step2_step3_wvclean_and_ratios.py` (`savgol_filter(..., 55, 3)`) | 55 点窗口，3 阶多项式 |
| 水汽模板支集与渐变窗 | $[1350, 1800.3]$，斜坡 $[1350, 1365]$ | `step2_step3_wvclean_and_ratios.py` (`W_raw`, `m_taper`) | $15.0\text{ cm}^{-1}$ 线性斜坡 |
| 水汽锚定归一化窗 | $\Omega_{\text{anc}}$, $\|W\|_{2,\Omega_{\text{anc}}} = 1$ | `step2_step3_wvclean_and_ratios.py` (`m_anchor`) | `[1700.0, 1800.0]` cm⁻¹ |
| 皮尔逊相关门控 | $|r_j| \ge r_{\text{gate}}$ | `step2_step3_wvclean_and_ratios.py` (`--corr-gate`) | `0.50` |
| 扣水汽后二次归零 | $\delta b_j = \operatorname{median}_{[1750, 1800]} \tilde{S}_j(\nu)$ | `step2_step3_wvclean_and_ratios.py` (`Post_BL_Rezero`) | `[1750.0, 1800.0]` cm⁻¹ |
| 高频纹波评价窗 | $\operatorname{HF\_RMS}$ ($\operatorname{SG}_{41,3}$) | `step2_step3_wvclean_and_ratios.py` (`m_eval`) | `[1480.0, 1720.0]` cm⁻¹，41 点 3 阶 |
| 共享 6 峰高斯去卷积 | $\mu_k, \sigma_k,\; c_k \ge 0,\; |c_0| \le 0.015$ | `step2_step3_wvclean_and_ratios.py` (`SHARED_BASIS_PEAKS`, `deconvolve_spectrum`) | `[1475.0, 1750.0]` cm⁻¹ |
| 窄带平均比值 | $A(\nu_0 \pm 2.0\text{ cm}^{-1})$ | `step2_step3_wvclean_and_ratios.py` (`band_mean(..., hw=2.0)`) | `1634/1600/1590/1568/1487` |
| 恒温共同窗口与回归 | $0 \le t_j \le t_{\max}$，OLS `linregress` | `step4_isothermal_fig4_template.py` (`--max-hold-min`) | `10.6` min |
| 对称热图色标定标 | $\operatorname{round}(\max_s P_{99.5}(\|\Delta S_s\|)+0.0005, 3)$ | `step4_isothermal_fig4_template.py` (`vmax_hold`, `vmax_rmp`) | `RdBu_r` 对称发散色标 |

---

## 五、命令行调用说明

```bash
# 1. 将待处理的原始 CSV / XLSX 光谱放入本地 ./raw_data/（该目录已被 .gitignore 忽略，不会上传）
# 2. 一键执行 Step 0 -> Step 1A -> Step 2 -> Step 3 -> Step 4 全流程
python3 run_pipeline.py

# 3. 或按模块分步调用并自定义参数
python3 scripts/step1A_baseline_truncate.py --input-dir ./raw_data --out-dir ./output/step1A
python3 scripts/step2_step3_wvclean_and_ratios.py --in-dir ./output/step1A --out-dir ./output/step2_3 --corr-gate 0.50 --max-hold-min 10.6
python3 scripts/step4_isothermal_fig4_template.py --in-dir ./output/step2_3 --out-dir ./output/step4 --metric all --max-hold-min 10.6
```
