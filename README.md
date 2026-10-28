# fastJFP：主代码与单配置示例

本目录可独立移动使用，只保留快速 JFP 主代码及每类问题的一个计算配置。没有绘图、计时与存储 benchmark、参数扫描、原始数据或自动运行全部算例的入口。

## 目录

```text
Project.toml / Manifest.toml   最小直接依赖与锁定环境
src/FastJFP.jl                统一加载入口
src/                         当前数值算法模块
examples/common.jl           加载入口、谱输入和系数求值
examples/integral_action.jl   单次分数积分作用
examples/abel.jl              Abel 积分方程
examples/multi_order.jl       多阶积分方程
examples/bbo.jl               BBO 方程及物理解重构
examples/sdc.jl               积分形式的延迟校正
examples/eigenvalue.jl        积分逆算子的特征值计算
examples/airy.jl              分数阶 Airy 边值问题
```

`src/` 保留原项目的 11 个数值模块，并增加统一入口 `FastJFP.jl`。其中 `BandRational.jl` 为带状有理预条件入口；`AbelRational.jl` 和 `AiryRational.jl` 都是其实际依赖。其余模块的相对加载关系不变。

`MathCompExamples.jl` 仅保留当前围道计算分支，移除完整矩阵比较及高精度诊断分支，因此不再依赖原目录中缺失的 `reference_mathcomp/Frac.jl`。其余 10 个模块原样复制；保留分支中的积分作用、预条件构造、迭代和重构公式不变。`ReferenceJFP.jl` 仍为现有主代码的加载依赖，保留在 `src/`，但没有附带完整矩阵对比算例。

## 环境与运行

使用 Julia **1.11.5**。直接依赖只包括 BandedMatrices、FFTW、Krylov、KrylovKit、SpecialFunctions 和所需标准库；没有 BenchmarkTools、绘图库或 CSV 依赖。`Manifest.toml` 从原锁定环境提取必要依赖，包版本保持不变。

在本目录下安装依赖：

```powershell
cd C:\mytex\fastJFP\togit
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
```

每次自行选择一个入口运行，不需要提供参数或数据文件：

```powershell
julia --threads=1 --project=. --startup-file=no examples/integral_action.jl
julia --threads=1 --project=. --startup-file=no examples/abel.jl
julia --threads=1 --project=. --startup-file=no examples/multi_order.jl
julia --threads=1 --project=. --startup-file=no examples/bbo.jl
julia --threads=1 --project=. --startup-file=no examples/sdc.jl
julia --threads=1 --project=. --startup-file=no examples/eigenvalue.jl
julia --threads=1 --project=. --startup-file=no examples/airy.jl
```

以上是七条独立命令，不是要求顺次执行的测试流程。每个脚本只有一个固定配置，只计算快速法，打印对应残差、误差或特征值，不生成图片或文件。也可以在 Julia REPL 中 `include("examples/abel.jl")`，从变量 `result` 读取系数、右端和求解结果。不要在同一 REPL 会话反复加载不同入口；每个入口使用独立进程即可。

## 单个算例的方程与参数

积分算子以下端点 $-1$ 为准：

\[
 (\mathcal I_x^\mu f)(x)
 =\frac1{\Gamma(\mu)}\int_{-1}^{x}(x-s)^{\mu-1}f(s)\,ds.
\]

基底为 $t^\alpha T_j(2t^\beta-1)$，其中 $t=(1+x)/2$。所有围道作用取 $M=160$，生产计算使用 Float64/ComplexF64。线性方程的带状预条件取上带宽 32，BLAS 使用单线程。

| 入口 | 唯一配置 | 计算内容 |
|---|---|---|
| `integral_action.jl` | $N=128$，$\mu=\beta=1/2$，$\alpha=0$ | 计算 $\mathcal I_x^{1/2}(1+x)$，与 $4(1+x)^{3/2}/(3\sqrt\pi)$ 比较 |
| `abel.jl` | $N=256$，$\mu=1/2$，$\alpha=0$，$\lambda=4$ | 求解谱右端的 Abel 方程，容差 $10^{-14}$ |
| `multi_order.jl` | $N=30$，$(\alpha,\beta)=(0,1/6)$ | 求解多阶方程，容差 $2\times10^{-14}$ |
| `bbo.jl` | $N=50$，$(\alpha,\beta)=(0,1/2)$ | 求解辅助积分方程并重构 $v$，容差 $2\times10^{-14}$ |
| `sdc.jl` | 10 个映射 Chebyshev 节点 | 用围道积分作用做延迟校正，离散残差容差 $10^{-15}$ |
| `eigenvalue.jl` | $N=500$，$(\alpha,\beta)=(1/2,3/2)$ | Arnoldi 子空间维数 40，容差 $10^{-13}$，计算 7 个最大模逆特征值 |
| `airy.jl` | $\epsilon=10^{-4}$，$N=680$，$(\alpha,\beta)=(1/2,1/2)$ | 辅助未知量形式，容差 $10^{-13}$，至多 3 次残差校正 |

### Abel

\[
 u(x)+2^{-\mu}\lambda(\mathcal I_x^\mu u)(x)=f_N(x).
\]

`common.jl` 的 `spectral_input` 沿用当前项目的右端系数生成规则，种子为 20260918，前约 $4N/5$ 个系数按指数包络下降，其余保留机器精度量级尾部。输出的 `solution.residual` 为重新计算的系数相对残差。

### 多阶方程

\[
 u+\sqrt{1+x}\,\mathcal I_x^{1/3}u
 +\mathcal I_x^{1/2}[(1+\cdot)^{1/3}u]=f,
\]

\[
 f(x)=(1+x)^{3/2}
 +\left[\frac{\Gamma(5/2)}{\Gamma(17/6)}
 +\frac{\Gamma(17/6)}{\Gamma(10/3)}\right](1+x)^{7/3}.
\]

精确解为 $u=(1+x)^{3/2}$，输出 2001 个多项式坐标点上的相对最大误差。

### BBO

\[
 u+\mathcal I_x^{1/2}u+\mathcal I_x^1u=-1,
 \qquad v=1+\mathcal I_x^1u.
\]

对应 $v'+{}^{\mathrm C}\mathcal D_x^{1/2}v+v=0$、$v(-1)=1$。辅助未知量为 $u=v'$；脚本重构 $v$，与原代码中基于 `erfcx` 的解析公式比较。

### SDC

\[
 u-\mathcal I_x^{1/2}u
 =\frac{\sqrt\pi}{2}(1+x)-\frac23(1+x)^{3/2}.
\]

精确解为 $u=\sqrt\pi(1+x)/2$。保留原有分段常数乘积积分初始化和校正，输出离散残差及节点上的绝对最大误差。节点值不是 JFP 系数。

### 特征值

\[
 u(x)+\lambda_{\rm eig}(\mathcal I_x^{3/2}u)(x)
 =\lambda_{\rm eig}\sqrt{\frac{1+x}{2}}(\mathcal I_x^{3/2}u)(1).
\]

对应 $-\mathcal D_x^{3/2}u=\lambda_{\rm eig}u$、$u(-1)=u(1)=0$。积分逆算子的特征值为 $1/\lambda_{\rm eig}$；脚本对其取倒数，返回原问题的特征值。没有附带高精度特征函数诊断。

### Airy

令 $v=\mathcal D_x^{3/2}u$，所求积分方程为

\[
 \epsilon i^{3/2}v(x)-x(\mathcal I_x^{3/2}v)(x)
 -ax\sqrt{\frac{1+x}{2}}=0,
 \qquad a+(\mathcal I_x^{3/2}v)(1)=1.
\]

重构 $u=\mathcal I_x^{3/2}v+a\sqrt{(1+x)/2}$，对应 $u(-1)=0$、$u(1)=1$。脚本求值时包含 $\alpha=1/2$ 的端点权重，返回物理解而不是去权后的正则因子。输出系数残差和边界误差，不把它们称为连续解误差。

## 整理与验证边界

本次只整理代码和说明，不安装依赖、不运行算例、不求解方程，也不重测已有实验。检查范围为 Julia 语法、相对依赖闭合、锁定依赖版本及原样复制模块的一致性；不将静态检查称为数值复验。原项目的 `code/`、数据、论文及历史交付包均不改写。
