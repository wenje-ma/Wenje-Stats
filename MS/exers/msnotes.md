# msnotes

## 绪论

### 简单随机样本

简单随机样本满足:

代表性: 总体中的每个个体被抽到的概率是等可能的;

独立性: 样本中每个个体 $X_i$ 取什么值, 不影响其他个体的取值.

### 联合分布

联合分布列 $p(x_1,x_2,\cdots,x_n)$ 针对离散情形, 其定义为

$$
p(x_1,x_2,\cdots,x_n)=P\{X_1=x_1,X_2=x_2,\cdots,X_n=x_n\},
$$

联合密度函数 $f(x_1,x_2,\cdots,x_n)$ 针对连续情形, 其定义为

$$
F(x_1,x_2,\cdots,x_n)=\int_{-\infty}^{x_1}\int_{-\infty}^{x_2}\cdots\int_{-\infty}^{x_n}f(t_1,t_2,\cdots,t_n)dt_1dt_2\cdots dt_n.
$$

特别地, 若 $X_1,X_2,\cdots,X_n$ 相互独立同分布, 则依据独立的定义, 同分布的性质 $X_1=X_2=\cdots=X_n=X$ 有

$$
\begin{aligned}
&P\{X_1=x_1,X_2=x_2,\cdots,X_n=x_n\}\\
&=\prod_{i=1}^nP\{X_i=x_i\}=\prod_{i=1}^nP\{X=x_i\}.
\end{aligned}
$$

对于联合密度函数, 在密度连续点处 (否则几乎处处成立), 由独立的定义, 同分布的性质 $f_{X_1}=f_{X_2}=f_{X_n}=f$ 有

$$
\begin{aligned}
f(x_1,x_2,\cdots,x_n)&=\frac{\partial^nF(x_1,x_2,\cdots,x_n)}{\partial x_1\partial x_2\cdots\partial x_n}\\
&=\frac{dF_{X_1}}{dx_1}\frac{dF_{X_2}}{dx_2}\cdots\frac{dF_{X_n}}{dx_n}\\
&=\prod_{i=1}^nf_{X_i}(x_i)=\prod_{i=1}^nf(x_i).
\end{aligned}
$$

### 样本分布族

样本分布族是样本 $X_1,X_2,\cdots,X_n$ 的联合分布 $F_n(x_1,x_2,\cdots,x_n;\theta)$ 组成的集合

$$
\{F_n(x_1,x_2,\cdots,x_n;\theta):\theta\in\Theta\},
$$

其中 $\Theta$ 为参数空间, 是所有参数的取值集合.

### 常用分布表

常用分布表:

|Dis|P/f|E|D|
|:---:|:---:|:---:|:---:|
|$B(n,p)$|$P\{X=x\}=\binom{n}{x}p^x(1-p)^{n-x},\quad x=0,1,\cdots,n$|$np$|$np(1-p)$|
|$P(\lambda)$|$P\{X=x\}=\lambda^xe^{-\lambda}/x!,\quad x=0,1,s$|$\lambda$|$\lambda$|
|$G(p)$|$P\{X=x\}=(1-p)^{x-1}p,\quad x=1,2,s$|$1/p$|$(1-p)/p^2$|
|$U[a, b]$|$f(x)=1/(b-a)I_{[a, b]}(x),\quad x\in\mathbb R$|$(a+b)/2$|$(b-a)^2/12$|
|$E(\lambda)$|$f(x)=\lambda e^{-\lambda x}I_{(0,+\infty)}(x),\quad x\in\mathbb R$|$1/\lambda$|$1/\lambda^2$|
|$N(\mu,\sigma^2)$|$f(x)=1/(\sqrt{2\pi}\sigma)\exp\{-(x-\mu)^2/(2\sigma^2)\},\quad x\in\mathbb R$|$\mu$|$\sigma^2$|

### 期望与方差

使用

$$
E(X)=\sum_xxP\{X=x\},
$$

或

$$
E(X)=\int_{-\infty}^{+\infty}xf(x)dx,
$$

$$
\begin{aligned}
D(X)&=E(X-E(X))^2\\
&=E(X^2)+E((E(X))^2)-2E(XE(X))\\
&=E(X^2)-(E(X))^2.
\end{aligned}
$$

易证.

### 统计量与经验分布函数

统计模型包括参数空间和样本分布族.

统计量 $g(X_1,X_2,\cdots,X_n)$ 是样本的函数, 只依赖样本而不依赖任何其他未知量.

经验分布函数 $F_n(x)$ 形如

$$
F_n(x)=\frac1n\sum_{i=1}^nI_{\{X_i\le x\}}.
$$

### 期望与方差的性质

示性函数 $I_A$ 的数学期望:

$$
E(I_{\{X\in A\}})=1 P(X\in A)+0 P(X\in\overline A)=P(X\in A).
$$

期望的线性性:

$$
E(a\sum_{i=1}^nY_i)=a\sum_{i=1}^nE(Y_i).
$$

方差数乘公式:

$$
D(aY)=a^2D(Y).
$$

独立随机变量方差和等于各方差之和:

$$
\begin{aligned}
D(X_1+X_2)&=E((X_1+X_2)-E(X_1+X_2))^2\\
&=E((X_1-E(X_1))+(X_2-E(X_2)))^2\\
&=(X_1-E(X_1))^2+(X_2-E(X_2))^2+2(X_1-E(X_1))(X_2-E(X_2))\\
&=D(X_1)+D(X_2)+2\text{Cov}(X_1,X_2)\\
&=D(X_1)+D(X_2),
\end{aligned}
$$

归纳可得 $n$ 维情形:

$$
D(\sum_{i=1}^nX_i)=\sum_{i=1}^nD(X_i)
$$

## 抽样分布

### 正态分布的可加性

正态分布的可加性:

对于独立的随机变量 $X_1,X_2,\cdots,X_n$, 若

$$
X_i\sim N(\mu_i,\sigma_i^2),
$$

则

$$
\sum_{i=1}^nX_i\sim N(\sum_{i=1}^n\mu_i,\sum_{i=1}^n\sigma_i^2),
$$

使用特征函数易证(非独立便不成立), 其中 $X$ 的特征函数 $\varphi_X(t)$ 为

$$
\varphi_X(t)=E(e^{itX}),\quad t\in\mathbb R.
$$

### 三大抽样分布

$\chi^2$ 分布: 独立同分布的 $X_1,X_2,\cdots,X_n\sim N(0,1)$, 则

$$
\sum_{i=1}^nX_i^2\sim\chi^2(n).
$$

$t$ 分布: $X\sim N(0,1)$, $Y\sim\chi^2(n)$ 独立, 则

$$
\frac X{\sqrt{Y/n}}\sim t(n).
$$

$F$ 分布: $X\sim\chi^2(m)$, $Y\sim\chi^2(n)$ 独立, 则

$$
\frac{X/m}{Y/n}\sim F(m,n).
$$

### $\chi^2$ 分布的性质

若 $X\sim\chi^2(n)$, 则 $EX=n$, $DX=2n$.

Gamma 函数的性质:

$$
\Gamma(n+1)=n\Gamma(n).
$$

$\chi^2$ 分布具有再生性: 若相互独立的 $X_i\sim\chi^2(n_i)$, $i=1,2,\cdots,k$, 则

$$
\sum_{i=1}^kX_i\sim\chi^2(\sum_{i=1}^kn_i).
$$

数学期望为 $2$ 的指数分布是 $\chi^2$ 分布:

$$
E(\frac12)=\Gamma(1,\frac12)=\Gamma(\frac22,\frac12)=\chi^2(2),
$$

其中 $E(\lambda)=\Gamma(1,\lambda)$, $\chi^2(n)=\Gamma(n/2,1/2)$.

### 变量变换法

由于密度函数由积分定义, 因此其换元时要带微分:

$$
f_{U,V}(u,v)dudv=f_{X,Y}(x,y)dxdy,
$$

$$
f_{U,V}(u,v)=f_{X,Y}(x,y)|\frac{\partial(x,y)}{\partial(u,v)}|.
$$

### Stirling 公式与 F 分布

Stirling 公式:

$$
n!\sim\sqrt{2\pi n}(\frac ne)^n
$$

若 $X\sim F(m,n)$, 则 $1/X\sim F(n,m)$:

$$
X=\frac{U/m}{V/n}\sim F(m,n),
$$

$$
\frac1X=\frac{V/n}{U/m}\sim F(n,m).
$$

### 正态总体的抽样分布定理

独立同分布的 $X_1,X_2,\cdots,X_n\sim N(\mu,\sigma^2)$ 满足

$$
\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1),
$$

其中

$$
S^2=\frac1{n-1}\sum_{i=1}^n(X_i-\overline X)^2.
$$

若 $\mu$ 已知, 则可提升一个自由度:

$$
\frac{nS_\mu^2}{\sigma^2}\sim\chi^2(n),
$$

其中

$$
S_\mu^2=\frac1n\sum_{i=1}^n(X_i-\mu)^2.
$$

$T\sim t(n)$, 则

$$
E(T)=0,\quad n>1,
$$

$$
D(T)=\frac n{n-2},\quad n>2.
$$

### 极限定理

CLT: 设独立同分布的 $X_1, X_2, \cdots, X_n$ 均值为 $\mu$, 方差为 $\sigma^2$(有限), 则当 $n \to \infty$ 时,

$$
\frac{\overline{X}-\mu}{\sigma/\sqrt{n}} \xrightarrow{\mathcal{L}} N(0, 1)
$$

Khinchin 大数定律: 设独立同分布的 $X_1, X_2, \cdots, X_n$ 数学期望有限, 则样本均值依概率收敛于总体均值.

Slutsky 定理: 设 ${X_n}$ 和 ${Y_n}$ 是两列随机变量, 若 $X_n \xrightarrow{\mathcal{L}} X$, 且 $Y_n \xrightarrow{P} c$, 则

$$
X_n+Y_n\xrightarrow{\mathcal{L}}X+c,
$$

$$
X_nY_n\xrightarrow{\mathcal{L}}cX,
$$

$$
X_n/Y_n\xrightarrow{\mathcal{L}}X/c.
$$

### 边际分布与协方差

已知 $(X,Y)$ 的联合密度 $f_{X,Y}(x,y)$，要求 $X$ 的密度，只需对 $Y$ 积分消去：

$$
f_X(x)=\int f_{X,Y}(x,y)dy.
$$

协方差

$$
\text{Cov}(X, Y)=E((X-EX)(Y-EY)).
$$

若两随机变量独立, 则其协方差为 $0$.

协方差满足双线性性:

$$
\begin{aligned}
&\text{Cov}(aX + bY,cU + dV)\\
&=ac\text{Cov}(X,U)+ad\text{Cov}(X,V)\\
&+bc\text{Cov}(Y,U)+bd\text{Cov}(Y,V).
\end{aligned}
$$

本质上是分配律展开, 由期望的线性性易证.

### 指数族

指数族 $\{f(x;\theta)\}$ 的一般形式:

$$
f(x;\theta)=C(\theta)\exp\{\sum_{i=1}^kQ_i(\theta)T_i(x)\}h(x).
$$

自然形式:

$$
f(x;\varphi)=C(\varphi)\exp\{\sum_{i=1}^k\varphi_iT_i(x)\}h(x)
$$

自然参数空间:

$$
\Theta^*=\{\varphi:\int_{\mathcal X}\exp\{\sum_{i=1}^k\varphi_iT_i(x)\}h(x)dx<+\infty\},
$$

$\mathcal X$ 为样本空间.

### 充分统计量与完备统计量

充分统计量 $T(X_1,X_2,\cdots,X_n)$ 是指在其条件下的样本的条件分布与参数无关.

因子分解定理: $T(X_1,X_2,\cdots,X_n)$ 是充分统计量当且仅当

$$
f(x_1,x_2,\cdots,x_n;\theta)=g(T(x_1,x_2,\cdots,x_n),\theta)h(x_1,x_2,\cdots,x_n).
$$

Poisson 分布具有再生性:

若 $X_i\sim P(\lambda)$, $i=1,2,\cdots,n$, 则

$$
\sum_{i=1}^nX_i\sim P(n\lambda).
$$

使用特征函数易证.

完备统计量 $T(X_1,X_2,\cdots,X_n)$ 满足若 $E_\theta(\varphi(T))=0$, $\forall\theta$, 则 $P_\theta(\varphi(T)=0)=1$.

自然形式指数族中, 若参数空间中有内点, 则充分统计量完备.

## 点估计

### 矩估计与极大似然估计

矩估计核心思想是用样本矩代替总体矩.

似然函数 $L(\theta)$ 是以参数 $\theta$ 为自变量的联合分布列(密度函数):

$$
\begin{aligned}
L(\theta)&=L(\theta,x_1,x_2,\cdots,x_n)\\
&=\prod_{i=1}^nf(x_i,\theta),
\end{aligned}\quad\theta\in\Theta.
$$

似然函数 $L(\theta)$ 的最大值即为参数 $\theta$ 的极大似然估计量, 通常先求对数再求导计算.

### 无偏性、有效性与相合性

无偏估计:

$$
E\widehat\theta=\theta.
$$

有效性: 对于两个无偏估计, 谁方差更小谁更有效.

相合估计: 估计量 $\widehat{\theta_n}$ 依概率收敛到 $\theta$, 则称 $\widehat{\theta_n}$ 为 $\theta$ 的相合估计.

Chebyshev 不等式:

$$
P\{|X - \mu| \ge \varepsilon\} \le \frac{\sigma^2}{\varepsilon^2}
$$

即随机变量的尾概率受方差控制.

### UMVUE 与 L-S 定理

L-S 定理 1: $g(\theta)$ 的任一无偏估计 $\widehat{g(\theta)}$ 关于充分完备统计量 $T(X_1,X_2,\cdots,X_n)$ 的条件期望

$$
E(\widehat{g(\theta)}|T(X_1,X_2,\cdots,X_n))
$$

是 UMVUE.

L-S 定理 2: 充分完备统计量的无偏估计是 UMVUE.

$\overline{X}$ 是 $\mu$ 的 UMVUE, $S^2$ 是 $\sigma^2$ 的 UMVUE.

### Fisher 信息量与 C-R 下界

Fisher 信息量度量的是数据中包含的关于未知参数 $\theta$ 的信息的多少:

$$
I(\theta)=E(\frac{\partial\ln f(X,\theta)}{\partial\theta})^2=-E(\frac{\partial^2\ln f(X,\theta)}{\partial\theta^2}).
$$

C-R 下界:对于 C-R 正则分布族 $\{f(x,\theta)\}$, $g(\theta)$ 的无偏估计 $\widehat{g(X)}$ 有 C-R 下界

$$
D(\widehat{g(X)})\ge\frac{(g'(\theta))^2}{nI(\theta)},\quad\forall \theta\in\Theta^*.
$$

比值

$$
e_n(\theta)=\frac{(g'(\theta))^2/nI(\theta)}{D(\widehat{g(X)})}
$$

称为无偏估计 $\widehat{g(X)}$ 的效率, 当效率为 $1$ 时称为有效估计.

Gamma 函数反射公式:

$$
\Gamma(x)\Gamma(1-x)=\frac{\pi}{\sin(\pi x)},
$$

则 $\Gamma(1/2)=\sqrt\pi$.

## 区间估计

### 置信区间与置信限

$\theta$ 的区间估计 $[\widehat{\theta_1}(X_1,X_2,\cdots,X_n),\widehat{\theta_2}(X_1,X_2,\cdots,X_n)]$ 若满足

$$
P_\theta\{\widehat{\theta_1}\le\theta\le\widehat{\theta_2}\}\ge1-\alpha,
$$

则称其为置信水平为 $1-\alpha$ 的置信区间.

$\theta$ 的区间估计上下界

$$
\widehat{\theta_L}(X_1,X_2,\cdots,X_n)\text{ or }\widehat{\theta_R}(X_1,X_2,\cdots,X_n)
$$

若满足

$$
P_\theta\{\theta\ge\widehat{\theta_L}(X_1,X_2,\cdots,X_n)\}\ge1-\alpha
$$

或

$$
P_\theta\{\theta\le\widehat{\theta_R}(X_1,X_2,\cdots,X_n)\}\ge1-\alpha
$$

则称 $\widehat{\theta_L}$ 或 $\widehat{\theta_R}$ 为置信下或上限.

### 枢轴量法

可测函数 $G(X_1,X_2,\cdots,X_n,\theta)$ 若其分布不依赖 $\theta$ 则被称为枢轴量.

|情形|枢轴量|分布|
|:---:|:---:|:---:|
|单总体均值, $\sigma$ 已知|$\frac{\overline X-\mu}{\sigma/\sqrt n}$|$N(0,1)$|
|单总体均值, $\sigma$ 未知|$\frac{\overline X-\mu}{S/\sqrt n}$|$t(n-1)$|
|单总体方差, $\mu$ 已知|$\frac{nS_\mu^2}{\sigma^2}$|$\chi^2(n)$|
|单总体方差, $\mu$ 未知|$\frac{(n-1)S^2}{\sigma^2}$|$\chi^2(n-1)$|
|双总体均值差, $\sigma_1,\sigma_2$ 已知|$\frac{(\overline Y-\overline X)-(\mu_2-\mu_1)}{\sqrt{\sigma^2_1/m+\sigma_2^2/n}}$|$N(0,1)$|
|双总体均值差, $\sigma_1=\sigma_2$ 未知|$\frac{(\overline Y-\overline X)-(\mu_2-\mu_1)}{S_\omega\sqrt{1/m+1/n}}$|$t(m+n-2)$|
|双总体均值差, $\sigma_1$, $\sigma_2$ 未知但 $m=n$|$\frac{\overline Z-(\mu_2-\mu_1)}{S_z/\sqrt{n}}$|$t(m+n-2)$|
|双总体方差比, $\mu_1,\mu_2$ 已知|$\frac{S_{\mu1}^2/S_{\mu2}^2}{\sigma_1^2/\sigma_2^2}$|$F(m,n)$|
|双总体方差比, $\mu_1,\mu_2$ 未知|$\frac{S_1^2/S_2^2}{\sigma_1^2/\sigma_2^2}$|$F(m-1,n-1)$|

其中

$$
\overline X=\frac1n\sum_{i=1}^nX_i,
$$

$$
S^2=\frac1{n-1}\sum_{i=1}^n(X_i-\overline X)^2,
$$

$$
S_\mu^2=\frac1n\sum_{i=1}^n(X_i-\mu)^2,
$$

$$
S_\omega^2=\frac{(m-1)S_1^2+(n-1)S_2^2}{m+n-2}.
$$

### 指数分布与大样本区间估计

对于 $X\sim E(\lambda)=\Gamma(1,\lambda)$,

$$
2n\lambda\overline X\sim\chi^2(2n).
$$

证明:

$$
2n\lambda\overline X=2\lambda\sum_{i=1}^nX_i\sim\Gamma(n,\frac{\lambda}{2\lambda})=\Gamma(n,\frac12)=\chi^2(2n).
$$

其中, 对于 $S\sim G(n,\lambda)$,

$$
Y=cS\sim G(n,\lambda/c).
$$

证明:

$$
\begin{aligned}
f_Y(y) &= f_S(s) |\frac{ds}{dy}|\\
&= \frac{\lambda^n}{\Gamma(n)} (\frac{y}{c})^{n-1}e^{-\lambda\frac{y}{c}}\frac{1}{c}\\
&= \frac{(\lambda/c)^n}{\Gamma(n)}y^{n-1}e^{-(\lambda/c)y}.
\end{aligned}
$$

二项分布与 Poisson 分布在大样本下使用 CLT 逼近 $N(0,1)$.

## 假设检验

### 两类错误与功效函数

第一类错误 $\alpha$: 原假设 $H_0$ 成立时拒绝原假设;

第二类错误 $\beta$: 备择假设 $H_1$ 成立时接受原假设.

功效函数

$$
\varphi(x) = P\{x\in D\},
$$

其中 $D$ 为拒绝域, 则

$$
\alpha=\varphi(\theta)I_{\{\theta\in\Theta_0\}},
$$

$$
\beta=(1-\varphi(\theta))I_{\{\theta\in\Theta_1\}},
$$

显著性水平是犯第一类错误的最大概率.

### 正态总体的假设检验

提出假设 $H_0$(必须带等号), $H_1$; 构造检验统计量, 使得其分布与参数无关; 确定拒绝域(与 $H_1$ 同向); 确定拒绝域的临界 $k$; 做出判断是否拒绝 $H_0$.

|情形|检验统计量|分布|
|:---:|:---:|:---:|
|单总体均值, $\sigma$ 已知|$\frac{\overline X-\mu}{\sigma/\sqrt n}$|$N(0,1)$|
|单总体均值, $\sigma$ 未知|$\frac{\overline X-\mu}{S/\sqrt n}$|$t(n-1)$|
|单总体方差, $\mu$ 已知|$\frac{nS_\mu^2}{\sigma^2}$|$\chi^2(n)$|
|单总体方差, $\mu$ 未知|$\frac{(n-1)S^2}{\sigma^2}$|$\chi^2(n-1)$|
|双总体均值差, $\sigma_1,\sigma_2$ 已知|$\frac{(\overline Y-\overline X)-(\mu_2-\mu_1)}{\sqrt{\sigma^2_1/m+\sigma_2^2/n}}$|$N(0,1)$|
|双总体均值差, $\sigma_1=\sigma_2$ 未知|$\frac{(\overline Y-\overline X)-(\mu_2-\mu_1)}{S_\omega\sqrt{1/m+1/n}}$|$t(m+n-2)$|
|双总体均值差, $\sigma_1$, $\sigma_2$ 未知但 $m=n$|$\frac{\overline Z-(\mu_2-\mu_1)}{S_z/\sqrt{n}}$|$t(m+n-2)$|
|双总体方差比, $\mu_1,\mu_2$ 已知|$\frac{S_{\mu1}^2/S_{\mu2}^2}{\sigma_1^2/\sigma_2^2}$|$F(m,n)$|
|双总体方差比, $\mu_1,\mu_2$ 未知|$\frac{S_1^2/S_2^2}{\sigma_1^2/\sigma_2^2}$|$F(m-1,n-1)$|

其中

$$
\overline X=\frac1n\sum_{i=1}^nX_i,
$$

$$
S^2=\frac1{n-1}\sum_{i=1}^n(X_i-\overline X)^2,
$$

$$
S_\mu^2=\frac1n\sum_{i=1}^n(X_i-\mu)^2,
$$

$$
S_\omega^2=\frac{(m-1)S_1^2+(n-1)S_2^2}{m+n-2}.
$$

对于 $X\sim E(\lambda)=\Gamma(1,\lambda)$,

$$
2n\lambda\overline X\sim\chi^2(2n).
$$

由于指数分布 $E(\lambda)$ 的均值 $\mu=1/\lambda$, 因此若在 $\lambda$ 上做假设, 此时的拒绝域与 $H_1$ 的方向相反.

Poisson 分布在大样本下使用 CLT 逼近 $N(0,1)$.

二项分布在大样本下使用 CLT 逼近 $N(0,1)$.

### 区间估计与假设检验的对偶性

区间估计与假设检验的对偶性: 一个参数值 $\theta_0$ 被一个 $1-\alpha$ 水平的置信区间所包含, 当且仅当以 $\theta_0$ 为零假设的双侧检验在显著性水平 $\alpha$ 下不被拒绝, 这也解释了为什么枢轴量与检验统计量的构造形式相同.
