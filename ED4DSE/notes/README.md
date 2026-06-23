# 📚 Experimental Design for Data Science and Engineering (ED4DSE) 学习笔记

> 教材：*Experimental Design for Data Science and Engineering* — V. Roshan Joseph  
> 本笔记基于 PyED4DSE 项目编写，覆盖全书11章全部知识点、代码示例和习题

---

## 📖 章节索引

| 章节 | 主题 | 笔记 |
|:----:|------|:----:|
| 第1章 | **Introduction** — 实验设计导论 | [📝 ch01.md](./ch01.md) |
| 第2章 | **Modeling** — 建模基础 | [📝 ch02.md](./ch02.md) |
| 第3章 | **Model-based Designs** — 基于模型的设计 | [📝 ch03.md](./ch03.md) |
| 第4章 | **Space-Filling Designs** — 空间填充设计 | [📝 ch04.md](./ch04.md) |
| 第5章 | **Representative Points** — 代表点 | [📝 ch05.md](./ch05.md) |
| 第6章 | **Screening Designs** — 筛选设计 | [📝 ch06.md](./ch06.md) |
| 第7章 | **Sequential Designs** — 序贯设计 | [📝 ch07.md](./ch07.md) |
| 第8章 | **Multilevel & Mixture Designs** — 多水平与混料设计 | [📝 ch08.md](./ch08.md) |
| 第9章 | **Model Calibration** — 模型校准 | [📝 ch09.md](./ch09.md) |
| 第10章 | **Data Subsampling** — 数据子采样 | [📝 ch10.md](./ch10.md) |
| 第11章 | **Data Analysis** — 数据分析 | [📝 ch11.md](./ch11.md) |

---

## 🎯 学习路线

1. **第1-2章**：了解实验设计的四大目标（逼近粗糙/光滑函数、优化、不确定性传播）和建模基础（多项式插值、RBF插值、克里金、高斯过程）
2. **第3-5章**：学习如何设计实验点——基于模型的设计、空间填充设计、代表点方法
3. **第6章**：筛选重要变量的方法（Sobol'敏感性、Morris方法、MOFAT）
4. **第7章**：序贯设计——如何在已有数据基础上逐步添加新点
5. **第8章**：处理多水平因子和混料约束的特殊设计
6. **第9章**：模型校准——用实验数据校正计算机模型参数
7. **第10章**：大数据场景下的数据子采样方法
8. **第11章**：数据分析——因子选择、FIRST算法、TwinGP

---

## 🔑 核心概念速查

| 概念 | 简要说明 | 章节 |
|:----:|----------|:----:|
| 实验设计 (DoE) | 系统性地选择输入点以达到特定目标 | Ch1 |
| 高斯过程 (GP) | 一种灵活的贝叶斯非参数模型 | Ch2 |
| Kriging | 基于高斯过程的插值方法 | Ch2 |
| IMSE | 积分均方误差，衡量预测精度 | Ch3 |
| MMSE | 最大均方误差，最坏情况下的预测精度 | Ch3 |
| 空间填充设计 | 让实验点在输入空间均匀分布 | Ch4 |
| LHD | 拉丁超立方体设计 | Ch4 |
| MaxPro | 最大化投影距离的准则 | Ch4 |
| 支持点 (Support Points) | 用少量点代表一个分布 | Ch5 |
| Sobol'敏感性 | 基于方差分解的敏感性分析 | Ch6 |
| Morris方法 | 经济高效的因子筛选方法 | Ch6 |
| ALC/ALMV/ALM | 主动学习序贯设计准则 | Ch7 |
| EI | 期望改善，用于优化 | Ch7 |
| 半正态图 | 两水平析因设计的效应分析方法 | Ch8 |
| HiGarrote | 贝叶斯变量选择方法 | Ch8 |
| ICAOD | 最优实验设计的迭代算法 | Ch9 |
| SPlit | 最优数据分割方法 | Ch10 |
| Data Twinning | 创建统计特性相似的子集 | Ch10 |
| FIRST | 相关数据中的因子选择算法 | Ch11 |
| TwinGP | 双高斯过程回归 | Ch11 |

---

## 📊 常用符号约定

| 符号 | 含义 |
|:----:|------|
| $n$ | 实验点数量 |
| $p$ | 输入变量（因子）维度 |
| $D$ | 设计矩阵，$n \times p$ |
| $x$ | 输入变量 |
| $y$ | 输出响应 |
| $f(x)$ | 真实未知函数 |
| $\hat{y}(x)$ | 预测值 |
| $\theta$ | 高斯核相关长度参数 |
| $\sigma^2$ / $\tau^2$ | 过程方差 |
| $\Sigma$ / $R$ | 协方差矩阵 / 相关矩阵 |
| $\beta$ | 回归系数 |
| $\rho$ | 相关系数 |

---

> 💡 **学习建议**：每章笔记末尾都有"例题详解"和"练习题"部分。请先通读知识点，然后独立完成练习题，再对照解析检查。
