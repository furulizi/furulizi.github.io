---
layout: post
title: "分类线性模型"
date: 2018-06-10 17:07:20 +0300
description: PRML 第四章 分类线性模型 # Add post description (optional)
img:  # Add image post (optional)
tags: [PRML]
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# PRML 第四章 分类线性模型
> 分类的目标是将输入变量 $x$分到 $K$个离散的类别 $C_{k}$ 的某一类。线性模型是指决策面是输入向量 $x$ 的线性函数，因此被定义为 $D$ 维输入空间中的 $D-1$ 维超平面。如果数据集可以被线性决策面精确地分类，那么我们就说数据集是线性可分的。

## 本章概要
1. Fisher准则的分类,以及它和最小二乘分类的关系 (Fisher分类是最小二乘分类的特例)
2. 概率生成模型的分类模型
3. 概率判别模型的分类模型
4. 全贝叶斯概率的Laplace近似

## 判别式模型
&emsp;&emsp;回忆一下概率生成模型的思路：如果二分类问题中 $P(C_{k}|\mathbf{x})>P(C_{j}|\mathbf{x})$ ，则我们认为 $x$ 属于 $k$ 类，于是我们希望求解 $P(C_{k}|\mathbf{x})$ ,则通过贝叶斯公式，把问题转化为求解 $P(\mathbf{x}|C_{k})$ 与 $P(C_{k})$ ，再假设类条件概率密度为高斯分布，最后用MLE求解后验概率。不妨算一算，生成式模型需要求MLE的参数个数（非常多，关于 $M$ 的二次变化）。于是会有一个简单直接的想法，直接计算 $P(C_{k}|\phi(\mathbf{x}))$ 。

### 1. mapping的例子
如果需要直接计算 $P(C_{k}|\mathbf{x})$ ，我们需要什么条件？当然是样本数据中能找到线性决策面把不同类别的点分开。那么如果样本数据在原空间中非线性可分，就需要通过基函数对原空间进行映射，如以下的例子。


<div align='center'>
	<img src='https://raw.githubusercontent.com/furulizi/furulizi.github.io/master/assets/img/prml-fig4-12.png' />
</div>

这幅图我们有必要重新解读一下：左图的是原始输入空间 $(x_{1},x_{2})$ ，其中有两个分类，分别用红点和蓝点表示；绿色圆圈为二维高斯基函数在 $(x_{1},x_{2})$ 上的投影，绿色十字标志为高斯基函数的中心，左下的为 $\phi_{1}(\mathbf{x})$ ,中间的为 $\phi_{2}(\mathbf{x})$ ；右图给出的则是特征空间 $(\phi_{1}(\mathbf{x}),\phi_{2}(\mathbf{x}))$ 的映射，黑色直线正是左图中的黑色圆圈。注意一下：左图中左下角的蓝点对应的是右图中的右下角的一堆蓝点。

### 2. logistic Regression
还记得概率生成式模型中曾得到一个结论：即使是一般情况下，类别 $C_{1}$ 的后验概率写成作用在特征向量 $\phi$ 的线性函数上的 logistic sigmoid 函数的形式，
$$
\begin{align}
p(C_{1}|\phi) = y(\phi) = \sigma(\omega^T\phi)
\end{align}
$$
这里的 $\sigma(\cdot)$ 是logistic sigmoid函数。

我们现在从上式开始，用MLE的思路可以得到以下推导：
$$
p(\mathbf{t}|\omega) = \sum_{n=1}^{N}y_{n}^{t_{n}}(1-y_{n})^{1-t_{n}}
$$
取似然函数的负对数，得到cross-entropy error function：
$$
E(\omega) = -\ln p(\mathbf{t}|\omega) = -\sum_{n=1}^{N}{t_{n}\ln y_{n} + (1-t_{n})\ln (1-y_{n})}
$$
代入 $y_{n} = \sigma(a_{n})$ 以及 $a_{n} = \omega^T\phi_{n}$ ，
$$\begin{split}
\nabla E(\omega)
&= -\sum_{n=1}^{N}(\frac{t_{n}}{y_{n}} \frac{\partial y_n}{\partial \omega} - \frac{1-t_n}{1-y_n} \frac{\partial y_n}{\partial \omega}) \\
&= \sum_{n=1}^{N}(\frac{t_{n}}{\sigma} \sigma(1- \sigma)\phi_n - \frac{1-t_n}{1-\sigma}  \sigma(1-\sigma)\phi_n) \\
&=\sum_{n=1}^{N}\{\frac{t_{n}}{\sigma} \sigma(1- \sigma) - \frac{1-t_n}{1-\sigma}  \sigma(1-\sigma)\}\phi_n \\
&= \sum_{n=1}^{N}(y_{n} - t_{n})\phi_{n} \\
\end{split}$$

到这步我们发现logistic sigmoid的导数因子已经消去了，得到一个漂亮的log likelihood function，所以数据点 $n$ 对梯度的贡献就在于目标值与模型预测值之间的“误差”与基函数的乘积。
### 3. Newton-Raphson Update
将上式写成矩阵形式：
$$
\nabla E(\omega) = \sum_{n=1}^{N}(y_{n} - t_{n})\phi_{n} = 	\Phi^T(\mathbf{y} - \mathbf{t})
$$
其中 $\Phi$ 是design matrix，$\Phi_{nj} = \phi_j(x_n)$ ，再由
$$
\frac {d\sigma}{da} = \sigma(1-\sigma)
$$
得
$$
H = \nabla \nabla E(\omega) = \sum_{n=1}^N y_{n}(1-y_{n})\phi_{n}\phi_{n}^T = \Phi^TR \Phi
$$
其中 $R_{nn} = y_n(1-y_n)$ 是一个 $N \times N$ 的对角线矩阵，而 $H$ 被称为 Hessian矩阵。 这是的Hessian不是一个常量，它将受weighting matrix $R$ 影响，实际依赖 $\omega$ 。这正好说明了一点：$E(\omega)$ 不是一个二次函数。

这样，我们就能得到 $\omega$ 的更新公式了：

$$\begin{split}
\omega^{new} = &\omega^{old} - (\Phi^T R \Phi)^{-1}\Phi^T(\mathbf{y}-\mathbf{t}) \\
=&(\Phi^TR\Phi)^{-1}\{\Phi^TR\Phi \omega^{old} - \Phi^T(\mathbf{y}-\mathbf{t})\} \\
=& (\Phi^TR\Phi)^{-1}\Phi^{T}R \mathbf{z}\\
\end{split}$$
其中， $z$ 是一个 $N$ 维向量， $z=\Phi \omega^{old} - R^{-1}(\mathbf{y}-\mathbf{t})$ ，这个算法叫做 iterative reweighted least squares(IRLS Rubin,1983).


试证明：我们知道对于任意向量 $u$ 都有 $u^THu>0$ ,所以Hessian矩阵是正定的。误差函数 $E(\omega)$ 是凸函数，以及它有唯一的最小值。

证明：已知 $p(C_{1}|\phi) = y(\phi) - \sigma(\omega^T\phi)$ 是有界的(sigmoid函数有界)，那么 $R$ 对角线上所有元素都是严格大于0的，那么矩阵 $R$ 是正定矩阵, $R^{1/2}$ 也是正定矩阵，所以对于任意的向量 $u$ ,
$$
u^T \Phi^T R \Phi u = (u^T \Phi^T R^{1/2})(R^{1/2} \Phi u) = \|R^{1/2} \Phi u\|^2 > 0
$$
也就证明 $\Phi^T R \Phi$ 是正定的。

另一方面，考虑 $E(\omega)$ 在最小值点 $\omega^{*}$ 处的泰勒展开式（一次项在  $\omega^{*}$ 处为零），
$$
E(\omega) = E(\omega^{*}) + \frac{1}{2}(\omega - \omega^{*})^T H(\omega - \omega^{*})
$$
现在我们令 $\omega = \omega^{*} + \lambda v$, 其中 $v$ 是在 $\omega$ 空间中的任意非零向量，那么上式变成：
$$\begin{split}
E(\omega) = &E(\omega - \lambda v) + \frac {1}{2}(\lambda v)^T H (\lambda v)\\
=&E(\omega - \lambda v) + \frac {1}{2}\lambda^{2} v^T H v
\end{split}$$
考虑
$$
\frac{\partial^{2}E}{\partial \lambda^{2}} = v^T H v > 0
$$
所以我们说 $E{\omega}$ 是凸函数。再者，如果要求 $E(\omega)$ 最小，则要令展开式的而此项为零，即
$$
H(\omega - \omega^*) = 0
$$
由于 $H$ 是正定的，且 $H^{-1}$ 存在（ $H = \Phi^{T}R\Phi$ ），那么 \omega = \omega^* ，证毕。


### 4. probit 回归
类条件概率分布是指数族分布描述的类条件概率分布最终能够求出后验类概率，且为在特征变量的线性函数上的logisitc 变换。 但是针对类条件概率密度是高斯混合模型建模时，最终的变换将是probit 变换。

### 5.扩展到多分类问题和指数族问题（略）

## 生成式模型与判别式的区别
1. Probabilistic Generative Models，通过MAP方式建立概率模型，需要先验概率 $$P(C_{k})$$ ,类条件概率和边缘概率 
$$
P(\mathbf{x}|C_{k})
$$
。

2. 判别式模型是直接求 
$$
P(C_{k}|\phi(\mathbf{x}))
$$
。

3. 如果上述2中的 $$\phi(\mathbf{x}) = \mathbf{x}$$，那么从生成式模型中会得到判别式模型，而且生成式模型中的MLE与判别式模型中使用的MLE推倒梯度是类似的。

## 总结贝叶斯

- 经验贝叶斯
$$
p(\omega|\mathbf{X}) =\frac{p(\mathbf{X}|\omega)p(\omega)}{\int_{\omega}p(\mathbf{X}|\omega)p(\omega)d(\omega)}
$$
你要从经验中获得 $p(\omega)$ .


- MAP贝叶斯
MAP方法的贝叶斯先求出使得marginal likelihood 最大化的参数 $$\alpha^*$$ 和 $$\beta^*$$ ，然后让hyper-parameter 取固定的值 $$\alpha^*$$ 和 $$\beta^*$$ ，再对 $$\mathbf{w}$$ 进行 marginalize:
$$
p(t|\mathbf{t})\approx p(t|\mathbf{t}.\alpha^*.\beta^*) = \int p(t|\mathbf{w}.\beta^*)p(\mathbf{w}|\mathbf{t}.\alpha^*.\beta^*)d\mathbf{w}
$$
这里 $$\alpha^*$$ 和 $$\beta^*$$ 为超参数。
