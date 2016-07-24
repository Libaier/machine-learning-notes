# logistic回归

标签（空格分隔）： 分类 有监督学习

---

#### 主要思想

使用线性回归做分类。

#### 产生背景

Logistic regression was developed by statistician David Cox in 1958。

#### 应用场景

* 分类问题

#### 核心理解

* 使用主要推导中的假设作为概率假设。
至于为什么是用这个假设，可以参考[从Bayesion的角度来看Logistic Regression](http://kubicode.me/2016/03/26/Machine%20Learning/Bayesian-Logistic-Regression/),在看[正态分布的前世今生](http://www.flickering.cn/%E6%95%B0%E5%AD%A6%E4%B9%8B%E7%BE%8E/2014/06/%E7%81%AB%E5%85%89%E6%91%87%E6%9B%B3%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83%E7%9A%84%E5%89%8D%E4%B8%96%E4%BB%8A%E7%94%9F%E4%B8%8B/)，感觉正态分布实在强，无敌。
* 损失函数叫做Cross-Entropy Error。

#### 主要推导
我们使用如下假设作为分类概率。
$$\eqalign{
  & \pi (x) = P(Y = 1|x) = {{{e^{(wx + b)}}} \over {1 + {e^{(wx + b)}}}}  \cr 
  & 1 - \pi (x) = P(Y = 0|x) = {1 \over {1 + {e^{(wx + b)}}}} \cr} $$
我们希望理想的参数可以以最大化的概率产生训练集D(极大似然估计法)，则优化问题转化为

$$\max \prod\limits_{i = 1}^N {\pi {{(x)}^{{y_i}}}{{(1 - \pi (x))}^{(1 - {y_i})}}} $$

连乘转化为连加

#### 求解算法

可使用梯度下降，随机梯度下降，牛顿，拟牛顿求解。

#### 伪代码

```
这里假设数据线性可分
输入:数据集D(N个样本)

过程:

使用主要推导中公式生成优化函数
使用优化方法求解

输出:训练好的模型(w,b)

```

#### 复杂度分析

时间复杂度：与优化算法复杂度相同

#### 大数据下改进

to be done...

#### 评价

* 优点

  * 实现简单；
  * 分类时计算量非常小，速度很快，存储资源低，也很容易并行；
  * 在处理分类问题的同时还可能给出一个概率值
  * 优化方法多：除了GD和SGD，应该还有拟牛顿法、BFGS、L-BFGS

* 缺点

  * 容易欠拟合，一般准确度不太高
  * 只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；

#### 算法改进

* 加入正则
* [Softmax回归??](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)

#### 参考资料

1. [机器学习常见算法个人总结](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)


  [1]: http://www.sersc.org/journals/IJDTA/vol7_no1/5.pdf

