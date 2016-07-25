# SVM

标签（空格分隔）： 有监督学习 分类

---

#### 主要思想

定义在特征空间上间隔最大的线性分类器。

#### 产生背景

The original SVM algorithm was invented by Vladimir N. Vapnik and Alexey Ya. Chervonenkis in 1963. In 1992, Bernhard E. Boser, Isabelle M. Guyon and Vladimir N. Vapnik suggested a way to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard incarnation (soft margin) was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.

1963-原始理论
1992-核技巧
1993-软间隔理论

#### 应用场景

* SVM-原始只能解决二分类问题，修改后也可以解决多分类问题
* SVR-回归问题

#### 核心理解

* 单层神经网络
* 线性支持向量机其实和logistic回归没那么大区别

#### 主要推导

**线性可分支持向量机**

首先我们定义一种间隔叫函数间隔

$$\eqalign{
  & {{\hat \gamma }_i} = {y_i}(w{x_i} + b)  \cr 
  & \hat \gamma  = \mathop {\min }\limits_{i = 1...N} {{\hat \gamma }_i} \cr} $$
我们会发现，等比例缩放w，b超平面没改变，但函数间隔发生了变化。

然后我们再定义一种间隔叫（带符号的）几何间隔

$$\eqalign{
  & {\gamma _i} = {y_i}{{(w{x_i} + b)} \over {||w||}}  \cr 
  & \gamma  = \mathop {\min }\limits_{i = 1...N} {\gamma _i} \cr} $$
  
我们可以推出
  
$$\gamma  = {{\hat \gamma } \over {||w||}}$$

线性可分支持向量机是找一个超平面，在满足正确分类的基础上，所有样本的几何间隔最大。则求解问题变为

$$\eqalign{
  & \mathop {\max }\limits_{w,b} \gamma   \cr 
  & s.t.  \qquad{y_i}{{(w{x_i} + b)} \over {||w||}} \ge \gamma {\qquad i = 1...N} \cr} $$

我们把式中的一部分替换为前边提到的函数间隔

$$\eqalign{
  & \mathop {\max }\limits_{w,b} {{\hat \gamma } \over {||w||}}  \cr 
  & s.t.\qquad {y_i}(w{x_i} + b) \ge \hat \gamma {\qquad   i = 1...N} \cr} $$

前边提到函数间隔的取值对超平面没有影响，这里我们把函数间隔取值为1，且$\max {1 \over {||w||}}$和$\min {1 \over 2}||w|{|^2}$等价，于是原问题可以转化为

$$\eqalign{
  & \min\limits_{w,b} {1 \over 2}||w|{|^2}  \cr 
  & s.t.\qquad {y_i}(w{x_i} + b) - 1 \ge 0{\qquad i = 1...N} \cr} $$

这是一个凸二次规划问题，可以通过对应的优化算法求解。

但我们还不满足，如果求解使用这个问题拉格朗日函数的对偶问题，我们发现可以更容易求解，而且可以自然的引入核函数。

拉格朗日函数：

$$L(w,b,\alpha ) = {1 \over 2}||w|{|^2} - \sum\limits_{i = 1}^N {{y_i}(w{x_i} + b){\alpha _i} + } \sum\limits_{i = 1}^N {{\alpha _i}} $$
$$s.t.\qquad{\alpha _i} \ge 0\qquad i = 1....N$$

对偶问题求极大极小：
$$\mathop {\max }\limits_\alpha  \mathop {\min }\limits_{w,b} L(w,b,\alpha )$$
  
最终求得的优化函数为
$$\eqalign{
  & \mathop {\min {1 \over 2}}\limits_\alpha  \sum\limits_{i = 1}^N {\sum\limits_{j = 1}^N {{\alpha _i}} } {\alpha _j}{y_i}{y_j}({x_i} \cdot {x_j}) - \sum\limits_{i = 1}^N {{\alpha _i}}   \cr 
  & s.t\qquad\sum\limits_{i = 1}^N {{\alpha _i}{y_i}}   \cr 
  & \qquad\qquad{\alpha _i} \ge 0,i = 1....N \cr} $$
  
最终先计算出最优${{\alpha}}$，然后通过此${{\alpha}}$计算对应的w,b

**线性支持向量机和软间隔最大化**
解释一：
为了满足有些样本函数间隔不能满足大于等于1的情况，为每个样本引入一个松弛变量$\xi$。

$$\eqalign{
  & \mathop {\min }\limits_{w,b,\xi } {1 \over 2}||w|{|^2} + C\sum\limits_{i = 1}^N {{\xi _i}}   \cr 
  & s.t.\qquad{y_i}(w{x_i} + b) - 1 + {\xi _i} \ge 0\qquad{i = 1...N}  \cr 
  & \qquad\qquad{\xi _i} \ge 0\qquad{{i = 1...N}} \cr} $$

支持向量是求解的结果中$\alpha _i>0$大于零对应的实例$x_i$？？。

解释二：
$$\sum\limits_{i = 1}^N {{{[1 - {y_i}(w{x_i} + b)]}_ + } + \lambda ||w|{|^2}} $$
其中
$${[z]_ + } = \left\{ \matrix{
  z,z > 0  \cr 
  0,z \le 0  \cr}  \right.$$

合页损失如下表示
$$L({y_i}(w{x_i} + b)) = {[1 - {y_i}(w{x_i} + b)]_ + }$$
式子右边表示L2正则

周志华P132讨论？

**核技巧**
令核函数为如下函数，其中$\phi (x)$为输入空间向另外一个特征空间的映射。

$$K(x,z) = \phi (x) \cdot \phi (z)$$

前边的求解目标可以替换为如下，

$$\mathop {\min }\limits_\alpha  {1 \over 2}\sum\limits_{i = 1}^N {\sum\limits_{j = 1}^N {{\alpha _i}} } {\alpha _j}{y_i}{y_j}K({x_i} \cdot {x_j}) - \sum\limits_{i = 1}^N {{\alpha _i}} $$

$K(x,z)$应满足是正定核函数。
常见的核函数有
1.多项式核函数
2.高斯核函数
3.字符串核函数

#### 求解算法

对应二次凸优化问题解法

SMO不断将原二次规划问题分解为只有两个变量的二次规划子问题，并对子问题进行解析求解，直到所有变量满足KKT条件??为止，这种启发式算法总体上比较高效。

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

  * 使用核函数可以向高维空间进行映射，可以解决非线性的分类
  * 分类思想很简单，就是将样本与决策面的间隔最大化，分类效果较好

* 缺点

  * 无法直接支持多分类，但是可以使用间接的方法来做

#### 算法改进

* 如何输出概率？？
* 回归问题？？
* 多分类？？

#### 参考资料

1. [机器学习常见算法个人总结](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)







