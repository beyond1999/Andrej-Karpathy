**Batch Normalization (BN)** 

BN出现的问题是，本来单个样本只与神经网络有关，但是用了BN，一个样本需要进行一定的偏移，而偏移大小与这个样本所在的batch有关



## 1. 概念

**Batch Normalization** 的核心思想是：
 👉 在训练过程中，每一层的输入分布会不断漂移（因为前面层的参数在变），这会导致训练不稳定。BN 就是把每一层的输入先“归一化”，再让模型学会自己调整合适的尺度。

------

## 2. 数学公式

假设某一层的输入（pre-activation）是 (h)，BN 的步骤是：

1. **计算 batch 内的均值和方差**
   $$
   \mu_B = \frac{1}{m}\sum_{i=1}^{m} h_i,
    \quad
    \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(h_i - \mu_B)^2
   $$

2. **标准化**（得到零均值、单位方差）
   $$
   \hat{h}_i = \frac{h_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
   $$

3. **缩放 + 平移**（可学习参数 (\gamma, \beta)）
   $$
   y_i = \gamma \hat{h}_i + \beta
   $$

其中：

- gamma = `bngain`，让网络学会调节尺度。
- beta= `bnbias`，让网络学会调节偏移。

------

## 3. 为什么要用 BN？

- **缓解梯度消失/爆炸**：输入归一化后，数值稳定，梯度传播更顺畅。
- **加快训练**：可以用更大学习率，不容易训练崩掉。
- **减少对初始化的依赖**：不用特别小心地挑选权重初始分布。
- **有正则化效果**：因为每个 batch 的均值和方差不同，带来随机性，类似 Dropout。

------

## 4. 怎么用 BN？

通常放在 **线性层（或卷积层）之后、激活函数之前**：

```
z = Wx + b
ẑ = BN(z)
h = activation(ẑ)
```

在代码里：

```python
nn.Sequential(
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),   # 对每个神经元的输出做 BN
    nn.ReLU()
)
```

对于卷积：

```python
nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3),
    nn.BatchNorm2d(128),  # 对每个通道做 BN
    nn.ReLU()
)
```

------

