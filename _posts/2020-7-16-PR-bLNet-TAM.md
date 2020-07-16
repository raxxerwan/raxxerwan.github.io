---
layout: post
title: 【Paper Reading】bLVNet+TAM
---

[**More Is Less: Learning Efficient Video Representations by Big-Little Network and Depthwise Temporal Aggregation**](https://arxiv.org/pdf/1912.00869.pdf)

# 动机

这篇文章的出发点我认为有两点：

1. 之前的许多视频模型已经说明，在模型容量可容纳的范围内，训练时使用更稠密的帧数能得到更好的结果。但是一昧增大输入帧数也可能带来许多问题。比如训练时间太长；或者由于模型容量问题，太大的输入帧数反而会掉点（如TSN-ResNet50在32帧输入的情况下就会掉点）。因此，该文章使用bLNet作为backbone。bLNet-ResNet50和ResNet50具有几乎相同的参数量。但是通过给bLNet的bigNet和LittleNet分别使用相邻的两帧输入，可以在不增加计算量的情况下将输入帧数double。

2. TSM的优异表现证明了，即使是在单纯的类TSN的2D网络中，如果能够很好地做到时序建模（比如TSM的通道平移），模型也能够达到媲美3D网络的效果。因此这篇文章提出了TAM模块，该模块可以认为是TSM模块的泛化形式，在不显著增加参数量和计算量的情况下，更好地对帧与帧之间的时序信息进行建模。


# 结构

![结构](https://raw.githubusercontent.com/raxxerwan/raxxerwan.github.io/master/images/2020-7-16-PR-bLNet-TAM/structrue.JPG)

bLVNet-TAM沿用了TSN的2D框架，只是将backbone换成了bLNet。同时将TSM模块替换成了TAM模块。

## 输入

bLNet分为两个子网络bigNet和LittleNet。因此可以给bLNet输入相邻的两帧。由于bigNet的计算量比LittleNet大，为了平衡计算量，给bigNet的输入帧分辨率是LittleNet的输入帧分辨率的一半。输入帧数的典型值为8 * 2以及16 * 2，即TSN的两倍。

## Temporal Aggregation Module (TAM)

TAM对TSM做了一个泛化。TAM的计算公式为：

![TAM公式](https://raw.githubusercontent.com/raxxerwan/raxxerwan.github.io/master/images/2020-7-16-PR-bLNet-TAM/TAM_form.JPG)

其中r为当前帧能够看到的前后帧的数量。<a href="https://www.codecogs.com/eqnedit.php?latex=y_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_t" title="y_t" /></a>为第t帧的特征图。<a href="https://www.codecogs.com/eqnedit.php?latex=w_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_t" title="w_t" /></a>为一个一维向量，其长度与<a href="https://www.codecogs.com/eqnedit.php?latex=y_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_t" title="y_t" /></a>的通道数一致。可以发现其实<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;w_j&space;\bigotimes&space;y_{t&space;&plus;j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;w_j&space;\bigotimes&space;y_{t&space;&plus;j}" title="w_j \bigotimes y_{t +j}" /></a>就是对特征图<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;y_{t&space;&plus;j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;y_{t&space;&plus;j}" title="y_{t +j}" /></a>做一个1 * 1的depth wise卷积，卷积核为<a href="https://www.codecogs.com/eqnedit.php?latex=w_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_t" title="w_t" /></a>。

TSM即为TAM的一个特例。比如，如果特征图<a href="https://www.codecogs.com/eqnedit.php?latex=y_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_t" title="y_t" /></a>的通道数为8时，那么当

<a href="https://www.codecogs.com/eqnedit.php?latex=r=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r=2" title="r=2" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=w_1=[1,1,0,0,0,0,0,0]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_1=[1,1,0,0,0,0,0,0]" title="w_1=[1,1,0,0,0,0,0,0]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=w_0=[0,0,0,0,1,1,1,1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_0=[0,0,0,0,1,1,1,1]" title="w_0=[0,0,0,0,1,1,1,1]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=w_{-1}=[0,0,1,1,0,0,0,0]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{-1}=[0,0,1,1,0,0,0,0]" title="w_{-1}=[0,0,1,1,0,0,0,0]" /></a>

时，TAM即退化为partial shift 1/2的TSM。

![TAM实现](https://raw.githubusercontent.com/raxxerwan/raxxerwan.github.io/master/images/2020-7-16-PR-bLNet-TAM/TAM.jpg)
TAM可以用r个1 * 1的depth wise卷积核来实现。



# 实验

## 训练

这篇文章用了一种很有意思的训练方式。文章先用8 * 2的输入帧数，0.01的lr对模型训练50个epoch；然后，再把输入改成16 * 2，0.01的initial lr（在Something-Something上是0.01，在Kinetics400上是0.005）训练25个epoch，这25个epoch期间，每10个epoch就将lr减少为之前的1/10。这相当于先用稀疏的输入帧数进行训练，然后再用稠密输入进行finetune。目的是为了减少训练时间。

## Inference

**Something-Something&MiT：** 中心crop，单clip。

**Kinetics400：** 10crop，10clip。

## 表现

![表现1](https://raw.githubusercontent.com/raxxerwan/raxxerwan.github.io/master/images/2020-7-16-PR-bLNet-TAM/performence.JPG)

![表现2](https://raw.githubusercontent.com/raxxerwan/raxxerwan.github.io/master/images/2020-7-16-PR-bLNet-TAM/performence2.JPG)

可以看到在对时序信息依赖比较大的Something-Something数据集上，bLVNet+TAM超过了TSM做到了当时的SotA。且在相同的backbone下，比起TSM，bLVNet+TAM几乎没有增加额外的参数量，且FLOPS甚至更低了。

但是在对时序信息依赖不大的Kinetics400上，bLVNet+TAM仍然没能超过永远滴神SlowFast。

## bLVNet+TAM对输入帧数-模型表现曲线的影响

![表现3](https://raw.githubusercontent.com/raxxerwan/raxxerwan.github.io/master/images/2020-7-16-PR-bLNet-TAM/performence3.JPG)

显然，在MiT数据集下，当使用valina-TSN时，**输入帧数-模型表现**曲线几乎是平的，8帧输入和32帧输入几乎没有表现上的差别。甚至在24帧时还有一定程度的下降。而bLVNet-TSN则使得这条曲线具备了一定上升趋势，即模型对稠密输入具有更好的适应性。