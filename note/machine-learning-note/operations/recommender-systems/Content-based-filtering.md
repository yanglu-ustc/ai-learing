# Content based filtering

* 基于内容过滤对比协同过滤
    * 协同过滤以对象为一个单位，考虑的特征取决于电影这样的目标值
        * 所有的特征几乎是未知的，我们不知道每一个特征代表了什么
    * 基于内容过滤是将两者的特征全部写出来，利用已有的特征进行拟合

<div align=center>
<img src="images/屏幕截图%202024-04-16%20181528.png" width=90%>
</div>

* 两者最终决定的特征向量的维度必须相等！！！
* 特征向量可以通过一些算法转换为维度相等的特征向量

<div align=center>
<img src="images/屏幕截图%202024-04-17%20203739.png" height="230">
<img src="images/屏幕截图%202024-04-17%20204444.png" height="230">
</div>

## deep learning for content-based filtering

<div align=center>
<img src="images/屏幕截图%202024-04-17%20205430.png" height="230">
<img src="images/屏幕截图%202024-04-17%20205632.png" height="230">
</div>

### 相关性的讨论和寻找

<div align=center>
<img src="images/屏幕截图%202024-04-17%20210118.png" width=90%>
</div>

## Recommending from a large catalogue

* Two steps: Retrieval & Ranking

<div align=center>
<img src="images/屏幕截图%202024-04-17%20210735.png" height="230">
<img src="images/屏幕截图%202024-04-17%20210956.png" height="230">
</div>

* 实际进行选取的时候，会适当增加考察数量，以求获得更好的数据结果
<div align=center>
<img src="images/屏幕截图%202024-04-17%20211408.png" width=90%>
</div>