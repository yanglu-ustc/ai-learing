# recommender-system

<div align=center>
<img src="images/屏幕截图%202024-04-15%20223810.png" width=49%>
<img src="images/屏幕截图%202024-04-15%20224216.png" width=49%>
</div>

## cost function

<div align=center>
<img src="images/屏幕截图%202024-04-15%20224613.png" width=49%>
<img src="images/屏幕截图%202024-04-15%20224701.png" width=49%>
</div>

## Collaborative filtering algorithm:协同过滤算法

* 上面的是知道特征训练偏好，而我们现在考虑知道偏好训练特征(知道$\vec{w}$，训练$\vec{x}$)

<div align=center>
<img src="images/屏幕截图%202024-04-16%20162518.png" width=90%>
</div>

* 将$\vec{x}$同样看作一个参数，会得到一个由两个cost function相加得到的cost function

<div align=center>
<img src="images/屏幕截图%202024-04-16%20163759.png" width=45%>
<img src="images/屏幕截图%202024-04-16%20163834.png" width=45%>
</div>

## Binary labels: favs, likes and clicks

* 问题示例
<div align=center>
<img src="images/屏幕截图%202024-04-16%20164813.png" width=90%>
</div>

* 将回归函数进行替换为Sigmoid函数
<div align=center>
<img src="images/屏幕截图%202024-04-16%20165146.png" height="220">
<img src="images/屏幕截图%202024-04-16%20165338.png" height="220">
</div>