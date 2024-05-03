# implementation：具体实现

## normalization

* 将数据进行均值归一化处理，将暂时没有数据的对象使用均值进行覆盖

<div align=center>
<img src="images/屏幕截图%202024-04-16%20172536.png" width=90%>
</div>

## TensorFlow implementation

```python
# 初始化，并且告诉程序这是一个变量
w = tf.Varible(3.0)
x = 1.0
y = 1.0  # target value
alpha = 0.01

iterations = 30
for iter in range(iterations):
    # Use TensorFlow's Gradient tape to record the steps
    # used to compute the cost J, to enable auto differentiation
    with tf.GradientTape() as tape:
        fwb = w * x
        costJ = (fwb - y) ** 2

    # Use the gradient tape tp calculate the gradients
    # of the cost with respect to the parameter w
    [dJdw] = tape.gradient(costJ, [w])

    # Run one step of gradient descent by updating
    # the value of w to reduce the cost
    w.assign_add(-alpha * dJdw)
```

<div align=center>
<img src="images/屏幕截图%202024-04-16%20174710.png" width=90%>
</div>

* 协同过滤的实现

```python
"""
说明：
    X、W、b：参数和特征
    Ynorm：表示的是评级均值归一化
    R：指定哪些值有等级
    num_users,num_movies：表示的是数量
    lambda：表示的是正则化参数
    zip()：这个函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
"""
```

<div align=center>
<img src="images/屏幕截图%202024-04-16%20175736.png" width=90%>
</div>

## Finding related items

* 特征向量的差距是说明相关性的

<div align=center>
<img src="images/屏幕截图%202024-04-16%20180722.png" width=90%>
</div>

## Limitations of Collaborative Filtering

<div align=center>
<img src="images/屏幕截图%202024-04-16%20180821.png" width=90%>
</div>
