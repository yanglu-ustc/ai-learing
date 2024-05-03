# multiclass

## Softmax

+ for example:

> $$
> z_{1}=\vec{w_{1}}\cdot\vec{x}+b_{1}
> \quad z_{2}=\vec{w_{2}}\cdot\vec{x}+b_{2}
> \quad z_{3}=\vec{w_{3}}\cdot\vec{x}+b_{3}
> \quad z_{4}=\vec{w_{4}}\cdot\vec{x}+b_{4}
> $$
>
> $$
> a_{1}=\frac{e^{z_{1}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}+e^{z_{4}}}
> \quad a_{2}=\frac{e^{z_{2}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}+e^{z_{4}}}
> \quad a_{3}=\frac{e^{z_{3}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}+e^{z_{4}}}
> \quad a_{4}=\frac{e^{z_{4}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}+e^{z_{4}}}
> $$
>
> $$
> a_{k}=\frac{e^{z_{k}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}+e^{z_{4}}}=P(y=k|\vec{x})
> $$

+ N possible outputs:

> $$
> z_{j}=\vec{w_{j}}\cdot\vec{x}+b_{j} \quad j=1,2,\dots,N
> $$
>
> $$
> a_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{N}e^{z_{k}}}=P(y=j|\vec{x})
> $$
>
> + note:$a_{1}+a_{2}+\dots+a_{N}=1$

### loss function

$$
loss(a_{1},a_{2},\dots,a_{N},y)=\left\{
\begin{aligned}
&-\log a_{1} \quad if \ y=1\\
&-\log a_{2} \quad if \ y=2\\
& \dots \\
&-\log a_{N} \quad if \ y=N\\
\end{aligned}
\right.
$$

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
Dense(units=25,activation='relu')
Dense(units=15,activation='relu')
Dense(units=1,activation='softmax')
])
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy())
model.fit(X,Y,epochs=100)
```

+ 减少误差.....
+ 计算机在计算的过程中，增加了中间步骤之后，会让差值得以增大，为了减少误差，我们采用下面这种方式进行编写代码

> + 逻辑回归运算(Sigmoid)
>
> ```python
> import tensorflow as tf
> from tensorflow.keras import Sequential
> from tensorflow.keras.layers import Dense
> model = Sequential([
> # use relu activation function in hidden layers
> Dense(units=25,activation='relu')
> Dense(units=15,activation='relu')
> # 采用线性回归
> Dense(units=1,activation='linear')
> ])
> from tensorflow.keras.losses import BinaryCrossentropy
> # model.compile(loss=BinaryCrossentropy())
> # 将激活函数带入线性回归方程
> model.compile(loss=BinaryCrossentropy(from_logits=True))
> model.fit(X,Y,epochs=100)
>
> logits = model(X_)
> f_x = tf.nn.sigmoid(logits)
> ```
> + Softmax
>
> ```python
> import tensorflow as tf
> from tensorflow.keras import Sequential
> from tensorflow.keras.layers import Dense
> model = Sequential([
> Dense(units=25,activation='relu')
> Dense(units=15,activation='relu')
> Dense(units=1,activation='linear')
> ])
> from tensorflow.keras.losses import SparseCategoricalCrossentropy
> model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
> model.fit(X,Y,epochs=100)
>
> logits = model(X_)
> f_x = tf.nn.softmax(logits)
> ```

# multiple classes(多标签分类，将多个类进行同时分析)
+ train one neural network with three outputs
+ 例子：分析一个图片中是否有车、人、马？这三个标签对照片进行分类，最后输出结果大概是一个含三个数据的向量值