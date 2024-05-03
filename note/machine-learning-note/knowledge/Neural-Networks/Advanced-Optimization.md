# Advanced-Optimization

## "Adam" algorithm

+ Go faster - increase $\alpha$
+ Go slower - decrease $\alpha$

> $$
> w_{k}=w_{k}-\alpha_{k} \frac{\partial}{\partial w_{k}}J(\vec{w},b)\ for \ k=1,\dots,N \quad b=b-\alpha_{N+1} \frac{\partial}{\partial b}J(\vec{w},b)
> $$
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
>
> # learning_rate=1e-3 —— 设置初始学习率
> model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
>   loss=SparseCategoricalCrossentropy(from_logits=True))
>
> model.fit(X,Y,epochs=100)
>
> logits = model(X_)
> f_x = tf.nn.softmax(logits)
> ```
