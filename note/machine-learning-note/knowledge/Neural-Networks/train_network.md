# Train a Neural Network in TensorFlow

> ```python
> import tensorflow as tf
> from tensorflow.keras import Sequential
> from tensorflow.keras.layers import Dense
> model = Sequential([
> Dense(units=25,activation='sigmoid')
> Dense(units=15,activation='sigmoid')
> Dense(units=1,activation='sigmoid')
> ])
> from tensorflow.keras.losses import BinaryCrossentropy
> # tell it what loss you want to use!!!!
> model.compile(loss=BinaryCrossentropy())
> # epochs:number of steps in gradient descent
> model.fit(X,Y,epochs=100)
> ```

## Training Details

$$
loss \ function:L(f(\vec{x}),y)=y^{(i)}\log f(\vec{x}^{(i)})+(1-y^{(i)})\log (1-f(\vec{x}^{(i)})
$$

+ in TensorFlow it also known as binary cross entropy(二元交叉熵)(BinaryCrossentropy)
+ MeanSequaredError(均方误差)

$$
cost \ function:J(W,B)=\frac{1}{m}\sum_{i=1}^{m}L(f(\vec{x}^{(i)}),y)
$$

## Alternatives to the sigmoid activation

+ ReLU:

  + most common choice in how neural networks(for hidden layers!!!!!)
  + reason:
    + easier
    + there is only one flat place but the Sigmoid has two, the gradient descents would be really slow
  + faster learning!!!!!

  $$
  a=g(z)=\max (0,z)
  $$
+ Linear activation function

  $$
  a=g(z)=z=\vec{w}\vec{x}+\vec{b}
  $$
+ choosing the activation functions

  + Binary classification:Sigmoid
  + Regression(y=+/-):Linear activation function
  + Regression(y=+):ReLU

> ```python
> import tensorflow as tf
> from tensorflow.keras import Sequential
> from tensorflow.keras.layers import Dense
> model = Sequential([
> # use relu activation function in hidden layers
> Dense(units=25,activation='relu')
> Dense(units=15,activation='relu')
> Dense(units=1,activation='sigmoid')
> ])
> from tensorflow.keras.losses import BinaryCrossentropy
> model.compile(loss=BinaryCrossentropy())
> model.fit(X,Y,epochs=100)
> ```

## why we need the activation functions?

+ if all $g(z)$ are linear ...
  + ... no different than linear regression
+ the same to Sigmoid, if all layers'$g(z)$ are linear, only the output layer is Sigmoid...
  + ... no different than Sigmoid
+ Don't
