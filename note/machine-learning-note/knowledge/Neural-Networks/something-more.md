## Convolutional Neural Network

+ 共享一部分视野，分别进行计算
+ [1,20],[11,30],[21,40]...[81,100]这就是一个典型的卷积层

## sympy库

> ```python
> import sympy
> J,w = sympy.symbols('J,w')
> J = w**2
> # 求导数的函数
> dJ_dw = sympy.diff(J,w)
> # 计算导数的具体值
> # w = 2带入导数的式子并求值
> dJ_dw.subs([(w,2)])
> ```

## Computation Graph
