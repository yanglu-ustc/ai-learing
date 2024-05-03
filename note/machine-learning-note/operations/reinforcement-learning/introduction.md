# reinforcement-learning —— 强化学习

## reward function:tell its score of the action

* positive reward : action well +1
* negative reward : action poorly -1000
* 状态：state

<div>
<img src="images/屏幕截图%202024-04-18%20160232.png" width=90%>
</div>

## return in reinforcement-learning

$$
Discount \ Factor:\gamma=0.9 \ or \ 0.99 \ or \ 0.999 \dots \quad
Reward:R_{i} \quad
Return=\sum_{i=1}^{n}\gamma^{i-1}R_{i}
$$

<div align=center>
<img src="images/屏幕截图%202024-04-18%20161738.png" width=50%>
<img src="images/屏幕截图%202024-04-18%20162116.png" width=48%>
</div>

## Making decisions:Policies in reinforcement-learning

$$
state \ \xrightarrow[\pi]{policy} \ action
$$

Policy:$\pi()$函数表示的是该状态下需要完成什么action

<div align=center>
<img src="images/屏幕截图%202024-04-18%20163119.png" width=41%>
<img src="images/屏幕截图%202024-04-18%20163127.png" width=50%>
</div>

## Markov Decision Process(MDP)

* 未来只取决于当前状态，而与之前的过程无关

<div align=center>
<img src="images/屏幕截图%202024-04-18%20163518.png" width=45%>
<img src="images/屏幕截图%202024-04-18%20164044.png" width=45%>
</div>
