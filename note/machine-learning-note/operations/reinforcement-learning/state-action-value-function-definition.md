# state-action-value-function-definition

* $Q(s,a):$在$s$这个状态进行$a$操作之后，在这个过程中所获得的最好的结果的$return$值
* 有些文献会使用$Q^{*}$来描述这个函数(Optimal Q function)
* 注意：在计算的时候，需要加上自身的reward值

<div align=center>
<img src="images/屏幕截图%202024-04-18%20165647.png" width=44%>
<img src="images/屏幕截图%202024-04-18%20170059.png" width=46%>
</div>

## Bellman Equation

$Q(s,a)=R(s)+\gamma \underset{a'}{\max}Q(s',a')$
$R(s):$即时奖励 —— immediate reward

<div align=center>
<img src="images/屏幕截图%202024-04-18%20173743.png" width=49%>
<img src="images/屏幕截图%202024-04-18%20174325.png" width=40%>
<img src="images/屏幕截图%202024-04-18%20174049.png" width=38%>
<img src="images/屏幕截图%202024-04-18%20174624.png" width=50%>
</div>
