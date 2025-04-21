# PP_scheduling_algorithm

A basic  and  optimized cross-data center  pipeline parallel scheduling algorithm

## background and reference

[ RLHFuse: Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion](https://arxiv.org/abs/2409.13221)

reproduce algorithm


### Explanation
+ 4/20: 修改了约束条件，但是目前的计算结果仍然是错的
+ 4/21: 修改了有效性判断，使用拓扑排序对约束图进行检查，修改了调度时间的计算方法

