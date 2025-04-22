# PP_scheduling_algorithm

A basic  and  optimized cross-data center  pipeline parallel scheduling algorithm

## background and reference

[ RLHFuse: Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion](https://arxiv.org/abs/2409.13221)

reproduce algorithm

### Explanation

+ 4/20: 修改了约束条件，但是目前的计算结果仍然是错的
+ 4/21: 修改了有效性判断，使用拓扑排序对约束图进行检查，修改了调度时间的计算方法



### quick start

在`basic_type.py`中编写运行逻辑

`Schedule(pp_stages)`设置调度的总体`pp_stages`数量，即流水线并行度

`add_model_tasks()`：添加模型任务，参数说明如下：

+ `model_id`：独属于一个模型的标识（用于标定batch属于哪个模型）
+ `micro_batch_num`：batch的总数
+ `pp_start_layer`：模型pp阶段的第一个pp层，比如论文中model_1第一个pp层为1，model_2第一个pp层为4
+ `pp_total_num`：当前模型的PP度，需要整除调度`pp_stages`，论文中model_1的pp为4，model_2的pp为2
+ `is_reversed`：默认为`False`，表示正常的从pp_stage1到pp_stage_n，`True`代表反向从pp_stage_n到pp_stage_1

`greedy_init_matrix()`执行贪心算法初始化一个调度情况

`simulated_annealing()`执行退火算法直到温度达到阈值，不断调整调度并计算时间

`data_standardization_for_visual()`将调度结果处理为标准格式

`create_figure()`将标准格式的数据可视化



##### config

在`global_config.py`中设置全局的运行参数
