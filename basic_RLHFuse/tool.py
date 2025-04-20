
from global_config import DC_DELAY_TIME

# 判断一个调度是否有效（满足约束条件）, 调度矩阵只有行条件，因此看每一行即可
# schedule_matrix = [[1,2,3],[4,5,6]] # 代表第1行的任务是1,2,3，第2行的任务是4,5,6
def check_schedule_is_valid(schedule_matrix,schedule):
    for task in schedule.task_list:
        for constraint_id in task.pre_task:
            # 找到intra-stage约束
            if schedule.tasks[constraint_id].pp_stage_id == task.pp_stage_id:
                 if schedule_matrix[task.pp_stage_id-1].index(constraint_id) >= schedule_matrix[task.pp_stage_id-1].index(task.task_id):
                     return False
    return True
    
            
# 计算给定调度S的总时间
# 输入：调度矩阵schedule_matrix，调度schdule，这两者并非永远对应
'''
    思考时间计算算法，schedule_matrix给定了每一行的任务，以及任务的intra_stage约束关系,意味着我们只需要从前往后就好了
    同时我们还需要每个任务的inter_stage约束关系，这样才能顶下所有任务的开始和结束时间
    根据调度顺序，我们每次考虑每一行的最前面的未调度的任务，将可调度任务加入调度并修改时间，调整调度约束关系，进行下一轮迭代
    重复上述操作直到调度矩阵内所有任务都完成。
    这里我们需要兼容跨DC情景，即多了DC_delay_forward和DC_delay_backward两个任务列表（它们是特殊的任务，有着特殊的约束）
    不同与贪心初始化的调度，每个任务被调度后，都要更新可调度任务列表
    不论如何修改调度算法，inter_stage约束关系始终不变，变的是不同情况下intra_stage约束（狭义上说，就是它的左侧事件）
    我们没调度之前的intra_stage约束是宽泛的约束
'''
# 输出：总时间 ,当前调度下的各个任务的时间
# task_time_dict = {task_id: task_end_time} # 记录每个任务的结束时间（跨DC任务把跨DC通信结束作为结束时间）
def energy_compute(schedule_matrix, schedule):
    # 构建临时字典记录当前调度的时间规划情况
    task_time_dict = construct_task_time_dict(schedule.task_counts)
    size_list = [len(schedule_matrix[i]) for i in range(schedule.pp_stages)] # 记录每一行迭代终点
    iter_list = [0 for i in range(schedule.pp_stages)] # 记录每一行的迭代位置
    total_end_time = 0
    
    for k in range(schedule.pp_stages):
        while iter_list[k] < size_list[k]:
            task_id = schedule_matrix[k][iter_list[k]]
            task = schedule.tasks[task_id]
            if iter_list[k] == 0:
                # 我们不去改task.end_time，因为每次的计算都是临时的结果，并非最终
                # task.end_time = compute_task_end_time(task_time_dict, task, schedule, iter_list[k]-1)
                task_time_dict[task_id] = compute_task_end_time(task_time_dict,task, schedule)
            else:
                left_task = schedule.tasks[schedule_matrix[k][iter_list[k]-1]]
                left_task_end_time = task_time_dict[left_task.task_id]
                task_time_dict[task_id] = compute_task_end_time(task_time_dict, task, schedule, left_task_end_time)
            # 每次更新最终完成时间
            if task_time_dict[task_id] > total_end_time:
                total_end_time = task_time_dict[task_id]

            iter_list[k] += 1
    # print(f"total_end_time: {total_end_time}")
    return total_end_time, task_time_dict


# 计算单个任务的结束时间
def compute_task_end_time(task_time_dict, task, schedule, left_task_end_time=0):
    task_id = task.task_id
    if task_time_dict[task_id] !=0:
        return task_time_dict[task_id]
    else:
        # 计算原始约束前驱任务的最迟完成时间以及当前“调度”下的最迟完成时间（左侧的第1个事件约束），递归式求解
        pre_task_end_time_max = max([compute_task_end_time(task_time_dict, schedule.tasks[p], schedule) for p in task.pre_task], default = 0)
        pre_task_end_time_max = max(pre_task_end_time_max, left_task_end_time)
        # 考虑跨DC通信延迟
        cross_DC_delay = 0 if task.is_cross_DC == False else DC_DELAY_TIME
        task_time_dict[task_id] = pre_task_end_time_max + task.cal_delay + cross_DC_delay
        return task_time_dict[task_id]


# 构造一个字典，键为任务id，值为改任务的结束时间，用于每次新调度的计算
def construct_task_time_dict(task_counts):
    return {i:0 for i in range(1,task_counts+1)}

# 判断一个任务是否为跨DC任务的初始化,给定当前任务的PP_Stage范围以及DC分界线上界
# stage_start, stage_end都是闭区间
def set_is_cross_DC_task(task, stage_start, stage_end, pp_DC_divided_up, is_reversed=False):
    my_pp_stage = task.pp_stage_id
    my_task_type = task.task_type
    up_edge = pp_DC_divided_up
    down_edge = up_edge + 1
    if my_task_type == 'forward':
        if is_reversed==False and stage_start<=up_edge and down_edge<=stage_end:# 0->1->2->3
            if my_pp_stage == up_edge:
                task.is_cross_DC = True
        
        elif is_reversed==True and stage_start>=down_edge and up_edge>=stage_end: # 3->2->1->0
            if my_pp_stage == down_edge:
                task.is_cross_DC = True
            
    elif my_task_type == 'backward':
        if is_reversed==False and stage_start<=up_edge and down_edge<=stage_end: # 3->2->1->0
            if my_pp_stage == down_edge:
                task.is_cross_DC = True
        elif is_reversed==True and stage_start>=down_edge and up_edge>=stage_end: # 0->1->2->3
            if my_pp_stage == up_edge:
                task.is_cross_DC = True
    else:
        raise ValueError("Invalid task type")


# 标准化调度格式，便于绘制可视化图像，输出为字典
def data_standardization_for_visual(schedule):
    data_dict = {}
    task_time_dict = schedule.schdule_time_map # 获取计算好的所有任务的结束时间映射
    for task in schedule.task_list:
        task_id = task.task_id
        start_time = task_time_dict[task_id] - task.cal_delay - (DC_DELAY_TIME if task.is_cross_DC else 0)
        data_dict[task_id] = {
            "start_time": start_time,
            "end_time": task_time_dict[task_id],
            "cal_time": task.cal_delay,
            "is_cross_DC": task.is_cross_DC,
            "task_type": task.task_type,
            "pp_stage": task.pp_stage_id,
            "micro_batch_id": task.micro_batch_id,
            "model_id": task.model_id
        }
    return data_dict
    