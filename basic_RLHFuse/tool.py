
from basic_type import Schedule


# 判断一个调度是否有效（满足约束条件）, 调度矩阵只有行条件，因此看每一行即可
def check_schedule_is_valid(schedule_matrix,schedule:Schedule):
    for task in schedule.task_list:
        for constraint in task.pre_task:
            # 找到intra-stage约束
            if schedule.tasks[constraint].pp_stage_id == task.pp_stage_id:
                 if schedule_matrix[task.pp_stage_id].index(constraint) >= schedule_matrix[task.pp_stage_id].index(task.task_id):
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
# 输出：总时间  
def energy_compute(schedule_matrix, schedule:Schedule):
    size_list = [len(schedule_matrix[i]) for i in range(schedule.pp_stages)] # 记录每一行迭代终点
    iter_list = [0 for i in range(schedule.pp_stages)] # 记录每一行的迭代位置
    while True:
        for k in range(schedule.pp_stages):
            while iter_list[k] < size_list[k]:
                task_id = schedule_matrix[k][iter_list[k]]
                task = schedule.tasks[task_id]
                if iter_list[k] == 0:
                    task.end_time = compute_task_end_time(task, schedule, iter_list[k]-1)
                else:
                    left_task = schedule.tasks[schedule_matrix[k][iter_list[k]-1]]
                    task.end_time = compute_task_end_time(task, schedule, iter_list[k]-1, left_task.end_time)
                
                iter_list[k] += 1



# 计算单个任务的结束时间
def compute_task_end_time(task, schedule:Schedule, left_task_end_time=0):
    if task.end_time !=0:
        return task.end_time
    else:
        # 计算原始约束前驱任务的最迟完成时间以及当前“调度”下的最迟完成时间（左侧的第1个事件约束），递归式求解
        pre_task_end_time_max = max([compute_task_end_time(schedule.tasks[p], schedule) for p in task.pre_task], default = 0)
        pre_task_end_time_max = max(pre_task_end_time_max, left_task_end_time)
        task.end_time = pre_task_end_time_max + task.cal_delay
        return task.end_time

            
    pass
