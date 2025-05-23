
from global_config import DC_DELAY_TIME

# 判断一个调度是否有效（满足约束条件）, 调度矩阵只有行条件，因此看每一行即可
# schedule_matrix = [[1,2,3],[4,5,6]] # 代表第1行的任务是1,2,3，第2行的任务是4,5,6
# 约束来源于两个大的方面，一个是初始化情况下的任务约束关系，一个是给定行调度顺序的约束关系
# 因此需要判断是否成环？进行拓扑排序，如果拓扑排序结果有效，我们用拓扑排序的结果去按顺序计算时间

def check_schedule_is_valid(schedule_matrix,schedule):
    dict = {} # 记录每个任务的前驱任务
    for i in range(len(schedule_matrix)):
        for j in range(len(schedule_matrix[i])):
            if schedule_matrix[i][j] not in dict:
                dict[schedule_matrix[i][j]] = set()
            if j > 0:
                dict[schedule_matrix[i][j]].add(schedule_matrix[i][j-1])
            dict[schedule_matrix[i][j]].update(schedule.tasks[schedule_matrix[i][j]].pre_task)
    topo_sort_list = [] # 记录拓扑排序的结果
    while len(dict) >0:
        is_ring = True # 是否有环
        for key in list(dict.keys()):
            if len(dict[key]) == 0:
                del dict[key]
                # 只要每轮删除了一个元素，就说明暂时无环
                is_ring = False
                topo_sort_list.append(key)
                for k in dict.keys():
                    if key in dict[k]:
                        dict[k].remove(key)
                break
        if is_ring == True:
            # print("find a ring in the schedule topo!")
            return False
    # print("find a valid schedule topo!")
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
    total_end_time = 0
    iter_list = [0 for i in range(len(schedule_matrix))] # 记录每一行的当前迭代位置，强制必须从左到右按顺序调度
    finished_task_list = [] # 记录已经完成的任务
    while True:
        all_line_finish = 0
        for i in range(len(schedule_matrix)):
            if iter_list[i] == len(schedule_matrix[i]):
                all_line_finish += 1
                continue
            task_id = schedule_matrix[i][iter_list[i]] # 当前行的当前任务id
            task = schedule.tasks[task_id]
            is_feasible = True # 是否可调度

            for constraint_id in task.pre_task:
                if constraint_id not in finished_task_list:
                    # 如果当前任务的前驱任务没有完成，则跳过
                    is_feasible = False
                    break
            if is_feasible == True:
                task_left_end_time = task_time_dict[schedule_matrix[i][iter_list[i]-1]] if iter_list[i] > 0 else 0
                task_time_dict[task_id] = compute_task_end_time(task_time_dict, task, schedule, task_left_end_time)
                if task_time_dict[task_id] > total_end_time:
                    total_end_time = task_time_dict[task_id]

                finished_task_list.append(task_id)
                iter_list[i] += 1
        # 每一行都被调度完成
        if all_line_finish == len(schedule_matrix):
            break
            
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
    correct_batch_num = {
        a:c for a,b,c in schedule.mirco_batches
        # model_id : micro_batch_num
    }
    for task in schedule.task_list:
        task_id = task.task_id
        start_time = task_time_dict[task_id] - task.cal_delay - (DC_DELAY_TIME if task.is_cross_DC else 0)
        corrected_num = 0 if task.task_type=="forward" else correct_batch_num[task.model_id]
        data_dict[task_id] = {
            "start_time": start_time,
            "end_time": task_time_dict[task_id],
            "cal_time": task.cal_delay,
            "is_cross_DC": task.is_cross_DC,
            "task_type": task.task_type,
            "pp_stage": task.pp_stage_id,
            "micro_batch_id": task.micro_batch_id - corrected_num,
            "model_id": task.model_id
        }
    return data_dict
    