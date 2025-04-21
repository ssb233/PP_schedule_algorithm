import numpy as np
import global_config as config
from tool import energy_compute, set_is_cross_DC_task, data_standardization_for_visual
from optimizer_choose import random_neighborhood
import random
from my_visual import create_figure
import json


class Task:
    def __init__(self,task_type, model_id=1, micro_batch_id=1, pp_stage_id=1, task_id=0):
        self.model_id = model_id
        self.micro_batch_id = micro_batch_id # S_j,前向任务是对应的，后向任务需要减去micro_batches
        self.task_id = task_id # 任务id,唯一标定每个任务
        self.pp_stage_id = pp_stage_id # S_i, 1~pp_stages
        self.task_type = task_type # forward / backward / inter-DC-delay
        self.cal_delay = config.parse_task_type_to_time(self.task_type) # 计算延迟
        self.start_time = 0
        self.end_time = 0
        self.pre_task = [] # 前驱任务,依赖关系,只有前驱任务都满足了才能调度, 内容为task_id
        self.left_task = None # 左邻居任务,依赖关系
        self.right_task = None # 右邻居任务,依赖关系
        self.is_feasible = False # 是否可调度,只有需求关系都满足了才能调度
        self.is_cross_DC = False # 是否跨数据中心,如果是跨数据中心，那么就需要考虑通信延迟
    def set_start_time(self, start_time):
        self.start_time = start_time
        
    def set_end_time(self, end_time):
        self.end_time = end_time
    
    # 更新任务的可调度性，只要自己的pre_task的所有事件都被调度了，那么自己的状态就切换到可调度
    def update_feasibility(self, scheduled_list):
        for constraint in self.pre_task:
            if constraint not in scheduled_list:
                return False
        self.is_feasible = True
        return True
        

class Schedule:
    def __init__(self, pp_stages):
        # 矩阵总大小为pp_stages * mirco_batches * 2，pp_stages行，mirco_batches*2列
        # 对于micro_batch_num = j的任务，j<=microbatches作为前向任务, micro_batch_num = j + micro_batches，作为对应的后向任务
        self.pp_stages = pp_stages # 阶段数
        self.task_counts = 0 # 任务计数器
        self.mirco_batches = [] #里面抛开论文中的M1+M2这样的，我们为单个模型单独设置micro_batches,[(model_1, pp_stages,micro_batch_num_1), (model_2, pp_stages,micro_batch_num_2)]
        # self.tasks = {} # 任务列表,key为(pp_stage_id, micro_batch_id, model_id)
        self.task_list = [] # 任务列表，其实本质没有调度，调度直接体现在各个任务的start_time和end_time,而不是调度矩阵
        self.scheduled_task = [] # 已经完成调度的任务（这里的调度指的是依赖关系以及前后任务都已设置）
        self.remained_task = [] # 剩余任务列表,未调度的
        self.optimizer_method = config.OPTIMIZER # 优化方法
        self.DC_delay_forward = [] # 前向延迟事件
        self.DC_delay_backward = [] # 后向延迟事件
        self.tasks = {} # task_id -> task映射字典
        self.schdule_time_map = {} # 调度时间映射,key为task_id,value为end_time，这个一直都要改，每次计算不一样
        self.total_end_time = 0 # 总的调度时间
        self.DC_debug = False # 是否开启DC调试模式，默认不开启,开启后会考虑跨DC通信延迟
        
        
    def task_counter(self):
        self.task_counts += 1
        return self.task_counts
    # 添加约束,constraint是一个元组，表示约束，比如(pp_stage_id, micro_batch_id) -> (pp_stage_id, micro_batch_id)
    # A -> B，表示A是B的前驱任务，B是A的后继任务
    # model_id用来区分来自不同的模型的任务
    def add_constraint(self, constraint, model_id):
        A_pp_stage_id, A_micro_batch_id, B_pp_stage_id, B_micro_batch_id = constraint
        A_task = None
        B_task = None
        for task in self.task_list:
            if task.model_id == model_id and task.pp_stage_id == B_pp_stage_id and task.micro_batch_id == B_micro_batch_id:
                B_task = task
            if task.model_id == model_id and task.pp_stage_id == A_pp_stage_id and task.micro_batch_id == A_micro_batch_id:
                A_task = task
        if  B_task is None:
            print(f"TaskB not found model_id:{model_id}, pp_stage_id:{B_pp_stage_id}, micro_batch_id:{B_micro_batch_id}")
            raise ValueError("TaskB not found bro!")
        if A_task is None:
            print(f"TaskA not found model_id:{model_id}, pp_stage_id:{A_pp_stage_id}, micro_batch_id:{A_micro_batch_id}")
            raise ValueError("TaskA not found bro!")
        B_task.pre_task.append(A_task.task_id) # 添加前驱约束

    
    # 设置事件约束，可能涉及两个模型，因此需要给定model_id,还有对应的micro_batch_num,is_reversed表示前向为从device_n到device_0
    # pp_start_device,pp_total_device，表示当前模型的pp范围
    ''' 思考:对于micro_batch_id,我们做讨论如下： 约定正常方向,特殊方向指的是bidirectional的情况从device_n 到 device_0
        1. 如果为前向任务,我们考虑其pp_stage_id,1<=pp_stage_id<=pp_stages,
        if pp_stage_id == 1:
            只有intra_stage_constraint: (pp_stage_id, micro_batch_id - 1) -> (pp_stage_id, micro_batch_id)
        elif pp_stage_id in range(2, pp_stages+1): 这里需要注意micro_batch_id为1的情况
            这里micro_batch_id都需要考虑其左侧以及左上侧的约束
            inter_stage_constraint: (pp_stage_id-1, micro_batch_id) -> (pp_stage_id, micro_batch_id)
            intra_stage_constraint: (pp_stage_id, micro_batch_id - 1) -> (pp_stage_id, micro_batch_id)
        
        2. 如果为后向任务,我们考虑pp_stage_id,
        if pp_stage_id == pp_stages:
            只有intra_stage_constraint: (pp_stage_id, micro_batch_id - 1) -> (pp_stage_id, micro_batch_id)
        elif pp_stage_id in range(1, pp_stages):
            这里需要注意micro_batch_id为1的情况
            inter_stage_constraint: (pp_stage_id+1, micro_batch_id) -> (pp_stage_id, micro_batch_id)
            intra_stage_constraint: (pp_stage_id, micro_batch_id - 1) -> (pp_stage_id, micro_batch_id)
    '''
    def set_all_constraint(self, model_id, micro_batch_num,pp_start_layer, pp_total_num, is_reversed=False):
        dp_group = self.pp_stages // pp_total_num # 阶段数，比如一个模型覆盖所有设备，那就只有一个dp组，否则有多个dp组,这里我们默认都能整除
        dp_group_batch_num = micro_batch_num // dp_group # 每个dp组的batch数
        step = 1 if is_reversed==False else -1
        flag = 1 if is_reversed==False else 0
        for i in range (dp_group):# device编号从1开始哈！，1-n
            for j in range(pp_start_layer + i * pp_total_num * step, pp_start_layer + (i + 1) * pp_total_num * step, step):
                # 前向任务
                for k in range(1+i*dp_group_batch_num, 1+(i+1)*dp_group_batch_num):
                    # 前向任务的首个行数，即第一行，需要考虑其左侧的约束
                    if (j-flag) % pp_total_num == 0:
                        if (k-1)%dp_group_batch_num !=0:# 第一个batch无约束
                            self.add_constraint((j, k-1, j, k), model_id)
                    else:
                        if (k-1)%dp_group_batch_num ==0: # 无左侧约束，有inter_stage约束
                            self.add_constraint((j-step, k, j, k), model_id)
                        else:
                            self.add_constraint((j, k-1, j, k), model_id)
                            self.add_constraint((j-step, k, j, k), model_id)

                # 后向任务
                for k in range(1+i*dp_group_batch_num, 1+(i+1)*dp_group_batch_num):
                    back_k = k + micro_batch_num # 后向任务的编号
                    # 第一行
                    if (j-1+flag) % pp_total_num == 0:
                        if (k-1)%dp_group_batch_num ==0:# 第一个batch只依赖于其前向
                            self.add_constraint((j, k, j, back_k), model_id)
                        else:# 其余的既依赖于其前向，也依赖于前一个后向
                            self.add_constraint((j, k, j, back_k), model_id)
                            self.add_constraint((j, back_k-1, j, back_k), model_id)
                    else:
                        if (k-1)%dp_group_batch_num ==0: # 无左侧约束，有inter_stage约束
                            self.add_constraint((j+step, back_k, j, back_k), model_id)
                        else:
                            self.add_constraint((j+step, back_k, j, back_k), model_id) # inter_stage约束
                            self.add_constraint((j, back_k-1, j, back_k), model_id) # intra_stage约束
                            
                
        print(f"Model {model_id} add constraint successfully!")

    def add_model_tasks(self, model_id, micro_batch_num, pp_start_layer, pp_total_num, is_reversed=False):
        dp_groups = self.pp_stages // pp_total_num
        dp_group_batch_num = micro_batch_num // dp_groups # 每个dp组的batch数
        step = 1 if is_reversed==False else -1
        for i in range(dp_groups):
            for j in range(pp_start_layer + i * pp_total_num * step, pp_start_layer + (i + 1) * pp_total_num * step, step):
                for k in range(1+i*dp_group_batch_num, 1+(i+1)*dp_group_batch_num):
                    # 同一行的前向和后向的编号差值为micro_batch_num
                    task_forward = Task('forward', model_id, k, j, self.task_counter())
                    task_backend = Task('backward', model_id, k+micro_batch_num, j, self.task_counter())
                    if self.DC_debug == True:
                        set_is_cross_DC_task(task_forward, pp_start_layer + i * pp_total_num * step, pp_start_layer + (i + 1) * pp_total_num * step-step, self.pp_stages//2, is_reversed)
                        set_is_cross_DC_task(task_backend, pp_start_layer + i * pp_total_num * step, pp_start_layer + (i + 1) * pp_total_num * step-step, self.pp_stages//2, is_reversed)
                    self.task_list.append(task_forward)
                    self.task_list.append(task_backend)
                    self.tasks[task_forward.task_id] = task_forward
                    self.tasks[task_backend.task_id] = task_backend

        self.mirco_batches.append((model_id,pp_total_num, micro_batch_num)) # 添加模型的micro_batch_num元组
        self.set_all_constraint(model_id, micro_batch_num, pp_start_layer, pp_total_num, is_reversed)
        
    # 采用贪心算法初始化调度矩阵
    def greedy_init_matrix(self):
        # 先初始化我们关键的两个列表
        self.scheduled_task = [[] for _ in range(self.pp_stages)] # 这应该是一个矩阵，调度完成得到矩阵，矩阵的位置关系表示行依赖关系, task_id矩阵
        self.remained_task = self.task_list.copy() # 复制一份任务列表
        finished_task_list = [] # 记录已经完成的任务的元组列表，一个任务完成就加入task_id
        while(len(self.remained_task)> 0):
            # 首先应该更新所有任务的is_feasible属性，将可调度任务单独拉出来，然后选取一个可调度任务进行调度
            '''思考
            1. 这一步是贪心调度，但是调度不等同于给出计算图，我们只需要针对每个pp_stage进行调度
            2. 这里需要考虑所有依赖关系（针对两个model的贪心初始化，单个model应该只需要考虑行约束？）
            3. 在调度阶段，我们抽象化所有的任务，将其时间都视作相同，这样在每个时间单位下，我们都去考虑所有pipe_stage的任务
            '''
            # 应该为每个pp_stage_id维护一个可调度任务列表，因为调度是在每一行中进行的
            # 每一轮可调度不应持续更新，只有在这一列的所有pp_stage都完成了才能调度下一列，因此先更新可调度任务
            # 也只在下一轮才更新finished_tuple_list
            for task in self.remained_task:
                task.update_feasibility(finished_task_list)
            for i in range(self.pp_stages):# 0~n-1, 实际id为1~n
                # 每一行我们维护一个可调度任务列表，然后从中选取一个可调度任务进行调度
                feasible_task_list = []
                for task in self.remained_task:
                    if task.pp_stage_id == i+1 and task.is_feasible==True:
                        feasible_task_list.append(task)
                bigger_model = 100 # 这里我们约定，model_id越小，代表模型越大
                target_task = None # 目标任务
                for task in feasible_task_list:
                    if task.model_id <= bigger_model:
                        bigger_model = task.model_id
                        target_task = task
                if target_task is None:
                    print("No feasible task found!")
                    continue
                else:
                    self.scheduled_task[i].append(target_task.task_id)
                    self.remained_task.remove(target_task)
                    finished_task_list.append(target_task.task_id)
        print("Greedy init matrix successfully!\n")
        with open('schedule_task.txt','w') as f:
            for i in range(len(self.scheduled_task)):
                f.write(f"Stage {i+1}:")
                for task_id in self.scheduled_task[i]:
                    model_id = self.tasks[task_id].model_id
                    batch_id = self.tasks[task_id].micro_batch_id
                    map = {a:c for a,b,c in self.mirco_batches}
                    if self.tasks[task_id].task_type == 'backward':
                        batch_id = batch_id - map[model_id]
                    f.write(f"({model_id}, {batch_id}) ")
                f.write("\n")
        # print([len(a) for a in self.scheduled_task])
    
    # 模拟退火 simulated annealing，最后得到的
    def simulated_annealing(self):
        print("Start simulated annealing!")
        energy_current, self.schdule_time_map = energy_compute(self.scheduled_task, self)
        Temperature = energy_current
        print(f"Initial energy: {energy_current}")
        while Temperature > config.Epsilon:
            optimized_matrix = random_neighborhood(self)
            optimized_energy, optimized_time_schedule = energy_compute(optimized_matrix, self)
            if optimized_energy < energy_current:
                energy_current = optimized_energy
                # self.scheduled_task = optimized_matrix
                self.schdule_time_map = optimized_time_schedule
            else:
                p = np.exp((energy_current - optimized_energy)/Temperature)
                rand = random.random()
                if rand <= p:
                    energy_current = optimized_energy
                    # self.scheduled_task = optimized_matrix
                    self.schdule_time_map = optimized_time_schedule
            Temperature = config.CoolingRate * Temperature
            print(f"Current temperature: {Temperature}")
        # 迭代计算完成,得到总调度时间，调度时间情况,schedule.schdule_time_map
        self.total_end_time = energy_current
        print("Simulated annealing successfully!")
        print(f"Total end time: {self.total_end_time}")
        print(f"Schedule time map: {self.schdule_time_map}")
        
                
         

if __name__ == "__main__":
    schedule = Schedule(4)
    schedule.add_model_tasks(1,4,1,4,False)
    schedule.add_model_tasks(2,4,4,2,True)
    schedule.greedy_init_matrix()
    schedule.simulated_annealing()
    data_dict = data_standardization_for_visual(schedule)
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f)  # 写入文件（覆盖原有内容）
    fig = create_figure(data_dict, schedule.pp_stages, schedule.total_end_time)
    
    