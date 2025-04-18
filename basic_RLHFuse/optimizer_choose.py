
from basic_type import Schedule
import random
from tool import check_schedule_is_valid

def choose_optimizer(schedule_matrix,method=1):
    if method == 1:
        return random_neighborhood(schedule_matrix)
    else:
        pass

def swap_schedule(schedule_matrix, i, j):
    schedule_matrix[i][j], schedule_matrix[i][j+1] = schedule_matrix[i][j+1], schedule_matrix[i][j]

def random_neighborhood(schedule: Schedule):
    N = schedule.pp_stages + 1
    M = sum(b for a,b in schedule.mirco_batches) * 2
    schedule_matrix = schedule.scheduled_task.copy() # 复制一份随便去改
    while True:
        i = random.randint(1, N)
        j = random.randint(1, M)
        # swap(S_ij, S_i{j+1})
        swap_schedule(schedule_matrix, i, j)
        # checkValid(new_schedule), if valid, break and return new_schedule
        if check_schedule_is_valid(schedule_matrix, schedule):
            return schedule_matrix 
        else: # 如果不合法，就换回来
            swap_schedule(schedule_matrix, i, j)