

import random


def choose_optimizer(schedule_matrix,method=1):
    if method == 1:
        return random_neighborhood(schedule_matrix)
    else:
        pass



def random_neighborhood(schedule_matrix):
    while True:
        i = random.randint(1, )