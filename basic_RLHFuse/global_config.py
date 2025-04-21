# config

# 前向计算，后向计算，跨DC通信时间
FORWARD_TIME = 1
BACKWARD_TIME = 2
DC_DELAY_TIME = 4

# 优化器配置
# 1：random_neighborhood
OPTIMIZER = 1

# 模拟退火参数
Epsilon = 0.0000001
CoolingRate = 0.999


# 绘图颜色
color_dict = {
    1: {
        'forward': '#1f77b4',  # 蓝色
        'backward': '#2ca02c'  # 绿色
    },
    2: {
        'forward': '#ff7f0e',  # 橙色
        'backward': '#d62728'  # 红色
    }
}



# tools函数
def parse_task_type_to_time(task_type):
    if task_type == 'forward':
        return FORWARD_TIME
    elif task_type == 'backward':
        return BACKWARD_TIME
    elif task_type == 'inter-DC-delay':
        return DC_DELAY_TIME