import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from global_config import color_dict
import json

pp_row_height = 1
comm_row_height = 0.5
space_row_height = 0.3

def create_figure(data_dict, pp_stages, total_time):
    fig = go.Figure()
    all_shapes = []  # 临时存储所有 shape
    for data in data_dict.values():
        # 处理PP调度图
        color = color_dict[data['model_id']][data['task_type']]
        y0 = data['pp_stage'] + 1 - pp_row_height / 2
        y1 = data['pp_stage'] + 1 + pp_row_height / 2
        shape = dict(
            type="rect",
            x0=data['start_time'],
            x1=data['start_time'] + data['cal_time'],
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(color="black", width=1),
            layer="below",
        )
        # 添加文字（居中）
        fig.add_annotation(
            x=data['start_time'] + data['cal_time'] / 2,
            y=(y0 + y1) / 2,
            text=str(data['micro_batch_id']),
            showarrow=False,
            font=dict(color="white", size=10),
            xanchor="center",
            yanchor="middle",
        )
        all_shapes.append(shape)
        if data['is_cross_DC']:
            color = color_dict[data['model_id']][data['task_type']]
            extra_height = 4 if data['task_type'] == 'backward' else 2
            y0 = pp_stages + extra_height - comm_row_height / 2
            y1 = pp_stages + extra_height + comm_row_height / 2
            shape = dict(
                type="rect",
                x0=data['start_time']+ data['cal_time'],
                x1=data['end_time'],
                y0=y0,
                y1=y1,
                fillcolor=color,
                line=dict(color="black", width=1),
                layer="below",
            )
            # 添加文字（居中）
            fig.add_annotation(
                x=data['start_time'] + data['cal_time'] + (data['end_time'] - data['start_time']-data['cal_time']) / 2,
                y=(y0 + y1) / 2,
                text=str(data['micro_batch_id']),
                showarrow=False,
                font=dict(color="white", size=10),
                xanchor="center",
                yanchor="middle",
            )
            all_shapes.append(shape)
    # 设置坐标轴与背景
    vline_shapes = []
    for x in range(0, int(total_time)+3):  # 每个时间单位中心对齐
        vline_shapes.append(dict(
            type="line",
            x0=x , x1=x,   # 放在每个时间段中心
            y0=0, y1=pp_stages + 5,   # 整个图的高度
            line=dict(color="lightgrey", width=0.2),
            layer="below"
        ))
    for y in range(0, pp_stages + 5):
        vline_shapes.append(dict(
            type="line",
            x0=0, x1=int(total_time)+3,
            y0=y+pp_row_height/2, y1=y+pp_row_height/2,  # 每个阶段的中心
            line=dict(color="lightgrey", width=0.2),
            layer="below"
        ))


    fig.update_layout(
        yaxis=dict(autorange='reversed')
    )

    fig.update_layout(shapes=(all_shapes + vline_shapes))
    fig.write_html("pp_schedule_interactive.html", include_plotlyjs='cdn')
    fig.show()
    return fig

def generate_sample_data():
    data = {}
    task_id = 0
    pp_stages = 4
    models = [1, 2]
    
    for model in models:
        for mb_id in range(3):  # 3个micro batch
            # 前向计算
            for stage in range(pp_stages):
                task_id += 1
                start = stage * 2 + mb_id * 10
                data[task_id] = {
                    "start_time": start,
                    "end_time": start + 1.5,
                    "cal_time": 1.5,
                    "is_cross_DC": False,
                    "task_type": "forward",
                    "pp_stage": stage,
                    "micro_batch_id": mb_id,
                    "model_id": model
                }
            
            # 跨数据中心通信 (前向)
            task_id += 1
            data[task_id] = {
                "start_time": pp_stages * 2 + mb_id * 10 - 2,
                "end_time": pp_stages * 2 + mb_id * 10 + 1,
                "cal_time": 1.5,
                "is_cross_DC": True,
                "task_type": "forward",
                "pp_stage": pp_stages-1,
                "micro_batch_id": mb_id,
                "model_id": model
            }
            
            # 后向计算
            for stage in reversed(range(pp_stages)):
                task_id += 1
                start = stage * 2 + mb_id * 10 + 15
                data[task_id] = {
                    "start_time": start,
                    "end_time": start + 2,
                    "cal_time": 2,
                    "is_cross_DC": False,
                    "task_type": "backward",
                    "pp_stage": stage,
                    "micro_batch_id": mb_id,
                    "model_id": model
                }
            
            # 跨数据中心通信 (后向)
            task_id += 1
            data[task_id] = {
                "start_time": pp_stages * 2 + mb_id * 10 + 13,
                "end_time": pp_stages * 2 + mb_id * 10 + 16,
                "cal_time": 2,
                "is_cross_DC": True,
                "task_type": "backward",
                "pp_stage": 0,
                "micro_batch_id": mb_id,
                "model_id": model
            }
    
    return data, pp_stages


if __name__ == "__main__":
    with open("data.json", "r") as f:
        data_dict = json.load(f)
        pp_stages = 4
        total_time = 0
        for task in data_dict.values():
            total_time = max(total_time, task['end_time'])
        create_figure(data_dict, pp_stages, total_time)