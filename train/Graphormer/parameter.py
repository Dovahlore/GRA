#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:57
"""
import os




# 更改工作目录到指定路径



import argparse
from texttable import Texttable


def parse_args():
    parser = argparse.ArgumentParser()  # 参数解析器对象

        # 其他已有参数...
    parser.add_argument('--seed', type=int, default=16, help='Random seed of the experiment')
    parser.add_argument('--exp_name', type=str, default='Exp', help='Name of the experiment')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Size of the training batch')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Size of the testing batch')
    parser.add_argument('--gpu_index', type=int, default=1, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate of AdamW')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of Graphormer layers')
    parser.add_argument('--node_dim', type=int, default=128, help='Hidden dimensions of node features')
    parser.add_argument('--edge_dim', type=int, default=128, help='Hidden dimensions of edge features')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--output_dim', type=int, default=1, help='Number of output node features')
    parser.add_argument('--max_in_degree', type=int, default=5, help='Max in degree of nodes')
    parser.add_argument('--max_out_degree', type=int, default=5, help='max out degree of nodes')
    parser.add_argument('--max_path_distance', type=int, default=5, help='Max pairwise distance between two nodes')

    args = parser.parse_args()  # 解析命令行参数

    return args


class IOStream():
    """训练日志文件"""
    def __init__(self, path):
        self.file = open(path, 'a') # 附加模式：用于在文件末尾添加内容，如果文件不存在则创建新文件

    def cprint(self, text):
        print(text)
        self.file.write(text + '\n')
        self.file.flush() # 确保将写入的内容刷新到文件中，以防止数据在缓冲中滞留

    def close(self):
        self.file.close()


def table_printer(args):
    """绘制参数表格"""
    args = vars(args) # 转成字典类型
    keys = sorted(args.keys()) # 按照字母顺序进行排序
    table = Texttable()
    table.set_cols_dtype(['t', 't']) # 列的类型都为文本(str)
    rows = [["Parameter", "Value"]] # 设置表头
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) # 下划线替换成空格，首字母大写
    table.add_rows(rows)
    return table.draw()