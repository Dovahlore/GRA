
# -*- coding: utf-8 -*-

import os, torch, random
import time


from Graphormer.parameter import parse_args, IOStream, table_printer
import torch
from torch_geometric.loader import DataLoader
from dataset import CustomGraphDataset
from sklearn.model_selection import train_test_split

def load_dataset(args):
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDataset400')


    train_dataset,test_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6)

    return  test_loader, dataset.num_node_features, dataset.num_edge_features


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    test_loader, num_node_features, num_edge_features = load_dataset(args)
    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化
    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 输出内容保存在之前的训练日志里
    IO.cprint('')
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint(
        'The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    model = torch.load('outputs/%s/best.pth' % args.exp_name, weights_only=False).to(device)
    model = model.eval()  # 创建一个新的评估模式的模型对象，不覆盖原模型
    test_loader, num_node_features, num_edge_features = load_dataset(args)
    for data in test_loader.dataset:
        data = data.to(device)
        with torch.no_grad():
            start_time = time.time()
            pred = model(data)
            end_time = time.time()
            print('Time: %.3f' % (end_time - start_time))
