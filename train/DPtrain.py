# -*- coding: utf-8 -*-

import sys,random


import os
current_directory = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

# 指定工作目录

# 指定工作目录

from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim
from Graphormer.parameter import parse_args, IOStream, table_printer
import torch
from torch_geometric.loader import DataLoader
from Graphormer.model import Graphormer
from dataset  import CustomGraphDataset



def load_dataset(args):
    dataset = CustomGraphDataset(root='./train/Datasets/CustomGraphDataset100')

    print(f"Dataset Size: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the dataset path and format.")

    print("Sample Data Example:", dataset[0])  # 打印第一个样本，确保数据格式正确

    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, dataset.num_node_features, dataset.num_edge_features
def train(args, IO, train_loader, num_node_features, num_edge_features):
    # 使用GPU or CPU

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {num_gpus}")

    model = Graphormer(args, num_node_features, num_edge_features)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = model.to(device)
        model = nn.DataParallel(model, device_ids = [0,1])  ## 确保模型在正确的设备上

    elif torch.cuda.device_count() == 1:
        model = model.to(device)




    torch.cuda.manual_seed(args.seed)  # 设置PyTorch GPU随机种子

    # 加载模型及参数量统计

    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    IO.cprint('Using AdamW')

    # 损失函数
    criterion = nn.L1Loss(reduction="sum")
    lowest_loss=float('inf')
    epochs = trange(args.epochs, leave=True, desc="Epochs")

    for epoch in epochs:
        #################
        ###   Train   ###
        #################
         # 训练模式
        train_loss = 0.0  # 一个epoch，所有样本损失总和


        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train_Loader"):
            data = data.to(device)  # Move data to the correct device
            optimizer.zero_grad()
            outputs = model(data)  # No need to call .to(device) on outputs
            print(f"outputs.shape: {outputs.shape}，data.y.shape: {data.y.shape}")
            loss = criterion(outputs, data.y)  # No need to call .to(device) on data.y
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        if train_loss/len(train_loader.dataset)<lowest_loss:
            lowest_loss=train_loss/len(train_loader.dataset)
            torch.save(model, 'outputs/%s/best.pth' % args.exp_name)
            IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/best.pth *********' % args.exp_name))

        IO.cprint('Epoch #{:03d}, Train_Loss: {:.6f}'.format(epoch, train_loss / len(train_loader.dataset)))


    torch.save(model, 'outputs/%s/model.pth' % args.exp_name)
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))


def test(args, IO, test_loader):
    """测试模型"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 输出内容保存在之前的训练日志里
    IO.cprint('')
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/best.pth *********' % args.exp_name))

    model = torch.load('outputs/%s/model.pth' % args.exp_name).to(device)
    model = model.eval()  # 创建一个新的评估模式的模型对象，不覆盖原模型

    ################
    ###   Test   ###
    ################
    test_loss = 0.0

    # 损失函数
    criterion = nn.L1Loss(reduction="sum")

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test_Loader"):
        print(f"x.shape: {data.x.shape}")
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, data.y)

        test_loss += loss.item()

    IO.cprint('TEST :: Test_Loss: {:.6f}'.format(test_loss / len(test_loader.dataset)))


def exp_init():
    """实验初始化"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)

    # 跟踪执行脚本，windows下使用copy命令，且使用双引号
    # os.system(f"copy main.py outputs\\{args.exp_name}\\main.py.backup")
    # os.system(f"copy data.py outputs\\{args.exp_name}\\data.py.backup")
    # os.system(f"copy layer.py outputs\\{args.exp_name}\\layer.py.backup")
    # os.system(f"copy model.py outputs\\{args.exp_name}\\model.py.backup")
    # os.system(f"copy parameter.py outputs\\{args.exp_name}\\parameter.py.backup")
    os.system('cp main.py outputs' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp layer.py outputs' + '/' + args.exp_name + '/' + 'layer.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp parameter.py outputs' + '/' + args.exp_name + '/' + 'parameter.py.backup')
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    exp_init()

    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化


    train_loader, test_loader, num_node_features, num_edge_features = load_dataset(args)

    print(f"num_node_features: {num_node_features}, num_edge_features: {num_edge_features}")
    if num_node_features is None or num_edge_features is None:
        raise ValueError("num_node_features or num_edge_features is None. Check dataset implementation.")
    print("num_node_features:", num_node_features, "num_edge_features:", num_edge_features)
    train(args, IO, train_loader, num_node_features, num_edge_features)
    test(args, IO, test_loader)