# -*- coding: utf-8 -*-

import sys,random


import os
current_directory = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)
print('sys.path:', sys.path)
# 指定工作目录

# 指定工作目录

from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim
from compose.Graphormer.parameter import parse_args, IOStream, table_printer
import torch
from torch_geometric.loader import DataLoader
from compose.Graphormer.model import Graphormer
from compose.train.dataset import CustomGraphDataset
from torch.nn.parallel import DataParallel
def load_dataset(args):
    # 每个样本：Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])
    dataset = CustomGraphDataset(root='compose/train/Datasets/CustomGraphDataset49')

    # 1128个样本用于graph-level prediction 训练：902；测试：226

    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=6)

    return train_loader, test_loader, dataset.num_node_features, dataset.num_edge_features
def train(args, IO, train_loader, num_node_features, num_edge_features):
    # 使用GPU or CPU

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {num_gpus}")
    device_ids = list(range( torch.cuda.device_count()))
    model = Graphormer(args, num_node_features, num_edge_features)
    model = DataParallel(model, device_ids=device_ids).to(device_ids[0])
    model.cuda()

    if args.gpu_index < 0:
        IO.cprint('Using CPU')
    else:

        if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        IO.cprint('Using GPU: {}'.format(args.gpu_index))
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
        model.train()  # 训练模式
        train_loss = 0.0  # 一个epoch，所有样本损失总和

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train_Loader"):
            data = data.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0) # 剪裁可迭代参数的梯度范数，防止梯度爆炸
            optimizer.step()




            train_loss += loss.item()
        if train_loss/len(train_loader.dataset)<lowest_loss:
            lowest_loss=train_loss/len(train_loader.dataset)
            torch.save(model, 'outputs/%s/model.pth' % args.exp_name)
            IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

        IO.cprint('Epoch #{:03d}, Train_Loss: {:.6f}'.format(epoch, train_loss / len(train_loader.dataset)))


    torch.save(model, 'outputs/%s/model.pth' % args.exp_name)
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))


def test(args, IO, test_loader):
    """测试模型"""

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # 输出内容保存在之前的训练日志里
    IO.cprint('')
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    model = torch.load('outputs/%s/model.pth' % args.exp_name).to(device)
    model = model.eval()  # 创建一个新的评估模式的模型对象，不覆盖原模型

    ################
    ###   Test   ###
    ################
    test_loss = 0.0

    # 损失函数
    criterion = nn.L1Loss(reduction="sum")

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test_Loader"):
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
    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化

    train_loader, test_loader, num_node_features, num_edge_features = load_dataset(args)
    print("num_node_features:", num_node_features, "num_edge_features:", num_edge_features)
    train(args, IO, train_loader, num_node_features, num_edge_features)
    test(args, IO, test_loader)