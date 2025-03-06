
# -*- coding: utf-8 -*-
from torch_geometric.datasets import Planetoid
import os, torch, random
from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim
from compose.Graphormer.data import load_ESOL
from compose.Graphormer.parameter import parse_args, IOStream, table_printer
import torch

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle


from torch_geometric.loader import DataLoader


from compose.Graphormer.model import Graphormer
from compose.train.dataset import CustomGraphDataset
from sklearn.model_selection import train_test_split

def load_dataset(args):
    # 每个样本：Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDataset')

    # 1128个样本用于graph-level prediction 训练：902；测试：226
    train_dataset,test_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6)

    return  test_loader, dataset.num_node_features, dataset.num_edge_features

def test(args, IO, test_loader):
    """测试模型"""

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 输出内容保存在之前的训练日志里
    IO.cprint('')
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint(
        'The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    model = torch.load('outputs/%s/model.pth' % args.exp_name,weights_only=False).to(device)
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
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    test_loader, num_node_features, num_edge_features = load_dataset(args)
    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化
    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化
    test(args, IO, test_loader)