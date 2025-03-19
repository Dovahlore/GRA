
# -*- coding: utf-8 -*-

import os, torch, random


from tqdm import tqdm
import torch.nn as nn
from Graphormer.parameter import parse_args, IOStream, table_printer
import torch
from torch_geometric.loader import DataLoader
from dataset import CustomGraphDataset
from sklearn.model_selection import train_test_split
from Graphormer.model import Graphormer
def load_dataset(args):
    # 每个样本：Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDataset100')

    # 1128个样本用于graph-level prediction 训练：902；测试：226
    train_dataset,test_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    print("Test_Dataset_Size:", len(dataset))

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
        'The current best model is saved in: {}'.format('******** outputs/%s/best.pth *********' % args.exp_name))
    checkpoint = torch.load("outputs/%s/model.pth" % args.exp_name, map_location="cpu")

    # 兼容 DDP 训练的模型
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 处理 "module." 前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model= Graphormer(args, num_node_features=num_node_features, num_edge_features=num_edge_features)
    model.load_state_dict(new_state_dict)

    model.to(device)  # 移动到 GPU
    model.eval()  # 进入评估模式
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