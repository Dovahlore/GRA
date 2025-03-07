import sys, random
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from Graphormer.parameter import parse_args, IOStream, table_printer
from Graphormer.model import Graphormer
from dataset import CustomGraphDataset
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import DistributedSampler


def load_dataset(args):
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDataset49')

    print(f"Dataset Size: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the dataset path and format.")

    print("Sample Data Example:", dataset[0])  # 打印第一个样本，确保数据格式正确

    # 创建DistributedSampler，确保每个进程加载不同的训练数据
    train_sampler = DistributedSampler(dataset, shuffle=True)
    test_sampler = DistributedSampler(dataset, shuffle=False)

    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=test_sampler, num_workers=0)

    return train_loader, test_loader, dataset.num_node_features, dataset.num_edge_features


def train(args, IO, train_loader, num_node_features, num_edge_features, rank, world_size):
    # 使用GPU or CPU
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")

    model = Graphormer(args, num_node_features, num_edge_features)
    model.train()

    model.to(device)
    model = DDP(model, device_ids=[device_id])  # 使用 DDP

    if rank == 0:
        IO.cprint(str(model))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        IO.cprint('Model Parameter: {}'.format(total_params))

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if rank == 0:
        IO.cprint('Using AdamW')

    # 损失函数
    criterion = nn.L1Loss(reduction="sum")
    lowest_loss = float('inf')
    epochs = trange(args.epochs, leave=True, desc="Epochs") if rank == 0 else range(args.epochs)

    for epoch in epochs:
        #################
        ###   Train   ###
        #################
        train_loss = 0.0  # 一个epoch，所有样本损失总和

        train_loader.sampler.set_epoch(epoch)  # 设置每个epoch的随机种子，确保不同epoch的数据顺序不同

        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)  # 前向传播
            loss = criterion(outputs, data.y.to(device))  # 计算损失，确保标签也在 GPU 上
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        if rank == 0:
            if avg_train_loss < lowest_loss:
                lowest_loss = avg_train_loss
                torch.save(model.state_dict(), 'outputs/%s/best.pth' % args.exp_name)
                IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/best.pth *********' % args.exp_name))

            IO.cprint('Epoch #{:03d}, Train_Loss: {:.6f}'.format(epoch, avg_train_loss))

    # 保存最终模型
    if rank == 0:
        torch.save(model.state_dict(), 'outputs/%s/model.pth' % args.exp_name)
        IO.cprint('The final model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))


def test(args, IO, test_loader, num_node_features, num_edge_features, rank):
    """测试模型"""
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")

    if rank == 0:
        IO.cprint('********** TEST START **********')
        IO.cprint('Reload Best Model')

    model = Graphormer(args, num_node_features, num_edge_features).to(device)
    # 确保所有进程加载相同的模型
    map_location = {'cuda:%d' % 0: 'cuda:%d' % device_id}  # 根据当前进程调整设备映射
    model.load_state_dict(torch.load('outputs/%s/best.pth' % args.exp_name, map_location=map_location))
    model = DDP(model, device_ids=[device_id])  # 测试时也需要用DDP包装以处理分布式数据
    model.eval()

    ################
    ###   Test   ###
    ################
    test_loss = 0.0
    criterion = nn.L1Loss(reduction="sum")

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, data.y)
        test_loss += loss.item()

    # 汇总所有进程的损失
    total_test_loss = torch.tensor(test_loss, device=device)
    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
    avg_test_loss = total_test_loss.item() / len(test_loader.dataset)  # 注意：数据集总长度是所有进程的总和

    if rank == 0:
        IO.cprint('TEST :: Test_Loss: {:.6f}'.format(avg_test_loss))


def exp_init(args, rank):
    """实验初始化（仅在rank 0执行）"""
    if rank != 0:
        return
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)

    os.system('cp main.py outputs' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp layer.py outputs' + '/' + args.exp_name + '/' + 'layer.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp parameter.py outputs' + '/' + args.exp_name + '/' + 'parameter.py.backup')


def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    exp_init(args, rank)  # 只在rank 0创建目录

    IO = IOStream('outputs/' + args.exp_name + '/run.log') if rank == 0 else None
    if rank == 0:
        IO.cprint(str(table_printer(args)))  # 参数可视化

    train_loader, test_loader, num_node_features, num_edge_features = load_dataset(args)

    if num_node_features is None or num_edge_features is None:
        raise ValueError("num_node_features or num_edge_features is None. Check dataset implementation.")

    # 训练和测试
    train(args, IO, train_loader, num_node_features, num_edge_features, rank, world_size)
    test(args, IO, test_loader, num_node_features, num_edge_features, rank)

    # 清理分布式环境
    dist.destroy_process_group()


if __name__ == '__main__':
    main()