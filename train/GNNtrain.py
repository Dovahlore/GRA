from multiprocessing import Manager
from dataset import CustomGraphDataset
import random
from torch_geometric.loader import DataLoader
from GNN.parameter import parse_args, IOStream, table_printer
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from GNN.GNN import LargeGCN
from tqdm import tqdm, trange



def load_dataset(args):
    dataset = CustomGraphDataset(root=args.dataset)

    return dataset, dataset.num_node_features, dataset.num_edge_features


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29504'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def exp_init(args):
    """实验初始化"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)


def train(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    setup(rank, world_size)

    # Create dataset and sampler
    if rank == 0:
        print("Loading dataset...", args.dataset)
    datasets, num_node_features, num_edge_features = load_dataset(args)
    sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(datasets, batch_size=args.train_batch_size, shuffle=False, sampler=sampler)
    model = LargeGCN(args, num_node_features, 1)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model Parameter: {}'.format(total_params))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Using Adam scheduler with learning rate:  %f"%args.lr)
    lowest_loss = float('inf')
    epochs = trange(args.epochs, leave=True, desc="Epochs")

    for epoch in epochs:
        sampler.set_epoch(epoch)  # DDP 需要每个 epoch 重新设置 sampler
        avg_loss = 0

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train_Loader"):
            data = data.to(rank)
            optimizer.zero_grad()
            out = model(data)  # 前向传播
            loss = criterion(out, data.y.squeeze(1))  # No need to call .to(device) on data.y
            loss.backward()
            optimizer.step()  # 反向传播
            total_loss_tensor = torch.tensor(loss.item(), device=rank)
            # 所有 GPU 进行 loss 归约（求和）
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            # 计算全局 loss 平均值
            avg_loss += total_loss_tensor.item() / world_size
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)


        if rank == 0:
            print('Epoch #{:03d}, Train_Loss: {:.6f}'.format(epoch, avg_loss))

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                torch.save(model.module.state_dict(), 'outputs/%s/best.pth' % args.exp_name)  # DDP 使用 model.module
            torch.save(model.module.state_dict(), 'outputs/%s/model.pth' % (args.exp_name))  # DDP 使用 model.module
            print(f"Model saved at 'outputs/%s/model.pth'" % args.exp_name)
    cleanup()


def main():
    args = parse_args()
    exp_init(args)

    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化

    world_size = torch.cuda.device_count()
    print(f"Let's use {world_size} GPUs!")
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
