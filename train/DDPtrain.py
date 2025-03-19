




from dataset  import CustomGraphDataset
import random
from torch_geometric.loader import DataLoader
from Graphormer.parameter import parse_args, IOStream, table_printer
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim

from Graphormer.model import Graphormer
from tqdm import tqdm, trange
def load_dataset(args):
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDatasetmix')



    return dataset, dataset.num_node_features, dataset.num_edge_features
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29504'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()





def train(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    args = parse_args()
    print(str(table_printer(args)))
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    setup(rank, world_size)



    # Create dataset and sampler
    datasets,num_node_features, num_edge_features = load_dataset(args)
    sampler = DistributedSampler(datasets, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(datasets, batch_size=args.train_batch_size, shuffle=False, sampler=sampler)
    model = Graphormer(args, num_node_features, num_edge_features)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    lowest_loss = float('inf')
    epochs = trange(args.epochs, leave=True, desc="Epochs")
    for epoch in epochs:
        sampler.set_epoch(epoch)  # DDP 需要每个 epoch 重新设置 sampler
        total_loss = 0

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train_Loader"):

            data = data.to(rank)

            optimizer.zero_grad()

            out= model(data) # 前向传播

            loss = criterion(out, data.y)  # No need to call .to(device) on data.y
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()


        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch}, Loss: {total_loss:.4f}")
            if total_loss<lowest_loss:
                torch.save(model.module.state_dict(), 'outputs/%s/best.pth' % args.exp_name)
                print("Save best model")# DDP 使用 model.module
            torch.save(model.module.state_dict(), 'outputs/%s/model.pth' % (args.exp_name))  # DDP 使用 model.module

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"Let's use {world_size} GPUs!")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()