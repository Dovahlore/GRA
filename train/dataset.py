# -*- coding: utf-8 -*-
import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
from torch_geometric.utils import degree

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)


    @property
    def raw_file_names(self):
        return []


    @property
    def processed_file_names(self):
        return ['data.pt']


    def download(self):  # Download to `self.raw_dir`.
        pass

    def process(self):
        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
         # 加载原始数据
        edge_index = pickle.load(open( './edge_index.pkl', 'rb'))  # 加载边索引
        print(edge_index.shape)

        for i in range(64):
            # 节点特征（100节点，2维）
            x = torch.rand((49, 2), dtype=torch.float)
            print(x.shape)
            # 图标签（标量）
            y = torch.randint(0, 1, (49,1), dtype=torch.long)
            print(y.shape)
            # 边特征（示例）
            edge_attr = torch.ones((edge_index.size(1),1),dtype=torch.long)
            print(edge_attr.shape)

            # 构造数据对象
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                edge_attr=edge_attr
            )
            data.deg = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
            data_list.append(data)




        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Processed dataset saved to {}'.format(self.processed_paths[0]))


if __name__ == '__main__':
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDataset49')
