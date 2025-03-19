# -*- coding: utf-8 -*-
import os,networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import random
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
from typing import Union, Tuple, Dict, List
def floyd_warshall_source_to_all(G, source, cutoff=None):
    "Floyd-Warshall算法查询最短路径(BFS遍历图)"
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break

    return node_paths, edge_paths
def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths

def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths
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
        size=600
        k=[31,32,33,34,35]
        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
         # 加载原始数据
        edge_indexs = {}
        for tk in k:
            edge_indexs[tk]= pickle.load(open( './edge_indexs/edge_index%d.pkl'%tk, 'rb'))  # 加载边索引


        for i in range(size):

            cur_k=random.choice(k)
            print(i,cur_k)
            # 节点特征（100节点，2维）
            x = torch.rand((cur_k*cur_k, 2), dtype=torch.float)

            # 图标签（标量）
            y = torch.randint(0, 2, (cur_k*cur_k,1), dtype=torch.float)

            # 边特征（示例）
            edge_attr = torch.ones((edge_indexs[cur_k].size(1),1),dtype=torch.long)


            # 构造数据对象
            data = Data(
                x=x,
                edge_index=edge_indexs[cur_k],
                y=y,
                edge_attr=edge_attr
            )

            #node_paths, edge_paths = shortest_path_distance(data)
            #data.node_paths = node_paths
            #data.edge_paths = edge_paths

            data.deg = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
            data_list.append(data)




        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Processed dataset saved to {}'.format(self.processed_paths[0]))


if __name__ == '__main__':
    dataset = CustomGraphDataset(root='./Datasets/CustomGraphDatasetmix600')
