import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
from geopy import distance
import numpy as np
import pdb
from tqdm import tqdm

class AllDataset(DGLDataset):
    def __init__(self, args):
        self.args = args

        self.names, self.pos, self.labels = self.read_nodes()
        self.node_num = len(self.labels)
        self.edges = self.read_edges()
        self.num_classes = 2
        super().__init__(name='AllDataset')

    def read_edges(self):
        edges = []
        with open('data/distance.txt', 'r') as f:
            lines = f.readlines()

            # add village-village edges
            for line in tqdm(lines):
                line = line.strip().split(',')
                head = int(line[0])
                tail = int(line[1])
                dist = float(line[2])
                if 0 < dist < self.args.village_distance:
                    edges.append([head, tail, dist])

        return np.array(edges)

    def read_nodes(self):
        ls_name = []
        ls_pos = []
        labels = []

        with open('data/all.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                name = line[0]
                lat = float(line[1])
                lng = float(line[2])
                label = int(line[3])

                ls_name.append(name)
                ls_pos.append([lat, lng])
                labels.append(label)

        labels = np.array(labels)
        pos = np.array(ls_pos)

        return ls_name, pos, labels


    def process(self):

        # n_feat = torch.randn(self.node_num, self.args.h_feats)
        edges_src = torch.from_numpy(self.edges[:, 0]).int()
        edges_dst = torch.from_numpy(self.edges[:, 1]).int()
        dist = torch.from_numpy(self.edges[:, 2])
        labels = torch.from_numpy(self.labels)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.node_num)
        # self.graph.ndata['feat'] = n_feat

        self.graph.ndata['label'] = labels
        self.graph.edata['distance'] = dist

        self.graph = dgl.add_self_loop(self.graph)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.node_num
        n_train = int(n_nodes * self.args.train_percent)
        n_val = int(n_nodes * (1 - self.args.train_percent) / 2)

        mask = torch.zeros(n_nodes, dtype = torch.int)

        mask[:n_train] = 0
        mask[n_train:n_train + n_val] = 1
        mask[n_train + n_val:] = 2

        perm = torch.randperm(n_nodes)
        mask = mask[perm]
        train_mask = mask == 0
        val_mask = mask == 1
        test_mask = mask == 2

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class VillageDataset(DGLDataset):
    def __init__(self, args):
        self.args = args
        self.village_num = 0
        self.path = 'data/' + self.args.dataset + '/'

        self.dic_pos, self.dic_village, self.labels = self.read_nodes()
        self.node_num = len(self.dic_pos)
        self.edges = self.read_edges()
        self.num_classes = 2
        super().__init__(name='VillageDataset')

    def read_edges(self):
        edges = []
        with open(self.path + 'distance.txt', 'r') as f:
            lines = f.readlines()

            # add village-village edges
            for line in lines:
                line = line.strip().split(',')
                head = int(line[0])
                tail = int(line[1])
                self.village_num = max([head, tail, self.village_num])
                dist = float(line[2])
                if dist < self.args.village_distance:
                    edges.append([head, tail, dist])

            self.village_num += 1

        # add village-town edges
        nodes_name = list(self.dic_pos.keys())

        for head in range(len(self.dic_village)):
            village = list(self.dic_village.keys())[head]
            town = self.dic_village[village]
            tail = nodes_name.index(town)
            dist = distance.distance(self.dic_pos[village], self.dic_pos[town]).km
            edges.append([head, tail, dist])
            edges.append([tail, head, dist])

        # add town-xian edges
        for head in range(self.village_num, self.node_num):
            tail = self.village_num
            dist = distance.distance(self.dic_pos[nodes_name[head]], self.dic_pos[nodes_name[tail]]).km
            edges.append([head, tail, dist])
            edges.append([tail, head, dist])

        return np.array(edges)

    def read_nodes(self):
        ls_name = []
        ls_pos = []
        labels = []

        with open(self.path + self.args.dataset + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                ls_name.append(line)

        with open(self.path + self.args.dataset + '_zhen.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                ls_name.append(line)

        with open(self.path + self.args.dataset + '.poi', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                lat = float(line[0])
                lng = float(line[1])
                ls_pos.append((lat, lng))

        with open(self.path + self.args.dataset + '_zhen.poi', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                lat = float(line[0])
                lng = float(line[1])
                ls_pos.append((lat, lng))

        dic_pos = {}
        for i in range(len(ls_name)):
            dic_pos[ls_name[i]] = ls_pos[i]

        dic_village = {}
        data = pd.read_excel('data.xlsx', index_col = 0, sheet_name = self.args.dataset)
        for i in range(2, len(data)):
            zhen = data.iloc[i, 0]
            cun = data.iloc[i, 1]
            dic_village[self.args.dataset + zhen + cun] = self.args.dataset + zhen

            if data.iloc[i, 2] == '●':
                labels.append(1)
            else:
                labels.append(0)

        for i in range(len(labels), len(dic_pos)):
            labels.append(0)
        labels = np.array(labels)

        return dic_pos, dic_village, labels


    def process(self):

        # n_feat = torch.randn(self.node_num, self.args.h_feats)
        edges_src = torch.from_numpy(self.edges[:, 0]).int()
        edges_dst = torch.from_numpy(self.edges[:, 1]).int()
        dist = torch.from_numpy(self.edges[:, 2])
        labels = torch.from_numpy(self.labels)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.node_num)
        # self.graph.ndata['feat'] = n_feat

        self.graph.ndata['label'] = labels
        self.graph.edata['distance'] = dist

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.node_num
        n_train = int(n_nodes * self.args.train_percent)
        n_val = int(n_nodes * (1 - self.args.train_percent) / 2)

        mask = torch.zeros(n_nodes, dtype = torch.int)

        mask[:n_train] = 0
        mask[n_train:n_train + n_val] = 1
        mask[n_train + n_val:] = 2

        perm = torch.randperm(n_nodes)
        mask = mask[perm]
        train_mask = mask == 0
        val_mask = mask == 1
        test_mask = mask == 2

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
