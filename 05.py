import torch
from torch.nn.functional import cosine_similarity
import pickle
import copy
import numpy as np
import os
import time
import argparse
import random
from torch_geometric.data import Data
from utils.model import *
from utils.tool import Bipartite_Graph, merge_bipartite_graph
from typing import List
import random

def find_most_similar_pair(X, Y):
    max_similarity = -1.0
    most_similar_pair = None

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            similarity = cosine_similarity(x, y, dim=0)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (i, j)

    return most_similar_pair, max_similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_tree_dir', type=str, default="bipartite_graph/step5_trees/type2/L")
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--load_model_file', type=str, default="tmp/model_step5_type2.pth")
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--save_tree_file', type=str, default="bipartite_graph/step5_merged_trees/type2/A_4.pkl")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GNNPolicy_with_MLP(emb_size=32, cons_nfeats=7, var_nfeats=7, edge_nfeats=1, output_dim=32)
    model = model.to(device)
    model.load_state_dict(torch.load(args.load_model_file))

    arr = os.listdir(args.load_tree_dir)
    tree_num = len(arr)

    x = random.randint(0, tree_num - 1)
    with open(os.path.join(args.load_tree_dir, arr[x]), "rb") as f:
        res_graph: Bipartite_Graph = pickle.load(f)
        f.close()

    for i in range(args.size - 1):
        x = random.randint(0, tree_num - 1)
        with open(os.path.join(args.load_tree_dir, arr[x]), "rb") as f:
            upd_graph: Bipartite_Graph = pickle.load(f)
            f.close()
        p = (res_graph.con_split_step+upd_graph.con_split_step) / (res_graph.con_split_step+upd_graph.con_split_step + res_graph.var_split_step+upd_graph.var_split_step)

        res_edge=torch.LongTensor(res_graph.edge).to(device)
        res_feat_var=torch.FloatTensor(res_graph.feat_var).to(device)
        res_feat_con=torch.FloatTensor(res_graph.feat_con).to(device)
        res_feat_edge=torch.FloatTensor(res_graph.feat_edge).to(device)

        upd_edge=torch.LongTensor(upd_graph.edge).to(device)
        upd_feat_var=torch.FloatTensor(upd_graph.feat_var).to(device)
        upd_feat_con=torch.FloatTensor(upd_graph.feat_con).to(device)
        upd_feat_edge=torch.FloatTensor(upd_graph.feat_edge).to(device)

        if random.random() < p: # merge con node
            merge_typ = 0
            node1_list = torch.tensor(random.sample(range(res_graph.num_con), args.k)).to(device)
            node2_list = torch.tensor(random.sample(range(upd_graph.num_con), args.k)).to(device)

            res_pred = model.get_embedding(constraint_features=res_feat_con,
                                           variable_features=res_feat_var,
                                           edge_indices=res_edge,
                                           edge_features=res_feat_edge,
                                           node_type=0,
                                           n_list=node1_list)
            upd_pred = model.get_embedding(constraint_features=upd_feat_con,
                                           variable_features=upd_feat_var,
                                           edge_indices=upd_edge,
                                           edge_features=upd_feat_edge,
                                           node_type=0,
                                           n_list=node2_list)

        else: # merge var node
            merge_typ = 1
            node1_list = torch.tensor(random.sample(range(res_graph.num_var), args.k)).to(device)
            node2_list = torch.tensor(random.sample(range(upd_graph.num_var), args.k)).to(device)

            res_pred = model.get_embedding(constraint_features=res_feat_con,
                                           variable_features=res_feat_var,
                                           edge_indices=res_edge,
                                           edge_features=res_feat_edge,
                                           node_type=1,
                                           n_list=node1_list)
            upd_pred = model.get_embedding(constraint_features=upd_feat_con,
                                           variable_features=upd_feat_var,
                                           edge_indices=upd_edge,
                                           edge_features=upd_feat_edge,
                                           node_type=1,
                                           n_list=node2_list)
        (n1, n2), p = find_most_similar_pair(res_pred, upd_pred)
        n1 = node1_list[n1].cpu().item()
        n2 = node2_list[n2].cpu().item()
        print(f"Merging node type={merge_typ}, n1={n1}, n2={n2} p={p}")
        res_graph = merge_bipartite_graph(res_graph, upd_graph, merge_typ, n1, n2)
    f = open(args.save_tree_file, 'wb')
    pickle.dump(res_graph, f)
    f.close()