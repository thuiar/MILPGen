import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import os
import numpy as np
import pickle
import time
import argparse

class Graph:
    def __init__(self, num_con, num_var, edges, node_features, edge_features, obj_sense):
        self.num_con = num_con
        self.num_var = num_var
        self.edges = edges
        self.node_features = node_features
        self.edge_features = edge_features
        self.obj_sense = obj_sense

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar
    
class DeepVGAE(VGAE):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(in_channels,
                                                          hidden_channels,
                                                          out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss
        
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=int, default=8)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--input_dir', type=str, default="bipartite_graph/4type_problem")
    parser.add_argument('--output_file', type=str, default="02")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    transform = RandomLinkSplit(is_undirected=True, num_val=0.05, num_test=0.1)
    graph_dir = os.path.join(os.getenv("MIPROOT"), args.input_dir)
    arr = os.listdir(graph_dir)
    graphs = []
    graph_name_list = []

    for prob in arr:
        filename = os.path.join(graph_dir, prob)
        if os.path.getsize(filename) > 0: 
            with open(filename, 'rb') as pickle_file:
                graph = pickle.load(pickle_file)
                graphs.append(graph)

            print(f"Loaded {filename}. node_num={graph.num_con+graph.num_var} edge_num={len(graph.edges)}")
            graph_name_list.append(prob)

    # graph_dir = os.path.join(os.getenv("MIPROOT"), "gen_graph")
    # arr = os.listdir(graph_dir)

    # for prob in arr:
    #     filename = os.path.join(graph_dir, prob)
    #     if os.path.getsize(filename) > 0: 
    #         with open(filename, 'rb') as pickle_file:
    #             graph = pickle.load(pickle_file)
    #             graphs.append(graph)

    #         print(f"Loaded {filename}. node num={graph.number_of_nodes} edge num={len(graph.edges)}")
    #         graph_name_list.append(prob)

    data_list = []
    for graph in graphs:
        edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(graph.node_features, dtype=torch.float) 

        x[:, 7] = 0 
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

    model = DeepVGAE(args.in_channels, args.hidden_channels, args.out_channels).to(device)
    # model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_st = time.time()
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        trained_cnt = 0
        epoch_st = time.time()
        for graph in graphs:

            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
            x = torch.tensor(graph.node_features, dtype=torch.float)

            # x[:, 7] = 0 
            data = Data(x=x, edge_index=edge_index).to(device)

            all_edge_index = data.edge_index
            train_data, val_data, test_data = transform(data)


            model.train()
            optimizer.zero_grad()
            loss = model.loss(data.x, train_data.edge_index, data.edge_index)
            print(f"Loss={loss.cpu().item()}")
            loss.backward()
            optimizer.step()

            trained_cnt += 1
            now_time = time.time()
            print(f"Epoch {epoch}, {trained_cnt}/{len(graphs)}", end=" ")
            result1 = f'{(now_time-epoch_st):.2f}'
            result2 = f'{(now_time-total_st):.2f}'
            print(f"epoch used time:{result1}, total used time:{result2}")

    f = open("result/"+args.output_file+".txt", 'w')
    cnt = 0
    for graph in graphs:

        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        x = torch.tensor(graph.node_features, dtype=torch.float)

        # x[:, 7] = 0 
        data = Data(x=x, edge_index=edge_index).to(device)

        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            graph_representation = z.mean(dim=0)

            rep = graph_representation.tolist()
            f.write(graph_name_list[cnt]+"\n")
            cnt += 1
            for i in range(args.out_channels):
                f.write(str(rep[i])+" ")
            f.write("\n")

    f.close()