import torch
import torch.nn as nn
import copy
import pickle
import numpy as np
import os
import time
import argparse
import random
from torch_geometric.data import Data
from utils.model import *
from utils.tool import Graph, Bipartite_Graph, load_bipartite_graph, load_graph
from typing import List
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["direct", "collect", "train", "inference"])
    # for direct (collecting)
    parser.add_argument('--input_dir', type=str, default="bipartite_graph/4type_problem")
    parser.add_argument('--class_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="bipartite_graph/step6_trees")
    parser.add_argument('--classify_file', type=str, default="03")
    # for collecting
    parser.add_argument('--problem_file', type=str, default="bipartite_graph/4type_problem/CAT_0")
    parser.add_argument('--gen_tuple_dir', type=str, default="bipartite_graph/step5_tuple/type1")
    parser.add_argument('--collect_steps', type=int, default=1000)
    # for training
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model_file', type=str, default="tmp/model_step5.pth")
    parser.add_argument('--writer_path', type=str, default="result/type0")
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--load_tuple_dir', type=str, default="bipartite_graph/step5_tuple/type1")
    parser.add_argument('--train_num', type=int, default=1000)
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # for both training and inference
    parser.add_argument('--load_model_file', type=str, default="tmp/model_step5.pth")
    # for inference
    parser.add_argument('--load_tree', type=int, default=1)
    parser.add_argument('--load_tree_file', type=str, default="bipartite_graph/step5_tree/type1/tuple999")
    parser.add_argument('--load_tuple_file', type=str, default="bipartite_graph/step5_tuple/type1/tuple999")
    parser.add_argument('--merge_steps', type=int, default=100)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--save_graph_file', type=str, default="result/step5_result.pkl")
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = GNNPolicy_with_MLP(emb_size=32, cons_nfeats=7, var_nfeats=7, edge_nfeats=1, output_dim=32)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    pos_label = torch.tensor([[1.0]]).float().to(device)
    neg_label = torch.tensor([[0.0]]).float().to(device)

    if args.mode == "direct": # directly collect tree
        graph_dir = os.path.join(os.getenv("MIPROOT"), args.input_dir)
        graphs = []
        prob_name_list = []
        f = open("result/"+args.classify_file+".txt", 'r')
        lines = f.readlines()
        f.close()
        data_list: List[Bipartite_Graph] = []
        for line in lines:
            tmp = line.split()
            if int(tmp[1]) != args.class_id:
                continue
            prob = tmp[0]

            filename = os.path.join(graph_dir, prob)
            if not os.path.isfile(filename):
                continue
            with open(filename, 'rb') as pickle_file:
                graph = pickle.load(pickle_file)
                data_list.append(load_bipartite_graph(graph))

            print(f"Loaded {filename}. var_num={graph.num_var} con_num={graph.num_con} edge num={len(graph.edges)}")
            prob_name_list.append(prob)
        print(f"Using class {args.class_id}, instance count = {len(data_list)}")
        split_num = 0
        total_start = time.time()
        for cur_graph in data_list:
            print("Splitting trees:", prob_name_list[split_num])
            time_start = time.time()
            split_step = cur_graph.num_edge - cur_graph.num_con - cur_graph.num_var + 1
            step_cnt = 0
            while True:
                cur_type, node_id, node_id2, node_id3 = cur_graph.split_node()
                step_cnt += 1
                if step_cnt % 100 == 0:
                    print(f"step_cnt={step_cnt}, ")
                    print(f"Current time:{time.time() - time_start}, Total time:{time.time() - time_start}")
                if step_cnt == args.collect_steps:
                    print(f"Done with {step_cnt} steps.")
                    break
            print(f"Split time for {prob_name_list[split_num]}: {time.time() - time_start}")
            print(f"Total time:{time.time() - time_start}")

            cur_graph.node_con = None
            cur_graph.node_var = None

            with open(os.path.join(args.output_dir, prob_name_list[split_num]+".pkl"), 'wb') as tup_file:
                pickle.dump(cur_graph, tup_file)
                tup_file.close()
            split_num += 1

    elif args.mode == "collect": # collect training data, step by step
        time_start = time.time()
        pickle_file = open(args.problem_file, 'rb')
        graph = pickle.load(pickle_file)
        cur_graph = load_bipartite_graph(graph=graph)
        step_cnt = 0
        while True:
            cur_type, node_id, node_id2, node_id3 = cur_graph.split_node()
            dump_graph = copy.deepcopy(cur_graph)
            dump_graph.node_con = None
            dump_graph.node_var = None
            tup = [dump_graph, cur_type, node_id, node_id2, node_id3]

            with open(os.path.join(args.gen_tuple_dir, "tuple"+str(step_cnt)), 'wb') as tup_file:
                pickle.dump(tup, tup_file)
                tup_file.close()
            step_cnt += 1
            if step_cnt % 100 == 0:
                print(f"step_cnt = {step_cnt}")
                print(f"Current time: {time.time() - time_start}")
            if step_cnt == args.collect_steps:
                print(f"Done with {step_cnt} steps.")
                break
        print(f"Total time: {time.time() - time_start}")
    elif args.mode == "train":
        writer = SummaryWriter(args.writer_path)
        if args.load_model == 1:
            model.load_state_dict(torch.load(args.load_model_file))
        tot_arr = os.listdir(args.load_tuple_dir)
        if len(tot_arr) < args.train_num + args.test_num:
            print("Graph not enough!")
            quit()
        graph_num = args.train_num
        test_num = args.test_num
        arr = tot_arr[:graph_num]

        test_arr = tot_arr[graph_num:graph_num+test_num]

        bipartite_graph_list: List[Bipartite_Graph] = []
        test_graph_list: List[Bipartite_Graph] = []
        node_list = []
        test_node_list = []

        time_start = time.time()
        for cnt, tup_filename in enumerate(arr):
            tup_file = os.path.join(args.load_tuple_dir, tup_filename)
            with open(tup_file, "rb") as f:
                graph, node_type, node_id, node_id2, node_id3 = pickle.load(f)
                graph.edge=copy.deepcopy(torch.LongTensor(graph.edge))
                graph.feat_var=copy.deepcopy(torch.FloatTensor(graph.feat_var))
                graph.feat_con=copy.deepcopy(torch.FloatTensor(graph.feat_con))
                graph.feat_edge=copy.deepcopy(torch.FloatTensor(graph.feat_edge))
                
                bipartite_graph_list.append(graph)
                node_list.append((node_type, node_id, node_id2, node_id3))
                f.close()
            if cnt % 50 == 0:
                print(f"Train: Loaded {cnt}/{graph_num} graphs.")

        for cnt, tup_filename in enumerate(test_arr):
            tup_file = os.path.join(args.load_tuple_dir, tup_filename)
            with open(tup_file, "rb") as f:
                graph, node_type, node_id, node_id2, node_id3 = pickle.load(f)
                graph.edge=copy.deepcopy(torch.LongTensor(graph.edge))
                graph.feat_var=copy.deepcopy(torch.FloatTensor(graph.feat_var))
                graph.feat_con=copy.deepcopy(torch.FloatTensor(graph.feat_con))
                graph.feat_edge=copy.deepcopy(torch.FloatTensor(graph.feat_edge))

                test_graph_list.append(graph)
                test_node_list.append((node_type, node_id, node_id2, node_id3))
                f.close()
            if cnt % 50 == 0:
                print(f"Test: Loaded {cnt}/{test_num} graphs.")  
        print(f"Load time={time.time() - time_start}, Loaded {graph_num+test_num} tuples.")
        for i in range(args.epoch):

            # Train Begin
            model.train()
            total_loss = 0
            acc_loss = torch.tensor(0.0).to(device)
            optimizer.zero_grad()
            for j in range(len(bipartite_graph_list)):
                # print("j =", j)
                bipartite_graph = bipartite_graph_list[j].to(device)
                
                node_type, node_id, node_id2, node_id3 = node_list[j]

                if node_type == 0:
                    if node_id3 == bipartite_graph.num_con:
                        node_id3 -= 1
                else:
                    if node_id3 == bipartite_graph.num_var:
                        node_id3 -= 1

                pos_pred = model(constraint_features=bipartite_graph.feat_con,
                                 variable_features=bipartite_graph.feat_var,
                                 edge_indices=bipartite_graph.edge,
                                 edge_features=bipartite_graph.feat_edge,
                                 node_type=node_type,
                                 n1_list=torch.tensor([node_id]).to(device),
                                 n2_list=torch.tensor([node_id2]).to(device))
                neg_pred = model(constraint_features=bipartite_graph.feat_con,
                                 variable_features=bipartite_graph.feat_var,
                                 edge_indices=bipartite_graph.edge,
                                 edge_features=bipartite_graph.feat_edge,
                                 node_type=node_type,
                                 n1_list=torch.tensor([node_id]).to(device),
                                 n2_list=torch.tensor([node_id3]).to(device))         
                acc_loss += criterion(pos_pred, pos_label) + criterion(neg_pred, neg_label)

                if j % args.batch_size == args.batch_size - 1 or j == graph_num - 1:
                    acc_loss.backward()
                    optimizer.step()
                    total_loss += acc_loss.cpu().item()
                    acc_loss = torch.tensor(0.0).to(device)
                    optimizer.zero_grad()                

            writer.add_scalar('Loss', acc_loss.cpu().item(), i)
            print(f"Epoch {i}, loss={total_loss}")
            # Test Begin
            acc = 0
            model.eval()
            with torch.no_grad():
                for j in range(len(test_graph_list)):
                    bipartite_graph = test_graph_list[j].to(device)
                    node_type, node_id, node_id2, node_id3 = test_node_list[j]

                    if node_type == 0:
                        if node_id3 == bipartite_graph.num_con:
                            node_id3 -= 1
                    else:
                        if node_id3 == bipartite_graph.num_var:
                            node_id3 -= 1

                    pos_pred = model(constraint_features=bipartite_graph.feat_con,
                                    variable_features=bipartite_graph.feat_var,
                                    edge_indices=bipartite_graph.edge,
                                    edge_features=bipartite_graph.feat_edge,
                                    node_type=node_type,
                                    n1_list=torch.tensor([node_id]).to(device),
                                    n2_list=torch.tensor([node_id2]).to(device))
                    neg_pred = model(constraint_features=bipartite_graph.feat_con,
                                    variable_features=bipartite_graph.feat_var,
                                    edge_indices=bipartite_graph.edge,
                                    edge_features=bipartite_graph.feat_edge,
                                    node_type=node_type,
                                    n1_list=torch.tensor([node_id]).to(device),
                                    n2_list=torch.tensor([node_id3]).to(device))
                    if pos_pred > neg_pred:
                        acc += 1
            print(f"Test acc {acc}/{len(test_graph_list)}, {acc/len(test_graph_list)}")
            writer.add_scalar('Acc', acc/test_num, i)
        writer.close()
        torch.save(model.state_dict(), args.save_model_file)
    elif args.mode == "inference":
        model.load_state_dict(torch.load(args.load_model_file))
        model.eval()

        if args.load_tree == 1:
            with open(args.load_tree_file, "rb") as f:
                cur_graph = pickle.load(f)
                print(f"Loaded graph. Var num={cur_graph.num_var}, Con num={cur_graph.num_con}, Edge num={cur_graph.num_edge}")
                f.close()
        else:
            with open(args.load_tuple_file, "rb") as f:
                tup = pickle.load(f)
                cur_graph, node_type, node_id, node_id2, node_id3 = tup
                print(f"Loaded graph. Var num={cur_graph.num_var}, Con num={cur_graph.num_con}, Edge num={cur_graph.num_edge}")
                f.close()

        time_start = time.time()
        with torch.no_grad():
            # execute for specified steps
            con_count = 0
            var_count = 0
            for i in range(args.merge_steps):
                edge=torch.LongTensor(cur_graph.edge).to(device)
                feat_var=torch.FloatTensor(cur_graph.feat_var).to(device)
                feat_con=torch.FloatTensor(cur_graph.feat_con).to(device)
                feat_edge=torch.FloatTensor(cur_graph.feat_edge).to(device)
                node_type = cur_graph.get_random_type()
                if node_type == 0: # merge constraint
                    con_count += 1
                    node1_list = torch.tensor(random.sample(range(cur_graph.num_con), args.k)).to(device)
                    node2_list = torch.tensor(random.sample(range(cur_graph.num_con), args.k)).to(device)
                else: # merge variable
                    var_count += 1
                    node1_list = torch.tensor(random.sample(range(cur_graph.num_var), args.k)).to(device)
                    node2_list = torch.tensor(random.sample(range(cur_graph.num_var), args.k)).to(device)

                pred = model(constraint_features=feat_con,
                             variable_features=feat_var,
                             edge_indices=edge,
                             edge_features=feat_edge,
                             node_type=node_type,
                             n1_list=node1_list,
                             n2_list=node2_list)    

                pair_id = pred.argmax().cpu().item()
                node1 = node1_list[pair_id].item()
                node2 = node2_list[pair_id].item()
                if node1 == node2:
                    continue
                cur_graph.merge_node(node_type, node1, node2)
                if i % 50 == 0:
                    print(f"Step {i}, merged_con_cnt={con_count} merged_var_cnt={var_count}, cost time={time.time()-time_start}")
            
        print(f"Result Graph: variable num={cur_graph.num_var} constraint num={cur_graph.num_con} edge num={cur_graph.num_edge}")
        print(f"Cost time: {time.time() - time_start}")
        graph = load_graph(cur_graph)
        graph_file = open(args.save_graph_file, 'wb')
        pickle.dump(graph, graph_file)
        graph_file.close()