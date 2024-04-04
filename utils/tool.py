import copy
import random
import torch_geometric
import torch
from typing import List

class Graph:
    def __init__(self, num_con, num_var, edges, node_features, edge_features, obj_sense):
        self.num_con = num_con
        self.num_var = num_var
        self.edges = edges
        self.node_features = node_features
        self.edge_features = edge_features
        self.obj_sense = obj_sense

class Edge():
    def __init__(self, t: int, idd: int):
        self.t = t # terminal node id
        self.id = idd # edge id
    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.t == other.t and self.id == other.id
        return False
    def __hash__(self):
        return hash((self.t, self.id))

class Node():
    def __init__(self, node_list: List, idd: int, edges: List[Edge], deg1: int, deg2: int):
        self.node_list = node_list
        self.id = idd # node id
        self.edges = copy.deepcopy(edges) # edges is a list of "Edge"
        self.deg1 = deg1 # degree dis=1
        self.deg2 = deg2 # degree dis=2
    def add(self, node_id: int, edge_id: int): # add new edge
        self.edges.append(Edge(t=node_id, idd=edge_id))
        # for i in range(self.deg1):
        #     self.node_list[self.edges[i].t].deg2 += 1
        self.deg1 += 1
    def rmv(self, node_id: int, edge_id: int): # remove edge
        # caution: not implemented for changes on deg2 !!!
        self.edges.remove(Edge(t=node_id, idd=edge_id))
        self.deg1 -= 1

class Bipartite_Graph(torch_geometric.data.Data):
    def __init__(self, num_con: int, num_var: int, num_edge: int, node_con: List[Node], node_var: List[Node],
                 edge: list, feat_con: list, feat_var: list, feat_edge: list, 
                 obj_sense: int, con_split_step: int=0, var_split_step: int=0):
        super().__init__()
        self.num_con = num_con
        self.num_var = num_var
        self.num_edge = num_edge

        self.node_con = copy.deepcopy(node_con)
        self.node_var = copy.deepcopy(node_var)

        self.edge = copy.deepcopy(edge) # (con_id, var_id)
        self.feat_var = copy.deepcopy(feat_var)
        self.feat_con = copy.deepcopy(feat_con)
        self.feat_edge = copy.deepcopy(feat_edge)

        self.obj_sense = obj_sense
        self.con_split_step = con_split_step
        self.var_split_step = var_split_step

    def get_random_type(self) -> int:
        if random.random() < self.con_split_step / (self.con_split_step + self.var_split_step):
            return 0
        return 1

    def split_node(self):
        cur_type, cur_node, max_degree = 0, 0, 0
        for i in range(self.num_con):
            if self.node_con[i].deg1 > max_degree:
                cur_type, cur_node = 0, i
                max_degree = self.node_con[i].deg1
        for i in range(self.num_var):
            if self.node_var[i].deg1 > max_degree:
                cur_type, cur_node = 1, i
                max_degree = self.node_var[i].deg1

        if cur_type == 0: # split con node
            self.con_split_step += 1
            cur_edge = random.choice(self.node_con[cur_node].edges)
            obj_node = cur_edge.t
            cur_edge_id = cur_edge.id
            # remove edges
            self.node_con[cur_node].rmv(obj_node, cur_edge_id)
            self.node_var[obj_node].rmv(cur_node, cur_edge_id)
            # create new node, we have positive pair (cur_node, cur_node2)
            cur_node2 = self.num_con
            self.feat_con.append(self.feat_con[cur_node])
            self.feat_con[cur_node2][4], self.feat_con[cur_node2][5] = 0, 0
            self.feat_con[cur_node2][6] = random.random()
            self.node_con.append(Node(self.node_var, cur_node2, [], 0, 0))
            self.num_con += 1
            # link edges
            self.node_con[cur_node2].add(obj_node, cur_edge_id)
            self.node_var[obj_node].add(cur_node2, cur_edge_id)
            self.edge[0][cur_edge_id] = cur_node2
            # create negative node pair (cur_node, cur_node3)
            cur_node3 = cur_node
            while cur_node3 == cur_node or cur_node3 == cur_node2:
                cur_node3 = random.randint(0, self.num_con)
        else: # split var node
            self.var_split_step += 1
            cur_edge = random.choice(self.node_var[cur_node].edges)
            obj_node = cur_edge.t
            cur_edge_id = cur_edge.id
            # remove edges
            self.node_var[cur_node].rmv(obj_node, cur_edge_id)
            self.node_con[obj_node].rmv(cur_node, cur_edge_id)
            # create new node, we have positive pair (cur_node, cur_node2)
            cur_node2 = self.num_var
            self.feat_var.append(self.feat_var[cur_node])
            self.feat_var[cur_node2][4], self.feat_var[cur_node2][5] = 0, 0
            self.feat_var[cur_node2][6] = random.random()
            self.node_var.append(Node(self.node_con, cur_node2, [], 0, 0))
            self.num_var += 1
            # link edges
            self.node_var[cur_node2].add(obj_node, cur_edge_id)
            self.node_con[obj_node].add(cur_node2, cur_edge_id)
            self.edge[1][cur_edge_id] = cur_node2
            # create negative node pair (cur_node, cur_node3)
            cur_node3 = cur_node
            while cur_node3 == cur_node or cur_node3 == cur_node2:
                cur_node3 = random.randint(0, self.num_var)
        return cur_type, cur_node, cur_node2, cur_node3
    
    def merge_node(self, node_type: int, node1: int, node2: int):
        # no longer maintain information in "Node" class
        if node1 > node2:
            node1, node2 = node2, node1
        new_edge = [[], []]
        new_edge_feat = []
        new_num_edge = self.num_edge
        new_feat = []
        if node_type == 0: # merge con node
            # create new con feat
            for i in range(node2):
                new_feat.append(self.feat_con[i])
            for i in range(node2 + 1, self.num_con):
                new_feat.append(self.feat_con[i])
            new_feat[node1][4] += self.feat_con[node2][4] # add degree of merged node
            # TODO: only maintain deg1
            self.feat_con = copy.deepcopy(new_feat)
            self.num_con -= 1
            # create new edge info
            link_node = set() # set of nodes link to node1 and node2
            for i in range(self.num_edge):
                con_id, var_id = self.edge[0][i], self.edge[1][i]
                if con_id == node1 or con_id == node2:
                    if var_id in link_node:
                        new_num_edge -= 1
                        self.feat_con[node1][4] -= 1
                        self.feat_var[var_id][4] -= 1
                        continue
                    new_edge[0].append(node1)
                    new_edge[1].append(var_id)
                    new_edge_feat.append(self.feat_edge[i])
                    link_node.add(var_id)
                elif con_id > node2:
                    new_edge[0].append(con_id - 1)
                    new_edge[1].append(var_id)
                    new_edge_feat.append(self.feat_edge[i])
                else:
                    new_edge[0].append(con_id)
                    new_edge[1].append(var_id)
                    new_edge_feat.append(self.feat_edge[i])
        else: # var node
            for i in range(node2):
                new_feat.append(self.feat_var[i])
            for i in range(node2 + 1, self.num_var):
                new_feat.append(self.feat_var[i])
            new_feat[node1][4] += self.feat_var[node2][4] # add degree of merged node
            # TODO: only maintain deg1
            self.feat_var = copy.deepcopy(new_feat)
            self.num_var -= 1
            # create new edge info
            link_node = set()
            for i in range(self.num_edge):
                con_id, var_id = self.edge[0][i], self.edge[1][i]
                if var_id == node1 or var_id == node2:
                    if con_id in link_node:
                        new_num_edge -= 1
                        self.feat_var[node1][4] -= 1
                        self.feat_con[con_id][4] -= 1
                        continue
                    new_edge[0].append(con_id)
                    new_edge[1].append(node1)
                    new_edge_feat.append(self.feat_edge[i])
                    link_node.add(con_id)
                elif var_id > node2:
                    new_edge[0].append(con_id)
                    new_edge[1].append(var_id - 1)
                    new_edge_feat.append(self.feat_edge[i])
                else:
                    new_edge[0].append(con_id)
                    new_edge[1].append(var_id)
                    new_edge_feat.append(self.feat_edge[i])
        self.edge = copy.deepcopy(new_edge)
        self.feat_edge = copy.deepcopy(new_edge_feat)
        self.num_edge = copy.deepcopy(new_num_edge)

def equal(x, y) -> bool:
    if abs(x - y) < 1e-6:
        return True
    return False

def load_graph(cur_graph: Bipartite_Graph) -> Graph:
    node_features = []
    edges = []
    edge_features = []

    for i in range(cur_graph.num_con):
        if equal(cur_graph.feat_con[i][0], 1):
            tmp = copy.deepcopy([0, 0, 0, 1, 0, 0, 0, 0])
        elif equal(cur_graph.feat_con[i][1], 1):
            tmp = copy.deepcopy([0, 0, 0, 0, 1, 0, 0, 0])
        elif equal(cur_graph.feat_con[i][2], 1):
            tmp = copy.deepcopy([0, 0, 0, 0, 0, 1, 0, 0])
        if equal(cur_graph.feat_con[i][0], 1):
            tmp[6] = cur_graph.feat_con[i][3] + random_epsilon(0, 0.02)
        if equal(cur_graph.feat_con[i][2], 1):
            tmp[6] = cur_graph.feat_con[i][3] - random_epsilon(0, 0.02)
        node_features.append(tmp)

    for i in range(cur_graph.num_var):
        if equal(cur_graph.feat_var[i][0], 1):
            tmp = copy.deepcopy([1, 0, 0, 0, 0, 0, 0, 0])
        elif equal(cur_graph.feat_var[i][1], 1):
            tmp = copy.deepcopy([0, 1, 0, 0, 0, 0, 0, 0])
        elif equal(cur_graph.feat_var[i][2], 1):
            tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0, 0])
        tmp[7] = cur_graph.feat_var[i][3]
        node_features.append(tmp)

    for i in range(cur_graph.num_edge):
        edge_features.append(cur_graph.feat_edge[i][0])
        edges.append((cur_graph.edge[0][i], cur_graph.edge[1][i] + cur_graph.num_con))
    
    return Graph(num_con=cur_graph.num_con, num_var=cur_graph.num_var,
                 edges=edges, node_features=node_features, edge_features=edge_features,
                 obj_sense=cur_graph.obj_sense)

def load_bipartite_graph(graph: Graph) -> Bipartite_Graph:
    # create node list
    node_con: List[Node] = []
    node_var: List[Node] = []
    for i in range(graph.num_con):
        node_con.append(Node(node_var, i, [], 0, 0))
    for i in range(graph.num_var):
        node_var.append(Node(node_con, i, [], 0, 0))

    # deal edge firstly
    num_edge = len(graph.edges)
    edge = [[], []]
    feat_edge = []
    for i in range(num_edge):
        feat_edge.append([graph.edge_features[i]])
        con_id, var_id = graph.edges[i]
        var_id -= graph.num_con
        edge[0].append(con_id)
        edge[1].append(var_id)
        node_con[con_id].add(var_id, i)
        node_var[var_id].add(con_id, i)

    # deal node secondly
    feat_con = []
    for i in range(graph.num_con):
        if graph.node_features[i][3] == 1:
            tmp = copy.deepcopy([1, 0, 0, 0, 0, 0, 0])
        elif graph.node_features[i][4] == 1:
            tmp = copy.deepcopy([0, 1, 0, 0, 0, 0, 0])
        elif graph.node_features[i][5] == 1:
            tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0])
        tmp[3] = graph.node_features[i][6]
        tmp[4], tmp[5] = node_con[i].deg1, node_con[i].deg2
        tmp[6] = random.random()
        feat_con.append(tmp)

    feat_var = []
    for i in range(graph.num_con, graph.num_con+graph.num_var):
        if graph.node_features[i][0] == 1:
            tmp = copy.deepcopy([1, 0, 0, 0, 0, 0, 0])
        elif graph.node_features[i][1] == 1:
            tmp = copy.deepcopy([0, 1, 0, 0, 0, 0, 0])
        elif graph.node_features[i][2] == 1:
            tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0])
        tmp[3] = graph.node_features[i][7]
        tmp[4], tmp[5] = node_var[i - graph.num_con].deg1, node_var[i - graph.num_con].deg2
        tmp[6] = random.random()
        feat_var.append(tmp)

    return Bipartite_Graph(num_con=graph.num_con, num_var=graph.num_var, num_edge=num_edge,
                           node_con=node_con, node_var=node_var, edge=edge,
                           feat_con=feat_con, feat_var=feat_var, feat_edge=feat_edge,
                           obj_sense=graph.obj_sense, con_split_step=0, var_split_step=0)

def random_epsilon(a: float, b: float) -> float:
    return a + random.random() * (b - a)

def merge_bipartite_graph(A: Bipartite_Graph, B: Bipartite_Graph, typ: int, x: int, y: int) -> Bipartite_Graph:
    # merge node x in graph A and node y in graph B, node x and y must be of the same type type.
    # only update feat_con, feat_var
    # node_con, node_var are no longer maintained 
    C = copy.deepcopy(A)
    C.con_split_step += B.con_split_step
    C.var_split_step += B.var_split_step
    C.num_edge += B.num_edge
    if typ == 0: # merge con node
        for i in range(B.num_con):
            if i == y:
                continue
            C.feat_con.append(B.feat_con[i])
        for i in range(B.num_var):
            C.feat_var.append(B.feat_var[i])
        C.num_con += B.num_con - 1
        C.num_var += B.num_var
        for i in range(B.num_edge):
            if B.edge[0][i] == y:
                C.edge[0].append(x)
                C.edge[1].append(B.edge[1][i] + A.num_var)
            else:
                if B.edge[0][i] < y:
                    C.edge[0].append(B.edge[0][i] + A.num_con)
                else:
                    C.edge[0].append(B.edge[0][i] + A.num_con - 1)
                C.edge[1].append(B.edge[1][i] + A.num_var)
            C.feat_edge.append(B.feat_edge[i])
    else: # merge var node
        for i in range(B.num_con):
            C.feat_con.append(B.feat_con[i])
        for i in range(B.num_var):
            if i == y:
                continue
            C.feat_var.append(B.feat_var[i])
        C.num_con += B.num_con
        C.num_var += B.num_var - 1
        for i in range(B.num_edge):
            if B.edge[1][i] == y:
                C.edge[0].append(B.edge[0][i] + A.num_con)
                C.edge[1].append(x)
            else:
                C.edge[0].append(B.edge[0][i] + A.num_con)
                if B.edge[1][i] < y:
                    C.edge[1].append(B.edge[1][i] + A.num_var)
                else:
                    C.edge[1].append(B.edge[1][i] + A.num_var - 1)
            C.feat_edge.append(B.feat_edge[i])
    return C