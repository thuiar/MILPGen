import pickle
import os
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import argparse

class Graph:
    def __init__(self, number_of_nodes, edges, node_features, edge_features, obj_sense):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.node_features = node_features
        self.edge_features = edge_features
        self.obj_sense = obj_sense

class GurobiSolver:

    def __init__(self):
        self.m = gp.Model()
        self.m.setParam('OutputFlag', 1)
        self.m.setParam('PreCrush', 1)

    def read_graph_to_model(self, graph, num_var, num_con):
        vars = []
        for i in range(num_con, num_con+num_var):
            if graph.node_features[i][1] == 1:  # integer variable
                var = self.m.addVar(vtype='I', obj=graph.node_features[i][7])
            elif graph.node_features[i][2] == 1:  # binary variable
                var = self.m.addVar(vtype='B', obj=graph.node_features[i][7])
            else:  # continuous variable
                var = self.m.addVar(vtype='C', obj=graph.node_features[i][7])
            vars.append(var)

        constraints = defaultdict(gp.LinExpr)
        for edge, coefficient in zip(graph.edges, graph.edge_features):
            constraints[edge[0]] += coefficient * vars[edge[1] - num_con]   

        for i in range(num_con):
            if graph.node_features[i][3] == 1:  # less than
                self.m.addConstr(constraints[i], GRB.LESS_EQUAL, graph.node_features[i][6])
            elif graph.node_features[i][4] == 1:  # equal to
                self.m.addConstr(constraints[i], GRB.EQUAL, graph.node_features[i][6])
            elif graph.node_features[i][5] == 1:  # greater than
                self.m.addConstr(constraints[i], GRB.GREATER_EQUAL, graph.node_features[i][6])
            else:
                raise ValueError

        if graph.obj_sense == 1:
            self.m.ModelSense = GRB.MAXIMIZE
        else:
            self.m.ModelSense = GRB.MINIMIZE
        self.m.update()

    def solve_model(self, time_limit):
        self.m.setParam('TimeLimit', time_limit)
        self.m.update()
        self.m.optimize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="tmp_graph/csched008")
    parser.add_argument('--time_limit', type=int, default=600)
    args = parser.parse_args()

    print(f"Solving problem: {args.file}")
    with open(args.file, 'rb') as f:
        graph = pickle.load(f)

    print("Var num =", graph.num_var, "Con num =", graph.num_con, "Edge num =", len(graph.edges))
        
    solver = GurobiSolver()
    solver.read_graph_to_model(graph, graph.num_var, graph.num_con)

    solver.solve_model(args.time_limit)