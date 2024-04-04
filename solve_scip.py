import pickle
import os
from pyscipopt import Model, quicksum
import argparse
import numpy as np
from utils.tool import Graph

def equal(x, y):
    if abs(x - y) < 1e-6:
        return True
    return False

def graph_to_scip_model(graph: Graph):
    model = Model()

    binary_cnt = 0
    cont_cnt = 0
    int_cnt = 0
    # Add variables to the model
    x = []
    for i in range(graph.num_var):
        node_feature = graph.node_features[i+graph.num_con]
        if equal(node_feature[2], 1):
            x.append(model.addVar(vtype='B', obj=node_feature[7]))
            binary_cnt += 1
        elif equal(node_feature[1], 1):
            x.append(model.addVar(vtype='I', obj=node_feature[7]))
            int_cnt += 1
        else:
            x.append(model.addVar(vtype='C', obj=node_feature[7]))
            cont_cnt += 1

    print(f"binary count={binary_cnt}, integer count={int_cnt}, continous count={cont_cnt}")
    # Add constraints to the model
    for i in range(graph.num_con):
        # print(f"i={i}")
        node_feature = graph.node_features[i]
        constraint_expr = quicksum(graph.edge_features[e_index] * x[e[1] - graph.num_con] for e_index, e in enumerate(graph.edges) if e[0] == i)
        if equal(node_feature[3], 1): # <= constraint
            model.addCons(constraint_expr <= node_feature[6])
        elif equal(node_feature[4], 1): # = constraint
            model.addCons(constraint_expr == node_feature[6])
        elif equal(node_feature[5], 1): # >= constraint
            model.addCons(constraint_expr >= node_feature[6])

    # set the sense of the objective
    if equal(graph.obj_sense, 1): # maximize
        model.setMaximize()
    else: # minimize
        model.setMinimize()

    return model, x

def calculate_statistics(values):
    values_np = np.array(values)
    mean = np.mean(values_np)
    std = np.std(values_np)
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--time_limit', type=int, default=1200)
    parser.add_argument('--solution', type=str, default="none")
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        graph = pickle.load(f)

    print("Var num =", graph.num_var, "Con num =", graph.num_con, "Edge num =", len(graph.edges))
        
    
    model, x = graph_to_scip_model(graph)
    model.setParam('limits/time', args.time_limit)
    # model.writeProblem("1.mps")
    model.optimize()
    print("Optimization status:", model.getStatus())

    if model.getStatus() == "optimal":
        primal_solution, dual_value = [model.getVal(var) for var in model.getVars()], model.getObjVal()
        primal_mean, primal_std = calculate_statistics(primal_solution)
        dual_mean, dual_std = calculate_statistics([dual_value])  # Assuming single dual value

        print(f"Primal Solution Mean: {primal_mean}, Primal Solution Std: {primal_std}")
        print(f"Dual Solution Mean: {dual_mean}, Dual Solution Std: {dual_std}")