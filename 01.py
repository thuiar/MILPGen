import numpy as np
import argparse
import pickle
import random
import time
import os
import copy

class Graph:
    def __init__(self, num_con, num_var, edges, node_features, edge_features, obj_sense):
        self.num_con = num_con
        self.num_var = num_var
        self.edges = edges
        self.node_features = node_features
        self.edge_features = edge_features
        self.obj_sense = obj_sense

def generate_IS(N, M):
    '''
    Function Description:
    Generate instances of the maximum independent set problem in a general graph.
    
    Parameters:
    - N: Number of vertices in the graph.
    - M: Number of edges in the graph.

    Return: 
    Relevant parameters of the generated maximum independent set problem.
    '''
    
    # n represents the number of decision variables, where each vertex in the graph corresponds to a decision variable.
    # m represents the number of constraints, where each edge in the graph corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraint: randomly generate an edge and impose a constraint that the vertices connected by the edge cannot be selected simultaneously.
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        site[i].append(x)
        value[i].append(1)
        site[i].append(y) 
        value[i].append(1)
        constraint[i] = 1
        constraint_type[i] = 1
        k[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a vertex is a random value.
    for i in range(N):
        coefficient[i] = random.random()
    
    node_features = []
    edges = []
    edge_features = []
    
    for i in range(m):
        if constraint_type[i] == 1:
            tmp = copy.deepcopy([0, 0, 0, 1, 0, 0, 0, 0])
        elif constraint_type[i] == 2:
            tmp = copy.deepcopy([0, 0, 0, 0, 0, 1, 0, 0])
        else: 
            tmp = copy.deepcopy([0, 0, 0, 0, 1, 0, 0, 0])
        tmp[6] = constraint[i]
        node_features.append(tmp)

    for i in range(n):
        tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0, 0])
        tmp[7] = coefficient[i]
        node_features.append(tmp)

    for i in range(m):
        for j in range(k[i]):
            edges.append((i, site[i][j] + m))
            edge_features.append(value[i][j])

    obj_sense = 1

    graph = Graph(num_con=m,
                  num_var=n,
                  edges=edges,
                  node_features=node_features,
                  edge_features=edge_features,
                  obj_sense=obj_sense)

    return graph

def generate_MVC(N, M):
    '''
    Function Description:
    Generate instances of the minimum vertex cover problem in a general graph.

    Parameters:
    - N: Number of vertices in the graph.
    - M: Number of edges in the graph.

    Return: 
    Relevant parameters of the generated minimum vertex cover problem.
    '''
    
    # n represents the number of decision variables, where each vertex in the graph corresponds to a decision variable.
    # m represents the number of constraints, where each edge in the graph corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraint: randomly generate an edge and impose a constraint that at least one of the vertices connected by the edge must be selected.
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        k[i] = 2
        site[i].append(x)
        value[i].append(1)
        site[i].append(y)
        value[i].append(1)
        constraint[i] = 1
        constraint_type[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a vertex is a random value.
    for i in range(N):
        coefficient[i] = random.random()
        

    node_features = []
    edges = []
    edge_features = []
    
    for i in range(m):
        if constraint_type[i] == 1:
            tmp = copy.deepcopy([0, 0, 0, 1, 0, 0, 0, 0])
        elif constraint_type[i] == 2:
            tmp = copy.deepcopy([0, 0, 0, 0, 0, 1, 0, 0])
        else:
            tmp = copy.deepcopy([0, 0, 0, 0, 1, 0, 0, 0])
        tmp[6] = constraint[i]
        node_features.append(tmp)

    for i in range(n):
        tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0, 0])
        tmp[7] = coefficient[i]
        node_features.append(tmp)

    for i in range(m):
        for j in range(k[i]):
            edges.append((i, site[i][j] + m))
            edge_features.append(value[i][j])

    obj_sense = 0
    
    graph = Graph(num_con=m,
                  num_var=n,
                  edges=edges,
                  node_features=node_features,
                  edge_features=edge_features,
                  obj_sense=obj_sense)

    return graph

def generate_SC(N, M):
    '''
    Function Description:
    Generate instances of the set cover problem, where each item is guaranteed to appear in exactly 3 sets.

    Parameters:
    - N: Number of sets.
    - M: Number of items.

    Return: 
    Relevant parameters of the generated set cover problem.
    '''

    # n represents the number of decision variables, where each set corresponds to a decision variable.
    # m represents the number of constraints, where each item corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraint: At least one of the four sets in which each item appears must be selected.
    for i in range(M):
        vis = {}
        for j in range(3):
            now = random.randint(0, N - 1)
            while(now in vis.keys()):
                now = random.randint(0, N - 1)
            vis[now] = 1

            site[i].append(now)
            value[i].append(1)
        k[i] = 3 
    for i in range(M):
        constraint[i] = 1
        constraint_type[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a set is a random value.
    for i in range(N):
        coefficient[i] = random.random()
    
    node_features = []
    edges = []
    edge_features = []
    
    for i in range(m):
        if constraint_type[i] == 1:
            tmp = copy.deepcopy([0, 0, 0, 1, 0, 0, 0, 0])
        elif constraint_type[i] == 2:
            tmp = copy.deepcopy([0, 0, 0, 0, 0, 1, 0, 0])
        else:
            tmp = copy.deepcopy([0, 0, 0, 0, 1, 0, 0, 0])
        tmp[6] = constraint[i]
        node_features.append(tmp)

    for i in range(n):
        tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0, 0])
        tmp[7] = coefficient[i]
        node_features.append(tmp)

    for i in range(m):
        for j in range(k[i]):
            edges.append((i, site[i][j] + m))
            edge_features.append(value[i][j])

    obj_sense = 0
    
    graph = Graph(num_con=m,
                  num_var=n,
                  edges=edges,
                  node_features=node_features,
                  edge_features=edge_features,
                  obj_sense=obj_sense)

    return graph

def generate_MAXCUT(N, M):
    '''
    函数说明：
    生成一般图当中最大割的问题实例。

    参数说明：
    - N: 图的点数。
    - M: 图的边数。
    '''
    n = N + N * N
    m = 2 * M
    k = []

    #site[i][j]表示第i个约束的第j个决策变量是哪个决策变量
    #value[i][j]表示第i个约束的第j个决策变量的系数
    #constraint[i]表示第i个约束右侧的数
    #constraint_type[i]表示第i个约束的类型，1表示<=，2表示>=
    #coefficient[i]表示第i个决策变量在目标函数中的系数
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    #先将问题转换为将点染成01，一条边连接的两个点若为一个0一个1，则这条边必须选取。
    #添加约束，每次随机生成一条边，设定边的目标函数系数为随机，添加约束使得：
    #1.一条边连接的两个点若为一个0一个1，则这条边必须选取(为1)；
    #2.一条边连接的两个点若为两个0，或两个1，则这条边必不选取(为0)。
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        site[i * 2].append(N + x * N + y)
        value[i * 2].append(1)
        site[i * 2].append(x) 
        value[i * 2].append(-1)
        site[i * 2].append(y) 
        value[i * 2].append(-1)
        constraint[i * 2] = 0
        constraint_type[i * 2] = 1
        k[i * 2] = 3

        site[i * 2 + 1].append(N + x * N + y) 
        value[i * 2 + 1].append(1)
        site[i * 2 + 1].append(x) 
        value[i * 2 + 1].append(1)
        site[i * 2 + 1].append(y) 
        value[i * 2 + 1].append(1)
        constraint[i * 2 + 1] = 2
        constraint_type[i * 2 + 1] = 1
        k[i * 2 + 1] = 3

        if(not(N + x * N + y in coefficient)):
            coefficient[N + x * N + y] = 0
        coefficient[N + x * N + y] += random.random()

    node_features = []
    edges = []
    edge_features = []
    
    for i in range(m):
        if constraint_type[i] == 1:
            tmp = copy.deepcopy([0, 0, 0, 1, 0, 0, 0, 0])
        elif constraint_type[i] == 2:
            tmp = copy.deepcopy([0, 0, 0, 0, 0, 1, 0, 0])
        else:
            tmp = copy.deepcopy([0, 0, 0, 0, 1, 0, 0, 0])
        tmp[6] = constraint[i]
        node_features.append(tmp)

    for i in range(n):
        tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0, 0])
        if i in coefficient.keys():
            tmp[7] = coefficient[i]
        else:
            tmp[7] = 0
        node_features.append(tmp)

    for i in range(m):
        for j in range(k[i]):
            edges.append((i, site[i][j] + m))
            edge_features.append(value[i][j])    
    
    obj_sense = 1

    graph = Graph(num_con=m,
                  num_var=n,
                  edges=edges,
                  node_features=node_features,
                  edge_features=edge_features,
                  obj_sense=obj_sense)

    return graph

def generate_CAT(N, M):
    '''
    Function Description:
    Generate instances of the set cover problem, where each item is guaranteed to appear in exactly 3 sets.

    Parameters:
    - N: Number of sets.
    - M: Number of items.

    Return: 
    Relevant parameters of the generated set cover problem.
    '''
    # n represents the number of decision variables, where each set corresponds to a decision variable.
    # m represents the number of constraints, where each item corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}

    # Add constraints.
    for i in range(M):
        vis = {}
        for j in range(3):
            now = random.randint(0, N - 1)
            while(now in vis.keys()):
                now = random.randint(0, N - 1)
            vis[now] = 1

            site[i].append(now)
            value[i].append(1)
        k[i] = 3   
    for i in range(M):
        constraint[i] = 1
        constraint_type[i] = 1
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a set is a random value.
    for i in range(N):
        coefficient[i] = random.random() * 1000
    
    node_features = []
    edges = []
    edge_features = []
    
    for i in range(m):
        if constraint_type[i] == 1:
            tmp = copy.deepcopy([0, 0, 0, 1, 0, 0, 0, 0])
        elif constraint_type[i] == 2:
            tmp = copy.deepcopy([0, 0, 0, 0, 0, 1, 0, 0])
        else:
            tmp = copy.deepcopy([0, 0, 0, 0, 1, 0, 0, 0])
        tmp[6] = constraint[i]
        node_features.append(tmp)

    for i in range(n):
        tmp = copy.deepcopy([0, 0, 1, 0, 0, 0, 0, 0])
        tmp[7] = coefficient[i]
        node_features.append(tmp)

    for i in range(m):
        for j in range(k[i]):
            edges.append((i, site[i][j] + m))
            edge_features.append(value[i][j])

    obj_sense = 1

    graph = Graph(num_con=m,
                  num_var=n,
                  edges=edges,
                  node_features=node_features,
                  edge_features=edge_features,
                  obj_sense=obj_sense)

    return graph

def generate_samples(
    problem_type : str,
    difficulty_mode : str,
    seed : int, 
    number : int,
    output_dir : str
):
    '''
    Function Description:
    Generate problem instances based on the provided parameters and package the output as data.pickle.

    Parameters:
    - problem_type: Available options are ['IS', 'MVC', 'MAXCUT', 'SC'], representing the maximum independent set problem, minimum vertex cover problem, maximum cut problem, minimum set cover problem, and Meituan flash sale problem, respectively.
    - difficulty_mode: Available options are ['easy', 'medium', 'hard'], representing easy (small-scale), medium (medium-scale), and hard (large-scale) difficulties.
    - seed: Integer value indicating the starting random seed used for problem generation.
    - number: Integer value indicating the number of instances to generate.

    Return: 
    The problem instances are generated and packaged as data.pickle. The function does not have a return value.
    '''
    # Set the random seed.
    random.seed(seed) 

    # Check and create using the os module.
    dir_name = 'example'
    if not os.path.exists(dir_name):  
        os.mkdir(dir_name)

    for i in range(number):
        # Randomly generate instances of the maximum independent set problem and package the output.
        if(problem_type == 'IS'):
            if(difficulty_mode == 'tiny'):
                N = 2500
                M = 7500
            elif(difficulty_mode == 'easy'):
                N = 10000
                M = 30000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 300000
            else:
                N = 1000000
                M = 3000000  
            graph = generate_IS(N, M)
            with open(output_dir + '/IS_' + str(i), 'wb') as f:
                pickle.dump(graph, f)
        
        # Randomly generate instances of the minimum vertex cover problem and package the output.
        if(problem_type == 'MVC'):
            if(difficulty_mode == 'tiny'):
                N = 2500
                M = 7500
            elif(difficulty_mode == 'easy'):
                N = 10000
                M = 30000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 300000
            else:
                N = 1000000
                M = 3000000 
            graph = generate_MVC(N, M)
            with open(output_dir + '/MVC_' + str(i), 'wb') as f:
                pickle.dump(graph, f)
        
        # Randomly generate instances of the minimum set cover problem and package the output.
        if(problem_type == 'SC'):
            if(difficulty_mode == 'tiny'):
                N = 2500
                M = 7500
            elif(difficulty_mode == 'easy'):
                N = 10000
                M = 30000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 300000
            else:
                N = 1000000
                M = 3000000 
            graph = generate_SC(N, M)
            with open(output_dir + '/SC_' + str(i), 'wb') as f:
                pickle.dump(graph, f)
        
        # Randomly generate instances of the combinatorial auction problem and package the output.
        if(problem_type == 'CAT'):
            if(difficulty_mode == 'tiny'):
                N = 2500
                M = 5000
            elif(difficulty_mode == 'easy'):
                N = 10000
                M = 20000
            elif(difficulty_mode == 'medium'):
                N = 100000
                M = 200000
            else:
                N = 1000000
                M = 2000000 
            graph = generate_CAT(N, M)
            with open(output_dir + '/CAT_' + str(i), 'wb') as f:
                pickle.dump(graph, f)

        if(problem_type == 'MAXCUT'):
            if(difficulty_mode == 'tiny'):
                N = 50
                M = 750
            elif(difficulty_mode == 'easy'):
                N = 100
                M = 1250
            elif(difficulty_mode == 'medium'):
                N = 250
                M = 25000
            else:
                N = 600
                M = 150000 
            graph = generate_MAXCUT(N, M)
            with open(output_dir + '/MAXCUT_' + str(i), 'wb') as f:
                pickle.dump(graph, f)    


        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", choices = ['IS', 'MVC', 'SC', 'CAT', 'MAXCUT'], default = 'SC', help = "Problem type selection")
    parser.add_argument("--difficulty_mode", choices = ['tiny', 'easy', 'medium', 'hard'], default = 'easy', help = "Difficulty level.")
    parser.add_argument('--seed', type = int, default = 0, help = 'Random generator seed.')
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    parser.add_argument("--output_dir", type = str, default = "bipartite_graph/4type_problem")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    generate_samples(**vars(args))