import torch.nn 
from torch_geometric.nn import NNConv, GCNConv
import torch.nn.functional as F
import torch_geometric

class BiGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, output_dim):
        super(BiGNN, self).__init__()

        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, num_node_features * hidden_channels))
        self.conv1 = NNConv(num_node_features, hidden_channels, nn1, aggr='mean')
        nn2 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, hidden_channels * output_dim))
        self.conv2 = NNConv(hidden_channels, output_dim, nn2, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x
    
class BiGNN_large(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, output_dim):
        super(BiGNN_large, self).__init__()

        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, num_node_features * hidden_channels))
        self.conv1 = NNConv(num_node_features, hidden_channels, nn1, aggr='mean')
        
        nn2 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, hidden_channels * hidden_channels))
        self.conv2 = NNConv(hidden_channels, hidden_channels, nn2, aggr='mean')
        
        nn3 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, hidden_channels * output_dim))
        self.conv3 = NNConv(hidden_channels, output_dim, nn3, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, output_dim):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
    
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, emb_size):
        super().__init__("add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class GNNPolicy(torch.nn.Module):
    def __init__(self, emb_size = 64, cons_nfeats = 3, edge_nfeats = 1, var_nfeats = 5, output_dim = 16, with_bias = True):
        super().__init__()

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        self.con_output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_dim, bias=with_bias),
            #torch.nn.Sigmoid()
        )
        self.var_output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, output_dim, bias=with_bias),
            #torch.nn.Sigmoid()
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        for i in range(3):
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )
        # A final MLP on the variable features
        pred_con = self.con_output_module(constraint_features).squeeze(-1)
        pred_var = self.var_output_module(variable_features).squeeze(-1)
        return pred_con, pred_var

class GNNPolicy_with_MLP(torch.nn.Module):
    def __init__(self, emb_size = 64, cons_nfeats = 3, edge_nfeats = 1, var_nfeats = 5, output_dim = 16):
        super().__init__()

        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        self.fc1 = torch.nn.Linear(2*emb_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)    
            
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(
        self, constraint_features, variable_features,
        edge_indices, edge_features,
        node_type, n1_list, n2_list
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        for i in range(3):
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )

        if node_type == 0: # contraint node
            vec1 = torch.index_select(constraint_features, 0, n1_list)
            vec2 = torch.index_select(constraint_features, 0, n2_list)
        else: # variable node
            vec1 = torch.index_select(variable_features, 0, n1_list)
            vec2 = torch.index_select(variable_features, 0, n2_list)
        x = torch.cat([vec1, vec2], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))

        return x
    def get_embedding(
        self, constraint_features, variable_features,
        edge_indices, edge_features,
        node_type, n_list
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        for i in range(3):
            constraint_features = self.conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )

        if node_type == 0: # contraint node
            vec1 = torch.index_select(constraint_features, 0, n_list)
        else: # variable node
            vec1 = torch.index_select(variable_features, 0, n_list)
        return vec1