##We have a set of brain functional connectivity networks
##We want to represent these networks in a lower dimesinal latent space 

##The program outputs representations of input networks in latent space

##Graph Autoencoder
##inputs: Adjacency matrices corresponding to brain networks, correlation matrices consisting of correlations between brain regions
##Ajacency matrices are encoded   
##Correlation matrices are used as node features
##The program learns how to encode adjacency matrices 
import numpy as np
import torch
import os
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.utils.data import Subset
from sklearn.model_selection import KFold

##Prepare input data

##Create dataset consisting of adjaceency matrices and correlation matrices
#"matix" is a folder where adjacency matrices and correlation matrices are stored as textfiles
folder_path = 'matrix'
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
l=len(file_list)
#Create file list sorted by file names
file_list = sorted(file_list)

##Create datalist from the dataset
##The datalist contains information of networks (node features, edge_index) as Pytorch tensors
datalist = []
for i in range(int(l/2)):
  #file path to adjacency matrices
  adj_file_path = file_list[i]
  #file path to correlation matrices
  corr_file_path = file_list[int(l/2+i)]
  #get node features as Pytorch tensors
  corr_mat = np.loadtxt(corr_file_path)
  x = torch.tensor(corr_mat).float()
  #get adjacency matrices as NumPy vectors
  adj_mat = np.loadtxt(adj_file_path)

  #create edge_index (a Pytorch tensor representing edges in terms of node pairs)
  edges = []
  #create a list of edges as pairs of node indices
  for i in range(len(adj_mat)):
    for j in range(len(adj_mat)):
        if i == j:
            continue
        elif adj_mat[i][j] == 1:
            edges.append([i, j])
  edges = torch.tensor(edges, dtype=torch.int).T

  #create data class representing networks
  datalist.append(Data(x=x, edge_index=edges))

##Define a dataset class called network 
#Graph data can be saved in a format that can be loaded efficiently from memory.
class network(InMemoryDataset):
    def __init__(self, transform = None):
        #initialise superclass (InMemoryDataset)
        super().__init__('.', transform)
        #load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])

    #names of raw data files
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    #processed filed will be saved in file 'data.pt'
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = datalist

        #If pre_filter function is defined, filter out unwanted data
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        #If pre_tansform function is defined, transform each data in data_list
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

#Define dataset
#the features of each graph in the dataset are normalised
dataset = network(transform=T.NormalizeFeatures())

##Train and Evaluate model

##Define Encoder (two-layer GCN)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) 
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

##Set parameters
#the dimension of latent space is set to be 13
out_channels = 13
#input dimention is number of features (In this case, 39(#nodes))
num_features = dataset.num_features

# model
model = GAE(GCNEncoder(num_features, out_channels))

##move to GPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loader =  DataLoader(dataset)
for data in loader:
    data.to(device)

##Create optimizer
optimizer = Adam(model.parameters(), lr=0.01)

##K-fold Cross Validation
# Define the number of epochs
num_epochs = 500
# Define the number of folds for cross-validation
num_folds = 5

##Implement cross validation
kf = KFold(n_splits=num_folds, shuffle=True)
with open("./loss/track_loss_epoch500.txt", "a") as f:
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

        # Create subsets for training and validation
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Train the model for the current fold
        for epoch in range(num_epochs):
            model.train()
            for graph in train_loader:
                data_list = graph.to_data_list()
                for data in data_list:
                    train_neg_edge_index = negative_sampling(data.edge_index).to(device)
                    data = train_test_split_edges(data)
                    optimizer.zero_grad()
                    train_pos_edge_index = data.train_pos_edge_index.to(device)
                    x=data.x.to(device)
                    z = model.encode(x, train_pos_edge_index)
                    loss = model.recon_loss(z=z, pos_edge_index=train_pos_edge_index, neg_edge_index=train_neg_edge_index)
                    loss.backward()
                    optimizer.step()

            # Evaluate the model on the test set for the current fold
            model.eval()
            with torch.no_grad():
                for graph in test_loader:
                    data_list = graph.to_data_list()
                    for data in data_list:
                        test_neg_edge_index = negative_sampling(data.edge_index).to(device)
                        data = train_test_split_edges(data)
                        test_pos_edge_index = data.train_pos_edge_index.to(device)
                        x=data.x.to(device)
                        z = model.encode(x, test_pos_edge_index)
                        loss = model.recon_loss(z, pos_edge_index=test_pos_edge_index, neg_edge_index=test_neg_edge_index)
                        #(auc, ap) = model.test(z, pos_edge_index=test_pos_edge_index, neg_edge_index=test_neg_edge_index)

    #loss, auc, ap, fold, epoch        
    val_string = f"{loss:.4f} {fold+1} {epoch+1} "
    f.write(val_string+"\n")

##Make notes on loss and accuracy
val_string = f"TestLoss: {loss:.4f}"
with open("./loss/loss_epoch500.txt", "w") as file:
    file.write(val_string)

##Get and save latent space representation
#They are saved in the folder "latent500"
loader = DataLoader(dataset, batch_size=1, shuffle=False)
for idx, graph in enumerate(loader):
      data_list = graph.to_data_list()
      for data in data_list:
        Z = model.encode(data.x, data.edge_index)
        #Graph representation
        Z_rep = torch.zeros(13)
        for i in range(39):
          Z_rep = Z_rep+Z[i]
        np.savetxt(f'./latent500/Z_{idx}_500.txt', Z_rep.detach().numpy())






