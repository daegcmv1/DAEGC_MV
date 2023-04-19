import argparse  
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from torch_geometric.datasets import Planetoid

import utils
from model import GAT
from evaluation import eva
from torch.fft import fft

import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.manifold import TSNE
torch.cuda.set_device(-1)

class DAEGC_MV(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC_MV, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A_pred, z, _, A = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q, A

    def get_Q(self, z):
        center = F.normalize(self.cluster_layer, p=2, dim=1)
        
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - center, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def trainer(dataset):
    model = DAEGC_MV(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    data = F.normalize(data, p=2, dim=1)

    #  FFT 
    x_real = fft(data).real
    x_imag = fft(data).imag
    x2 = x_real + x_imag
    x2 = F.normalize(x2, p=2, dim=1)
    x2 = x2 * 10

   
    a = 1.1
    x_0 = data.cpu().numpy()
    x3 = (1.0 / pow(2, 0.5)) * np.exp((1j) * a * np.pi * x_0)
    x3 = x3.real + x3.imag
    x3 =torch.from_numpy(x3).to(device)
    x3 = F.normalize(x3, p=2, dim=1)


    data = torch.stack((data, x2, x3), dim=2)  


    with torch.no_grad():
        _, z, _, _ = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pretrain')
    max_acc = 0.0003
    A_pred, z, Q, A_pred2 = model(data, adj, M)
    for epoch in range(args.max_epoch):
        model.train()

        A_pred, z, q, A_pred2 = model(data, adj, M)
        p = target_distribution(Q.detach())
        q1 = q.detach().data.cpu().numpy().argmax(1)  # Q
        acc, nmi, ari, f1 = eva(y, q1, epoch)

        if acc > max_acc:
            max_acc = acc
            print("Upadate Q !")
            Q = q
            print('Saving the best model!')
            torch.save(
                model.state_dict(), "./pretrain/predaegc1116.pkl")

        q1 = Q.detach().data.cpu().numpy().argmax(1)
        eva(y, q1, epoch)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 1000 * kl_loss + re_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.load_state_dict(torch.load("./pretrain/predaegc1116.pkl", map_location='cpu'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)   #learning rate
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--embedding_size', default=12, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]


    if args.name == 'Citeseer':
      args.lr = 0.0001
      args.k = None
      args.n_clusters = 6
    elif args.name == 'Cora':
      args.lr = 0.02
      args.k = None
      args.n_clusters = 7
    elif args.name == "Pubmed":
      args.lr = 0.001
      args.k = None
      args.n_clusters = 3
    else:
      args.k = None

    args.pretrain_path = f"./pretrain/predaegc_{args.name}_1115.pkl"
    args.input_dim = dataset.num_features

    print(args)
    trainer(dataset)

