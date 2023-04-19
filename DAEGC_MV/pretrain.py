import argparse
import itertools

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

import utils
from model import GAT
from evaluation import eva

from torch.fft import fft
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.manifold import TSNE

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pretrain(dataset):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label 
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    x = F.normalize(x, p=2, dim=1)
    max_acc = 0.0003


    #  FFT 

    x_real = fft(x).real
    x_imag = fft(x).imag
    x2 = x_real + x_imag
    x2 = F.normalize(x2, p=2, dim=1)
    x2 = (x2 * 10).to(device)


    a = 1.1
    x_0 = x.cpu().numpy()
    x3 = (1.0 / pow(2, 0.5)) * np.exp((1j) * a * np.pi * x_0)
    x3 = x3.real + x3.imag
    x3 =torch.from_numpy(x3).to(device)
    x3 = F.normalize(x3, p=2, dim=1)

    x = torch.stack((x, x2, x3), dim=2)  

    best_epoch = -1


    for epoch in range(args.max_epoch):
    
        model.train()
        A_pred, z, c, A_pred2 = model(x, adj, M)
    
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1)) + 0.004 * F.binary_cross_entropy(
            A_pred2.view(-1), adj_label.view(-1)) + 0.0001 * torch.norm(A_pred - A_pred2, 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z, _, _ = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)

        if acc > max_acc:
            max_acc = acc
        
            print('Saving the best model!')
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{args.name}_1115.pkl"
            )

    

        '''if epoch % 100 ==0:
            model.load_state_dict(torch.load(f"./pretrain/predaegc_{args.name}_1115.pkl", map_location='cpu'))
            _, z, _, _ = model(x, adj, M)
            z = z.detach().data.cpu().numpy()
            model2 = cluster.KMeans(n_clusters =args.n_clusters, max_iter = 1000)
            model2.fit(z)
            predicted = model2.predict(z)
            data_TSNE = TSNE(n_components=2).fit_transform(z[:,0:12])  
            x1_axis = data_TSNE[:, 0]
            x2_axis = data_TSNE[:, 1]
            plt.xticks([])
            plt.yticks([])
            #plt.title('Epoch: {}'.format(epoch))
            plt.scatter(x1_axis, x2_axis, 10, c=predicted)
          
            plt.savefig('pre_Cora4{}.pdf'.format(epoch))
            plt.show()'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Cora")
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--embedding_size", default=12, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]
    print('dataset:{}'.format(dataset))

    if args.name == "Citeseer":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.0005
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)


