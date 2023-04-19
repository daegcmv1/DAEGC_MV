import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft

import tensorly as tl
tl.set_backend('pytorch')
device = torch.device('cuda:0')

class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features, 3)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

       
        self.a_self = nn.Parameter(torch.zeros(size=(out_features * 3, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

       
        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features * 3, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def complex_pro(self, A, B):

        A_ROWS, A_COLS = A.shape
        B_ROWS, B_COLS = B.shape
        A_real, A_imag = A.real, A.imag
        B_real, B_imag = B.real, B.imag

        if A_COLS != B_ROWS:
            print("Cannot be multiplied")

        else:
            res_real = torch.zeros((A_ROWS, B_COLS), dtype=torch.complex128)
            res_imag = torch.zeros((A_ROWS, B_COLS), dtype=torch.complex128)
            res = torch.zeros((A_ROWS, B_COLS), dtype=torch.complex128)
            res_real = torch.mm(A_real, B_real) - torch.mm(A_imag, B_imag)
            res_imag = torch.mm(A_real, B_imag) + torch.mm(A_imag, B_real)

            res = torch.complex(res_real, res_imag)
            return res

    def tt_product(self, input_x, input_w):

        x_new = fft(input_x, dim=2)
        w_new = fft(input_w, dim=2)
        h = torch.zeros(([input_x.shape[0], input_w.shape[1], input_x.shape[2]]), dtype=torch.complex128)

        for i in range(input_x.shape[2]):

            h[:, :, i] = self.complex_pro(x_new[:, :, i], w_new[:, :, i])

        h = ifft(h, dim=2).real
        h = h.to(torch.float32)
        return h

    def forward(self, input, adj, M, concat=True):

        h1 = self.tt_product(input, self.W)  
        h = h1.reshape(input.shape[0], h1.shape[1] * h1.shape[2])  
        h = h.to(device)
        attn_for_self = torch.mm(h, self.a_self)     
        attn_for_neighs = torch.mm(h, self.a_neighs)     
       
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1) 
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)  
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)
        h_prime = h_prime.reshape(h1.shape[0], h1.shape[1], h1.shape[2])
        if concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )
