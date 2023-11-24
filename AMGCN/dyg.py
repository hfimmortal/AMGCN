from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import math
class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho
class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c):
        super(spatio_conv_layer, self).__init__()
        #self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)
class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x
class mixers(nn.Module):
    def __init__(self, kt,c_in,hidden,c_out,s,Shared_parameter):  # s:要分解成几个时间序列
        super(mixers, self).__init__()
        self.kt = kt
        self.c_in = c_in
        self.c_out = c_out
        self.s = s
        self.Shared_parameter = Shared_parameter
        self.activation = nn.GELU()
        if Shared_parameter == True:
            self.mlp1 = torch.nn.Linear(c_in, hidden)
            self.mlp2 = torch.nn.Linear(hidden, c_out)
        else:
            self.mlp1 = torch.nn.Linear(c_in, hidden*s)
            self.mlp2 = torch.nn.Linear(hidden*s, c_out*s)


    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = torch.stack([x[:, :, :, i::self.s] for i in range(self.s)], dim=3)
        if self.Shared_parameter == True:
            x = self.mlp1(x)
            x = self.activation(x)
            x = self.mlp2(x)
            x = x.flatten(3)
        else:
            x = self.mlp1(x)
            x = self.activation(x)
            x = self.mlp2(x)
            x = torch.stack([x[:,:,:,i,i*self.c_out:(i+1)*self.c_out] for i in range(self.s)],dim=3)
            x = x.flatten(3)
        x=x.permute(0,1,3,2)
        return x


class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1,padding=(1,0))
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1,padding=(1,0))

    def forward(self, x):
        x_in = self.align(x)
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)
class st_conv_block2(nn.Module):
    def __init__(self,c,num_nodes,dropout,kt,device,model):
        super(st_conv_block2, self).__init__()
        gcn_depth=2
        self.gc=graph_constructor(num_nodes, 20, 40, device, alpha=3, static_feat=None)
        self.gcn1 = mixprop(c[1], c[1], gcn_depth, 0.1, 0.05)
        self.gcn2 = mixprop(c[1], c[1], gcn_depth, 0.1, 0.05)
        self.sconv = spatio_conv_layer(3, c[1])
        self.ln = nn.LayerNorm([num_nodes, c[2]])
        self.dropout = nn.Dropout(dropout)
        self.idx = torch.arange(num_nodes).to(device)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2], 'sigmoid')
        self.model = model
        if self.model == 'mixer':
            self.mixer = mixers(kt, int(c[3]/2), 256, int(c[4]/2), 2, True)
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
    def forward(self,x,Lk):
        if self.model == 'mixer':
            x= self.mixer(x)
        x = self.tconv1(x)
        adj = self.gc(self.idx)
        x_in = x.permute(0,1,3,2)
        x_m = self.gcn1(x_in, adj) + self.gcn2(x_in, adj.transpose(1, 0))
        x_s = self.sconv(x, Lk)
        x_m=x_m.permute(0,1,3,2)
        x=torch.sigmoid(x_m)+torch.tanh(x_s)
        del x_s,x_m,x_in
        x = self.tconv2(x)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x),adj


class fully_conv_layer(nn.Module):
    def __init__(self, c, model):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)
        s = 12
        self.conv2 = nn.Conv2d(s, 3, 1)

    def forward(self, x):
        x = self.conv(x).permute(0, 2, 1, 3)
        return self.conv2(x)


class output_layer(nn.Module):
    def __init__(self, c, T, n, model):
        super(output_layer, self).__init__()
        # self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        #
        # self.ln = nn.LayerNorm([n, c])
        # self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        # self.c = c
        self.fc = fully_conv_layer(c, model)

    def forward(self, x):
        # print(self.c)
        # x_t1 = self.tconv1(x)
        # x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x_t2 = self.tconv2(x_ln)
        # print('11',x_t2.shape)
        return self.fc(x)


class STGCN2(nn.Module):
    def __init__(self, ks, kt, bs, T, n, p, model='tem_conv'):
        super(STGCN2, self).__init__()
        self.st_conv1 = st_conv_block2(bs[0],n,p,kt,'cuda:0',model=model)
        #self.st_conv2 = st_conv_block2(bs[1],n,p,kt,'cuda:0',model=model)
        # self.st_conv3 = st_conv_block(ks, kt, n, bs[2], p, model)

        self.output = output_layer(64, T - 4 * (kt - 1), n, model)

    def forward(self, x, Lk):
        x,adj = self.st_conv1(x, Lk)
        #x = self.st_conv2(x, Lk)
        # x_st3 = self.st_conv3(x_st2, Lk)
        # print('x_st3',x_st3.shape)
        return self.output(x),adj
# x=torch.Tensor(20,32,12,207).to(device)
# Lk=torch.Tensor(3,207,207).to(device)
# #scb=st_conv_block2(32,207,device).to(device)
# #y=scb(x)
# #print(y.shape)
# num_nodes=207
# device='cuda'
# gc = graph_constructor(num_nodes, 20, 40, device, alpha=3, static_feat=None).to(device)
# idx = torch.arange(num_nodes).to(device)
# a=gc(idx)
# conv_channels=32
# residual_channels=32
# gcn_depth=2
# x=torch.Tensor(20,1,12,207).to(device)
# gcn1=mixprop(conv_channels, residual_channels, gcn_depth, 0.1, 0.05).to(device)
# gcn2=mixprop(conv_channels, residual_channels, gcn_depth, 0.1, 0.05).to(device)
# #x = gcn1(x, a)+gcn2(x, a.transpose(1,0))
# #print(x.shape)
# Ks,Kt=3,3
# n_his = 12
# n_pred = 3
# n_route = 207
# blocks = [[1, 32, 64],[64,64,64],[64,64,64]]
# drop_prob = 0.01
# train_model='tem_conv'
# model=STGCN(Ks, Kt, blocks, n_his, n_route, drop_prob,model=train_model).to(device)
# y=model(x,Lk)
# print(y.shape)
