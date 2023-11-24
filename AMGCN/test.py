import numpy as np
import torch
from torch.nn.functional import interpolate
import torch.nn as nn
import pandas as pd
# a=torch.tensor([0,1,2,3,4,5,6,7,8,9],dtype=float)
# b=torch.stack([a,a,a,a,a,a])
# a=torch.stack([b,b,b,b])
# print(a.shape,a.dtype)
#
# c=torch.nn.Linear(10,20,dtype=float)
# e=c(a)
# print(e.shape)
# n=2
# c=torch.cat([a[:,:,i::n] for i in range(n)],dim=0)
# print(c.shape)
# a=torch.Tensor(20,1,12,207)
#
# dataset='la'
#
# heu_matrix_path = "dataset/"+dataset+"_Heuristic_graph.csv"
# kl_matrix_path = "dataset/"+dataset+"_kl.csv"
# heu_W = pd.read_csv(heu_matrix_path, header=None).values.astype(float)
# heu_W = np.array(heu_W)
# def scaled_laplacian(A):
#     n = A.shape[0]
#     d = np.sum(A, axis=1)
#     L = np.diag(d) - A
#     for i in range(n):
#         for j in range(n):
#             if d[i] > 0 and d[j] > 0:
#                 L[i, j] /= np.sqrt(d[i] * d[j])
#     lam = np.linalg.eigvals(L).max().real
#     return 2 * L / lam - np.eye(n)
# heu_W=scaled_laplacian(heu_W)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train=torch.Tensor(27360,207)
train = scaler.fit_transform(train)
y=torch.Tensor(20,3,1,207)
y_pred=torch.Tensor(20,3,1,207)
y=y.view(-1,207)
y_pred=y_pred.view(-1,207)
y_l = scaler.inverse_transform(y.cpu().numpy())
y=y_l.reshape(-1)
y_pred_l = scaler.inverse_transform(y_pred.cpu().numpy())
y_pred=y_pred_l.reshape(-1)
x=np.random.uniform(0,1,[3,4,5,6])
np.save('111.npy',x)
y=np.load('metrresult.npy',allow_pickle=True)
print(y[0].shape)
print(y.shape)





# a=a.permute(0,3,1,2)
# print(a.shape)
# n=3
# a=torch.cat([a[:,:,:,i::n] for i in range(n)],dim=2)
# print(a.shape)
# c=torch.nn.Linear(4,256)
# d=torch.nn.Linear(256,4)
# a=c(a)
# a=a.flatten(2)
# print(a.shape)


