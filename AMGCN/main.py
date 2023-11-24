import random
import torch
import numpy as np
import pandas as pd
from dyg import *
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from stgcn import *
from fusiongraph import *
#Set Random Seed And Device
seed=923
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
print("cuda") if torch.cuda.is_available() else print("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
dataset='la'
#Set File Path
fusion_matrix_path = 'dataset/'+dataset+'_adj.csv'
simi_matrix_path = "dataset/"+dataset+"_simi.csv"
adj_matrix_path = "dataset/"+dataset+"_adj.csv"
keda_matrix_path = "dataset/"+dataset+"_keda.csv"
heu_matrix_path = "dataset/"+dataset+"_heu.csv"
kl_matrix_path = "dataset/"+dataset+"_kl1.csv"
data_path = "dataset/"+dataset+"_speed.csv"
save_path = "save/model.pt"
#Read Graph
Ks, Kt = 3, 3
noise=0
fusion_W = load_matrix(fusion_matrix_path)
#fusion_W = fusion_W+(noise)*np.random.normal(size=(207,207))
simi_W = load_matrix(simi_matrix_path)
adj_W = load_matrix(adj_matrix_path)
keda_W = load_matrix(keda_matrix_path)
heu_W = load_matrix(heu_matrix_path)
kl_W = load_matrix(kl_matrix_path)
fusion_L = torch.tensor(scaled_laplacian(fusion_W)).to(device)
simi_L = torch.tensor(scaled_laplacian(simi_W)).to(device)
adj_L = torch.tensor(scaled_laplacian(adj_W)).to(device)
keda_L = torch.tensor(scaled_laplacian(keda_W)).to(device)
heu_L=torch.tensor(scaled_laplacian(heu_W)).to(device)
kl_L = torch.tensor(scaled_laplacian(kl_W)).to(device)
#fusion_L=fusion_L+(noise**0.5)*(torch.randn(fusion_L.shape).to(device))
simi_L=simi_L+(noise**0.5)*(torch.randn(simi_L.shape).to(device))
adj_L=adj_L+(noise**0.5)*(torch.randn(adj_L.shape).to(device))
keda_L=keda_L+(noise**0.5)*(torch.randn(keda_L.shape).to(device))
heu_L=heu_L+(noise**0.5)*(torch.randn(heu_L.shape).to(device))
kl_L=kl_L+(noise**0.5)*(torch.randn(kl_L.shape).to(device))
# graph_list = [simi_L,adj_L,keda_L,kl_L,heu_L]
# graph_list = [adj_L,keda_L,kl_L,heu_L]
#graph_list = [simi_L,keda_L,kl_L,heu_L]
# graph_list = [simi_L,adj_L,kl_L,heu_L]
graph_list = [simi_L,adj_L,keda_L,heu_L]
# graph_list = [simi_L,adj_L,keda_L,kl_L]
graph = torch.stack(graph_list).to(torch.float).to(device)
print(graph.shape)
L = scaled_laplacian(load_matrix(fusion_matrix_path))
Lk = cheb_poly(L, Ks)
print(L.shape,Lk.shape)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
#Parameters
day_slot = 288
train_model='mixer'
if dataset=='la':
    n_train, n_val, n_test = 95, 10, 10
    n_route = 207
else:
    n_train, n_val, n_test = 145, 18, 10
    n_route = 325
n_his = 12
n_pred = 3

blocks = [[1, 32, 64,12,12],[64,64,64,128,12],[64,64,64]]
drop_prob = 0.01
batch_size = 32
epochs = 50
lr = 5*1e-4
grap=dict(
        use=['dist', 'neighb', 'distri','tempp', 'func'],
        fix_weight=False,
        tempp_diag_zero=True,
        matrix_weight=True,
        distri_type='exp',
        func_type='ours',
        attention=True,
    )
#Standardization
train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)
print(train.shape)
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
#Transform Data
x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)
# noise=0
# x_train=x_train+(noise**0.5)*torch.randn(x_train.shape)
# x_val=x_val+(noise**0.5)*torch.randn(x_val.shape)
# x_test=x_test+(noise**0.5)*torch.randn(x_test.shape)
# y_train=y_train+(noise**0.5)*torch.randn(y_train.shape)
# y_val=y_val+(noise**0.5)*torch.randn(y_val.shape)
# y_test=y_test+(noise**0.5)*torch.randn(y_test.shape)
#DataLoader
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data,1)
#Set Loss Fuunction , Model , Optimizer And LR Scheduler
loss = nn.MSELoss()
#model = STGCN(Ks, Kt, blocks, n_his, n_route, drop_prob,model=train_model).to(device)
model = STGCN2(Ks, Kt, blocks, n_his, n_route, drop_prob,model=train_model).to(device)
graph_fusion_model = FusionGraphModel(graph, device, grap, dict(in_dim=1,out_dim=1,hist_len=24,pred_len=24,type='pm25',),8, 12, 0.1).to(device)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
optimizer = torch.optim.RMSprop([{'params':model.parameters()},{'params':graph_fusion_model.parameters()}],lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
#Train
train_l=[]
val_l=[]
min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        G=graph_fusion_model()
        Gk = torch.stack(cheb_polynomial_torch(G, 3)).to(torch.float).to(device)
        m,adj = model(x, Gk)
#         print(m.shape)
        y_pred = m.permute(0,2,1,3)
#         print(y_pred.shape)
#         print(y.shape)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter,Gk)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    train_l.append(l_sum/n)
    val_l.append(val_loss)
adj=adj.to('cpu').detach().numpy()
G=G.to('cpu').detach().numpy()
np.save('graph_bay.npy', G)
np.save('adj_bay.npy', adj)

#Load Best Model
best_model = STGCN2(Ks, Kt, blocks, n_his, n_route, drop_prob,model=train_model).to(device)
#best_model = STGCN(Ks, Kt, blocks, n_his, n_route, drop_prob,model=train_model).to(device)
best_model.load_state_dict(torch.load(save_path))
#Evaluation
l = evaluate_model(best_model, loss, test_iter,Gk)
MAE, MAPE, RMSE,y_list,y_list_pred = evaluate_metric(best_model, test_iter, scaler,Gk)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)