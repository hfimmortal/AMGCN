import torch
import numpy as np


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)
def cheb_polynomial_torch(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

def evaluate_model(model, loss, data_iter,Lk):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x,Lk)[0].permute(0,2,1,3)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler,Lk):
    model.eval()
    y_list,y_pred_list=[],[]
    with torch.no_grad():
        mae, mape, mse = [], [], []
        mae_c, mape_c, mse_c = [], [], []
#        for x, y in data_iter:
        for i, data in enumerate(data_iter):
            x,y=data
            y_l = scaler.inverse_transform(y.view(-1,207).cpu().numpy())
            y=y_l.reshape(-1)
            y_pred_l = scaler.inverse_transform(model(x,Lk)[0].permute(0,2,1,3).view(-1,207).cpu().numpy())
            y_pred=y_pred_l.reshape(-1)
            y_list.append(y_l)
            y_pred_list.append(y_pred_l)
            d = np.abs(y - y_pred)
            # if((i%288>6 and i%288<9)or(i%288>17 and i%288<20)):
            #     mae_c += d.tolist()
            #     mape_c += (d / y).tolist()
            #     mse_c += (d ** 2).tolist()
            # else:
            #     mae += d.tolist()
            #     mape += (d / y).tolist()
            #     mse += (d ** 2).tolist()
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
                
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        # MAE_C = np.array(mae_c).mean()
        # MAPE_C = np.array(mape_c).mean()
        # RMSE_C = np.sqrt(np.array(mse_c).mean())
        y_numpy = np.array(y_list)
        y_pred_numpy = np.array(y_pred_list)
        np.save('bayreal.npy', y_numpy)
        np.save('bayresult.npy', y_pred_numpy)
        
        return MAE, MAPE, RMSE,y_list,y_pred_list
