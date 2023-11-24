import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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

class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p,model='tem_conv'):
        super(st_conv_block, self).__init__()
        self.sconv = spatio_conv_layer(ks, c[1])
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2],'sigmoid')
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)
        self.model = model
        if self.model == 'mixer':
            self.mixer = mixers(kt,int(c[3]/2),256,int(c[4]/2),2,True)
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")

    def forward(self, x, Lk):
        if self.model == 'mixer':
            x= self.mixer(x)
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1, Lk)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)

class fully_conv_layer(nn.Module):
    def __init__(self, c,model):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)
        s=12
        self.conv2 = nn.Conv2d(s, 3, 1)
    def forward(self, x):
        x=self.conv(x).permute(0, 2, 1, 3)
        return self.conv2(x)

class output_layer(nn.Module):
    def __init__(self, c, T, n,model):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
       
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.c=c
        self.fc = fully_conv_layer(c,model)

    def forward(self, x):
        # print(self.c)
        # x_t1 = self.tconv1(x)
        # x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x_t2 = self.tconv2(x_ln)
        # print('11',x_t2.shape)
        return self.fc(x)

class STGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, p,model='tem_conv'):
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p,model)
        #self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, model)
        #self.st_conv3 = st_conv_block(ks, kt, n, bs[2], p, model)
        
        self.output = output_layer(64, T - 4 * (kt-1 ), n,model)

    def forward(self, x, Lk):
        x_st1 = self.st_conv1(x, Lk)
        #x_st2 = self.st_conv2(x_st1, Lk)
        #x_st3 = self.st_conv3(x_st2, Lk)
        #print('x_st3',x_st3.shape)
        return self.output(x_st1)
