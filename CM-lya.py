import numpy as np
import matplotlib.pyplot as plt

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdiffeq import odeint
from torchdyn import *
from torchdyn.numerics import odeint_hybrid
import torch
import torch.utils.data as data
from pytorch_lightning.loggers import TensorBoardLogger
import time
from torch.autograd.functional import jvp
import torch.nn as nn
import pytorch_lightning as pl

import os

device=torch.device("cpu")

d = ToyDataset()
X, yn = d.generate(n_samples=512, noise=1e-2, dataset_type='spheres', dim=2)
import matplotlib.pyplot as plt

colors = ['orange', 'blue'];
fig = plt.figure(figsize=(3,3));
ax = fig.add_subplot(111);
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], s=1, color=colors[yn[i].int()]);

plt.show()

device = torch.device("cpu") # all of this works in GPU as well :)

X_train = torch.Tensor(X).to(device)
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

import torch.nn as nn
import pytorch_lightning as pl

class Net(nn.Module):
    def __init__(self, t_f):
        super().__init__()
        self.f = nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 3)
    )
        # self.t1 = t_f
        self.t1 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # self.n_steps = 100
        self.dt = 0.01

    def step_size_func(self, t0, t1):
        """Computes a variable step size based on the current value of t1."""
        dt = t1 / self.n_steps
        dt = self.dt
        self.n_steps = torch.ceil(t1 / dt).int() - 1
        return self.n_steps

    def dyn(self, x):
        y0 = x.view(-1, 3)
        odefunc = lambda t, y: self.f(y)
        return odefunc

    def dot(self, x):
        return self.f(x)

    def forward(self, x):
        y0 = x.view(-1, 3)
        odefunc = self.dyn(x)
        t0, t1 = nn.Parameter(torch.tensor([0.0]), requires_grad=False), self.t1
        # t_seq = torch.linspace(t0.item(), t1.item(), self.dt)
        t_seq = torch.arange(t0.item(), t1.item() - 1e-6, self.dt)
        t_seq = torch.cat((t_seq, t1.view(1)), dim=0)
        # if t_seq[-1] < t1:
        #     t_seq = torch.cat((t_seq, t1.view(1)), dim=0)
        # t_seq = torch.cat((t0, t0 + torch.cumsum(self.step_size_func(t0, t1), dim=0)))
        y = odeint(odefunc, y0, t_seq, method='rk4')
        return t_seq, y

class Net1(nn.Module):
    def __init__(self, t_f):
        super().__init__()
        self.f = nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 3)
    )
        self.t1 = t_f
        # self.t1 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # self.n_steps = 100
        self.dt = 0.01

    def step_size_func(self, t0, t1):
        """Computes a variable step size based on the current value of t1."""
        dt = t1 / self.n_steps
        dt = self.dt
        self.n_steps = torch.ceil(t1 / dt).int() - 1
        return self.n_steps

    def dyn(self, x):
        y0 = x.view(-1, 3)
        odefunc = lambda t, y: self.f(y)
        return odefunc

    def dot(self, x):
        return self.f(x)

    def forward(self, x):
        y0 = x.view(-1, 3)
        odefunc = self.dyn(x)
        t0, t1 = nn.Parameter(torch.tensor([0.0]), requires_grad=False), self.t1
        t_seq = torch.arange(t0.item(), t1.item(), self.dt)
        # t_seq = torch.linspace(t0.item(), t1.item(), self.dt)
        # if t_seq[-1] < t1:
        #     t_seq = torch.cat((t_seq, t1.view(1)), dim=0)
        # t_seq = torch.cat((t0, t0 + torch.cumsum(self.step_size_func(t0, t1), dim=0)))
        y = odeint(odefunc, y0, t_seq, method='rk4')
        return t_seq, y


class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model= model
        self.lr = 0.03
        self.order = 1

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y_hat, y):
        y_in  = y[None,:].expand(y_hat.size(0),-1).flatten(0,1)
        test = nn.functional.cross_entropy(y_hat[:, :, :-1].flatten(0,1), y_in, reduction='none')
        def v_ndot(order: int, oc_in):
            assert isinstance(order, int) and order >= 0, \
                f"[ERROR] Order({order}) must be non-negative integer."
            if order == 0:
                return nn.functional.cross_entropy(oc_in[:, :, :-1].flatten(0,1), y_in, reduction='none')
            elif order == 1:
                return jvp(func=lambda oc_in: v_ndot(0, oc_in),
                           inputs=oc_in,
                           v=self.model.dot(oc_in),
                           # dyn_fun.eval_dot(t_sample, tuple(oc_in), x_in),
                           create_graph=True)

            else:
                returns = tuple()
                for i in range(1, order):
                    returns += v_ndot(i, *oc_in)
                returns += (jvp(func=lambda *x: v_ndot(order - 1, *x)[-1],
                                inputs=tuple(oc_in),
                                v=self.model.dyn_fun.eval_dot(tuple(oc_in), x_in),
                                create_graph=True)[-1],)
                return returns

        # t_samples, h_sample_in = self.make_samples(x, y, x_in, y_in, batch_size)
        if self.order == 0:
            raise NotImplementedError('[TODO] Implement this.')
        elif self.order == 1:
            v, vdot = v_ndot(1, y_hat)
            violations = torch.relu(vdot + 20. * v.detach())
        elif self.order == 2:
            v, vdot, vddot = v_ndot(2, x)
            violations = torch.relu(vddot + 20 * vdot + 100 * v.detach())
        elif self.order == 3:
            v, vdot, vddot, vdddot = v_ndot(3, x)
            violations = torch.relu(vdddot + 1000 * vddot + 300 * vdot + 30 * v)
        else:
            raise NotImplementedError("[ERROR] Invalid lyapunov order.")
        violation_mask = violations > 0
        effective_batch_size = (violation_mask).sum()
        nominal_batch_size = y_in.shape[0]

        loss = violations.mean()
        x_in = None
        y_in = None
        return loss

    # def aug_loss(self, y_hat, y):
    #     # for i in range(y_hat.size(0)):
    #     #     sub_data = y_hat[i]
    #     # y = y.expand_as(y_hat)
    #     y_ = y.unsqueeze(0).expand(6,-1)
    #     y_hat = y_hat[:][:,:,:-1]
    #     # y_ = y_.unsqueeze(-1)
    #     # y_ = y.unsqueeze(0).expand_as(y_hat[:,:,0])
    #     loss = nn.CrossEntropyLoss(reduction='mean')(y_hat.permute(0,2,1), y_)
    #     return loss


    def training_step(self, batch, batch_idx):
        x, y = batch
        controller = torch.zeros(len(X), 1)
        x = torch.cat((x,controller),1)
        t_eval, y_hat = self.model(x)
        # print(t_eval[-1])
        loss = self.compute_loss(x, y_hat, y)
        # loss = self.aug_loss(y_hat, y)
        # y_hat_ = y_hat[:][:,:,:-1] # select last point of solution trajectory
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': 'my_loss'
        }

    def train_dataloader(self):
        return trainloader


def train_with_increasing_tf_for_one_epoch(model, trainloader, initial_tf, max_tf, tf_increment, threshold):
    tf = initial_tf
    cross_entropy_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    while tf <= max_tf:
        model.model.t1 = torch.tensor(tf)  # 更新模型的 tf 值
        one_epoch_trainer = pl.Trainer(max_epochs=1)
        one_epoch_trainer.fit(model, trainloader)  # 用新的 tf 值重新训练模型

        model.eval()  # 将模型设置为评估模式
        total_ce = 0
        total_samples = 0

        with torch.no_grad():  # 在不计算梯度的情况下执行
            for batch in trainloader:
                inputs, labels = batch
                controller = torch.zeros(len(inputs), 1)
                inputs = torch.cat((inputs, controller), 1)
                t_eval, outputs = model(inputs)  # 获取模型对当前批次的预测
                loss = cross_entropy_loss(outputs[-1][:,:-1], labels)
                total_ce += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        average_ce = total_ce / total_samples  # 计算平均交叉熵
        # print(average_ce)
        if average_ce < threshold:
            break  # 如果交叉熵低于阈值，停止训练

        tf += tf_increment  # 增加 tf 值
        print(tf)



f = nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 3)
    )

#
logger = TensorBoardLogger('tb_logs', name='augmented_lya_t_f')
# #
t_f = torch.tensor(0.05)
fullNet = Net1(t_f)
learn = Learner(fullNet)
trainer = pl.Trainer(min_epochs=300, max_epochs=800, logger=logger)
start = time.time()
trainer.fit(learn)
end = time.time()
print(end-start)
# torch.save(fullNet.state_dict(), 'augmented_CM_lya_20_rk4_greedy.pt')
start = time.time()

train_with_increasing_tf_for_one_epoch(learn, trainloader, initial_tf=0.05, max_tf=1.0, tf_increment=0.01, threshold=0.01)
end = time.time()
print(end-start)
torch.save(fullNet.state_dict(), 'augmented_CM_lya_20_rk4_greedy_.pt')

# fullNet.load_state_dict(torch.load('augmented_CM_lya_20_rk4_greedy.pt'))

# Net.load_state_dict(torch.load('augmented_CM_lya_20_rk4.pt'))

# model.load_state_dict(torch.load('CA.pt'))
# controller = torch.zeros(len(X_train), 1)
# Z_train = torch.cat((X_train,controller),1)
# start = time.perf_counter()
# t_eval, trajectory = Net(Z_train)
# end = time.perf_counter()
# print(end-start)
# trajectory = trajectory.detach().cpu()
# trajectory = trajectory[:,:,:-1]
# print(nn.CrossEntropyLoss()(trajectory[-1, :, :], yn.long()))
# fig = plt.figure(figsize=(10,2));
# ax0 = fig.add_subplot(121);
# ax1 = fig.add_subplot(122);
# for i in range(500):
#     ax0.plot(t_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
#     ax1.plot(t_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
#
# ax0.set_xlabel(r"$t$ [Depth]") ; ax0.set_ylabel(r"$h_0(t)$");
# ax1.set_xlabel(r"$t$ [Depth]") ; ax1.set_ylabel(r"$z_1(t)$");
# ax0.set_title("Dimension 0") ; ax1.set_title("Dimension 1");
#
# colors = ['orange', 'blue'];
# fig = plt.figure(figsize=(3,3));
# ax = fig.add_subplot(111);

# colors = ['orange', 'blue'];
# fig = plt.figure(figsize=(5,4));
# ax = fig.add_subplot(111);

# j=49
# for i in range(500):
#   ax.scatter(trajectory[j,i,0], trajectory[j,i,1], s=1, color=colors[yn[i].int()]);
#   a = trajectory[j,i,:].unsqueeze(0)
#   b = yn[i].unsqueeze(0).long()
#   print(nn.CrossEntropyLoss()(trajectory[j,i,:].unsqueeze(0), yn[i].unsqueeze(0).long()))
# #
# plt.show()
# print(trajectory[0,1,0], trajectory[0,1,1])
# print(trajectory[49,1,0], trajectory[49,1,1])

# n_pts = 10
# x = torch.linspace(trajectory[:,:,0].min(), trajectory[:,:,0].max(), n_pts)
# y = torch.linspace(trajectory[:,:,1].min(), trajectory[:,:,1].max(), n_pts)
# XX, YY = torch.meshgrid(x, y)
# z = torch.cat([XX.reshape(-1,1), YY.reshape(-1,1)], 1)
# t_eval_f, trajectory_f = prob(z, t_span)
#
# derivative = trajectory_f
# dt = t_eval[1] - t_eval[0]
# for i in range(len(t_eval)-1):
#     derivative[i] = (trajectory_f[i+1, :, :] - trajectory_f[i, :, :]) / dt
#
# derivative = derivative.detach().numpy()
# z = z.detach().numpy()

# try:
#   os.mkdir(os.path.join(nb_path, "anelli_concentrici"))
# except Exception as e:
#   print("Folder already exists")

# plt.ioff()
# colors = ['orange', 'blue']
# for j in range(len(t_span)):
#   fig = plt.figure(figsize=(5,5))
#   ax = fig.add_subplot(111)
#   for i in range(len(z)):
#     ax.quiver(z[i, 0], z[i, 1], derivative[j, i, 0], derivative[j, i, 1], color='black', alpha = 0.2)
#   for i in range(len(X)):
#     ax.scatter(trajectory[j,i,0], trajectory[j,i,1], s=30, color=colors[yn[i].int()])
#   # fig.savefig(os.path.join(nb_path, "anelli_concentrici", "anelli_concentrici_" + str(j)) , dpi=150)
#   plt.show()
#   # plt.close(fig)
import torch.nn.functional as tf
def plot_decision_boundary(model, X_train, xlim=[-1, 1], ylim=[-1, 1],  n_grid=200, n_points=512, device=torch.device("cpu")):
    # x0, _ = sample_annuli(n_samples=n_points)
    model = model.to(device)
    xx, yy = torch.linspace(xlim[0], xlim[1], n_grid), torch.linspace(ylim[0], ylim[1], n_grid)
    X, Y = torch.meshgrid(xx, yy) ; z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1).to(device)
    controller = torch.zeros(len(z), 1)
    z = torch.cat((z, controller), 1)
    # _, D = model(z,t_span = torch.linspace(0,1,50))
    _, D = model(z)
    D = D[-1][:,:-1]
    D = tf.softmax(D)
    D = D[:,1]
    D = D.detach().numpy().reshape(n_grid, n_grid)
    D[D > 1], D[D < 0] = 1, 0
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.contourf(X, Y, D, 100, cmap="inferno_r")
    ax.scatter(X_train[:, 0], X_train[:, 1], s=1);
    # ax.scatter(x0[:, 0], x0[:,1], color='lime', alpha=.5, s=20)
    ax.set_xlim(X.min(), X.max()) ; ax.set_ylim(Y.min(), Y.max())
    plt.show()
    return model

# plot_decision_boundary(Net, X_train)

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch.nn.functional as tf
def dec_bound(model, x):
    P = [p for p in model[-1].parameters()]
    w1, w2, b = P[0][0][0].cpu().detach(), P[0][0][1].cpu().detach(), P[1][0].cpu().detach().item()
    return (-w1*x - b + .5)/w2


# x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2)
# x0 = torch.Tensor(x0).to(device)
# ys = torch.LongTensor(ys.long()).to(device)

import pickle

# 假设 X, yn 是您的数据
# X, yn = d.generate(n_samples=512, noise=1e-2, dataset_type='spheres', dim=2)

# 保存数据
# with open('data.pkl', 'wb') as f:
#     pickle.dump((x0, ys), f)

with open('data.pkl', 'rb') as f:
    x0, ys = pickle.load(f)

def plot_traj(model, x0, ys, str, device=torch.device("cpu")):
    # x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2);
    x0 = torch.Tensor(x0).to(device)
    ys = torch.LongTensor(ys.long()).to(device)
    s = torch.linspace(0, 1, 50)
    controller = torch.zeros(len(x0), 1)
    x0 = torch.cat((x0, controller), 1)

    plt.figure(figsize=(8, 8))
    model = model.cpu();
    # t_eval, xS = model(x0, s)
    t_eval, xS = model(x0)
    xS = xS.detach().cpu()
    # xS = model[0].trajectory(x0, s).detach() ;
    model = model.to(device)
    r = 1.05*torch.linspace(xS[:,:,-2].min(), xS[:,:,-2].max(), 2)
    pS = torch.cat([tf.softmax(xS[:,i,-3:-1].to(device))[:,1]for i in range(len(x0))])

    fig, ax = plt.subplots(1, 1, figsize=(8,8), sharex=True, sharey=True)
    for i in range(len(x0)):
        x, y, p = xS[:,i,-3].numpy(), xS[:,i,-2].numpy(), tf.softmax(xS[:,i,-3:-1].to(device))[:,1]
        points = np.array([x, y]).T.reshape(-1, 1, 2) ;
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(pS.min(), pS.max())
        lc = LineCollection(segments, cmap='inferno', norm=norm, alpha=.3)
        lc.set_array(p) ; lc.set_linewidth(4) ; line = ax.add_collection(lc)
    # pS_ = model[-1](xS[-1,:,-2:].to(device)).view(-1).detach().cpu().numpy()
    ax.scatter(xS[-1,:,-3], xS[-1,:,-2], c='lime', edgecolor='none', s=30)
    ax.scatter(xS[0,:,-3], xS[0,:,-2], c='black', alpha=.5, s=30)
    # ax.plot(r, dec_bound(model, r), '--k')
    ax.set_xlim(xS[:,:,-3].min(), xS[:,:,-3].max()) ; ax.set_ylim(xS[:,:,-2].min(), xS[:,:,-2].max())
    plt.xlabel(r'$\mathbf{x_1}$', fontsize=20, fontweight='bold')
    plt.ylabel(r'$\mathbf{x_2}$', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('fulltraj_lya'+str+'.pdf', bbox_inches='tight')
    plt.show()
    return model
# plot_traj(Net)
# torch.save(model.state_dict(), 'CA.pt')

# def plot_lossviatime(model1, model2, device=torch.device("cpu")):
#     x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2);
#     x0 = torch.Tensor(x0).to(device)
#     ys = torch.LongTensor(ys.long()).to(device)
#     s = torch.linspace(0, 1, 50)
#     controller = torch.zeros(len(x0), 1)
#     x0 = torch.cat((x0, controller), 1)
#     # t, s = model2(x0)
#     # t_eval, xS = model(x0, s)
#     t_eval_1, xS_1 = model1(x0)
#     xS_1 = xS_1.detach().cpu()
#     loss1=[]
#     for i in range(len(t_eval_1)):
#         y_hat = xS_1[i][:, :-1]  # select last point of solution trajectory
#         loss = nn.CrossEntropyLoss()(y_hat, ys)
#         loss1.append(loss)
#     # model2 = model2.cpu()
#     t_eval_2, xS_2 = model2(x0)
#     xS_2 = xS_2.detach().cpu()
#     loss2 = []
#     for i in range(len(t_eval_2)):
#         y_hat = xS_2[i][:, :-1]  # select last point of solution trajectory
#         loss = nn.CrossEntropyLoss()(y_hat, ys)
#         loss2.append(loss)
#     # xS = model[0].trajectory(x0, s).detach() ;
#     # model = model.to(device)
#     # r = 1.05 * torch.linspace(xS[:, :, -2].min(), xS[:, :, -2].max(), 2)
#     # pS = torch.cat([tf.softmax(xS[:, i, -3:-1].to(device))[:, 1] for i in range(len(x0))])
#     #
#     # fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
#     # for i in range(len(x0)):
#     #     x, y, p = xS[:, i, -3].numpy(), xS[:, i, -2].numpy(), tf.softmax(xS[:, i, -3:-1].to(device))[:, 1]
#     #     points = np.array([x, y]).T.reshape(-1, 1, 2);
#     #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     #     norm = plt.Normalize(pS.min(), pS.max())
#     #     lc = LineCollection(segments, cmap='inferno', norm=norm, alpha=.3)
#     #     lc.set_array(p);
#     #     lc.set_linewidth(2);
#     #     line = ax.add_collection(lc)
#     # # pS_ = model[-1](xS[-1,:,-2:].to(device)).view(-1).detach().cpu().numpy()
#     # ax.scatter(xS[-1, :, -3], xS[-1, :, -2], c='lime', edgecolor='none', s=30)
#     # ax.scatter(xS[0, :, -3], xS[0, :, -2], c='black', alpha=.5, s=30)
#     # # ax.plot(r, dec_bound(model, r), '--k')
#     # ax.set_xlim(xS[:, :, -3].min(), xS[:, :, -3].max());
#     # ax.set_ylim(xS[:, :, -2].min(), xS[:, :, -2].max())
#     l1=plt.plot(t_eval_1, loss1, color='b')
#     l2=plt.plot(t_eval_2, loss2, color='g', linestyle='dashed')
#     plt.xlim(0,1)
#     plt.legend((l1,l2),labels=['fixed', 'optimized'], loc='best')
#     plt.xlabel('$t$')
#     plt.ylabel('loss')
#     plt.show()
#     return model1, model2
# # model1 = Net()
# # model2 = Net()
# # model1.load_state_dict(torch.load('augmented_CM_T_fix.pt'))
# # model2.load_state_dict(torch.load('augmented_CM_T.pt'))
# # plot_lossviatime(model1, model2)
# from matplotlib import pyplot
# palette = pyplot.get_cmap('Set1')
# def plot_lossviatime(Net, device=torch.device("cpu")):
#     plt.figure(figsize=(8, 8))
#     x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2);
#     x0 = torch.Tensor(x0).to(device)
#     ys = torch.LongTensor(ys.long()).to(device)
#     s = torch.linspace(0, 1, 50)
#     controller = torch.zeros(len(x0), 1)
#     x0 = torch.cat((x0, controller), 1)
#     # t, s = model2(x0)
#     # t_eval, xS = model(x0, s)
#     t_eval, xS = Net(x0)
#     xS_1 = xS.detach().cpu()
#     loss1 = []
#     for j in range(20):
#         loss_1 = []
#         for i in range(len(t_eval)):
#             y_hat = xS_1[i][j, :-1]  # select last point of solution trajectory
#             loss = nn.CrossEntropyLoss()(y_hat, ys[j])
#             loss_1.append(loss)
#         loss1.append(loss_1)
#     # model2 = model2.cpu()
#     # t_eval_2, xS_2 = model2(x0)
#     # xS_2 = xS_2.detach().cpu()
#     # loss2 = []
#     # for i in range(len(t_eval_2)):
#     #     y_hat = xS_2[i][:, :-1]  # select last point of solution trajectory
#     #     loss = nn.CrossEntropyLoss()(y_hat, ys)
#     #     loss2.append(loss)
#     # xS = model[0].trajectory(x0, s).detach() ;
#     # model = model.to(device)
#     # r = 1.05 * torch.linspace(xS[:, :, -2].min(), xS[:, :, -2].max(), 2)
#     # pS = torch.cat([tf.softmax(xS[:, i, -3:-1].to(device))[:, 1] for i in range(len(x0))])
#     #
#     # fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
#     # for i in range(len(x0)):
#     #     x, y, p = xS[:, i, -3].numpy(), xS[:, i, -2].numpy(), tf.softmax(xS[:, i, -3:-1].to(device))[:, 1]
#     #     points = np.array([x, y]).T.reshape(-1, 1, 2);
#     #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     #     norm = plt.Normalize(pS.min(), pS.max())
#     #     lc = LineCollection(segments, cmap='inferno', norm=norm, alpha=.3)
#     #     lc.set_array(p);
#     #     lc.set_linewidth(2);
#     #     line = ax.add_collection(lc)
#     # # pS_ = model[-1](xS[-1,:,-2:].to(device)).view(-1).detach().cpu().numpy()
#     # ax.scatter(xS[-1, :, -3], xS[-1, :, -2], c='lime', edgecolor='none', s=30)
#     # ax.scatter(xS[0, :, -3], xS[0, :, -2], c='black', alpha=.5, s=30)
#     # # ax.plot(r, dec_bound(model, r), '--k')
#     # ax.set_xlim(xS[:, :, -3].min(), xS[:, :, -3].max());
#     # ax.set_ylim(xS[:, :, -2].min(), xS[:, :, -2].max())
#     color = palette(1)
#     avg = np.mean(loss1, axis=0)
#     std = np.std(loss1, axis=0)
#     r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
#     r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
#     plt.fill_between(t_eval, r1, r2, color=color, alpha=0.2)
#     # l1=plt.plot(t_eval, loss1, color='b')
#     l1 = plt.plot(t_eval, avg, color=color, label='Vanilla', linewidth=3.5)
#     # l2=plt.plot(t_eval_2, loss2, color='g', linestyle='dashed')
#     plt.xlim(0,1)
#     # plt.legend((l1,l2),labels=['fixed', 'optimized'], loc='best')
#     plt.xlabel('$t$')
#     plt.ylabel('loss')
#     str= '10'
#     plt.savefig('lossviatime_lya20'+str+'.pdf', bbox_inches='tight')
#     plt.show()
#     return Net
# # model1 = Net()
# # model2 = Net()
# # model1.load_state_dict(torch.load('augmented_CM_T_fix.pt'))
# # model2.load_state_dict(torch.load('augmented_CM_T.pt'))
# plot_lossviatime(Net)


from matplotlib import pyplot
palette = pyplot.get_cmap('Set1')
def plot_lossviatime(model, x0, ys, str, device=torch.device("cpu")):
    plt.figure(figsize=(8, 8))
    # x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2);
    x0 = torch.Tensor(x0).to(device)
    ys = torch.LongTensor(ys.long()).to(device)
    s = torch.linspace(0, 1, 50)
    controller = torch.zeros(len(x0), 1)
    x0 = torch.cat((x0, controller), 1)
    # t, s = model2(x0)
    # t_eval, xS = model(x0, s)
    t_eval, xS = model(x0)
    xS_1 = xS.detach().cpu()
    loss1 = []
    for j in range(20):
        loss_1 = []
        for i in range(len(t_eval)):
            y_hat = xS_1[i][j, :-1]  # select last point of solution trajectory
            loss = nn.CrossEntropyLoss()(y_hat, ys[j])
            loss_1.append(loss)
        loss1.append(loss_1)
    # model2 = model2.cpu()
    # t_eval_2, xS_2 = model2(x0)
    # xS_2 = xS_2.detach().cpu()
    # loss2 = []
    # for i in range(len(t_eval_2)):
    #     y_hat = xS_2[i][:, :-1]  # select last point of solution trajectory
    #     loss = nn.CrossEntropyLoss()(y_hat, ys)
    #     loss2.append(loss)
    # xS = model[0].trajectory(x0, s).detach() ;
    # model = model.to(device)
    # r = 1.05 * torch.linspace(xS[:, :, -2].min(), xS[:, :, -2].max(), 2)
    # pS = torch.cat([tf.softmax(xS[:, i, -3:-1].to(device))[:, 1] for i in range(len(x0))])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    # for i in range(len(x0)):
    #     x, y, p = xS[:, i, -3].numpy(), xS[:, i, -2].numpy(), tf.softmax(xS[:, i, -3:-1].to(device))[:, 1]
    #     points = np.array([x, y]).T.reshape(-1, 1, 2);
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     norm = plt.Normalize(pS.min(), pS.max())
    #     lc = LineCollection(segments, cmap='inferno', norm=norm, alpha=.3)
    #     lc.set_array(p);
    #     lc.set_linewidth(2);
    #     line = ax.add_collection(lc)
    # # pS_ = model[-1](xS[-1,:,-2:].to(device)).view(-1).detach().cpu().numpy()
    # ax.scatter(xS[-1, :, -3], xS[-1, :, -2], c='lime', edgecolor='none', s=30)
    # ax.scatter(xS[0, :, -3], xS[0, :, -2], c='black', alpha=.5, s=30)
    # # ax.plot(r, dec_bound(model, r), '--k')
    # ax.set_xlim(xS[:, :, -3].min(), xS[:, :, -3].max());
    # ax.set_ylim(xS[:, :, -2].min(), xS[:, :, -2].max())
    color = palette(1)
    avg = np.mean(loss1, axis=0)
    print(avg)
    std = np.std(loss1, axis=0)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
    plt.fill_between(t_eval, r1, r2, color=color, alpha=0.2)
    # l1=plt.plot(t_eval, loss1, color='b')
    l1 = plt.plot(t_eval, avg, color=color, label='Vanilla', linewidth=5)
    # l2=plt.plot(t_eval_2, loss2, color='g', linestyle='dashed')
    plt.xlim(0,1)
    # plt.legend((l1,l2),labels=['fixed', 'optimized'], loc='best')
    plt.xlabel(r'$\mathbf{t}$', fontsize=20, fontweight='bold')
    plt.ylabel('loss', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.savefig('lossviatime_lya'+str+'.pdf', bbox_inches='tight')
    plt.show()
    return Net
model1 = Net(torch.tensor(1))
model2 = Net(torch.tensor(1))
model3 = Net(torch.tensor(1))
model4 = Net1(torch.tensor(0.09))
# model5 = Net1(torch.tensor(0.05))
model1.load_state_dict(torch.load('augmented_CM_lya_1_rk4.pt'))
model2.load_state_dict(torch.load('augmented_CM_lya_10_rk4.pt'))
model3.load_state_dict(torch.load('augmented_CM_lya_20_rk4.pt'))
model4.load_state_dict(torch.load('augmented_CM_lya_20_rk4_t_f.pt'))
# model5.load_state_dict(torch.load('augmented_CM_aug_005_rk4_t_f.pt'))
# plot_traj(model1, x0, ys, '1')
# plot_traj(model2, x0, ys, '10')
# plot_traj(model3, x0, ys, '20')
# plot_traj(model4, x0, ys, '20_005')
# plot_traj(model5, x0, ys, 'aug_005')
# plot_lossviatime(model1, x0, ys, '1')
# plot_lossviatime(model2, x0, ys, '10')
# plot_lossviatime(model3, x0, ys, '20')
# plot_lossviatime(model4, x0, ys, '20_005')
# plot_lossviatime(model5, x0, ys, 'aug_005')

plot_lossviatime(fullNet, x0, ys, '1')