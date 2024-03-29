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
import torch.nn as nn
import pytorch_lightning as pl
import random
from torch.autograd.functional import jvp
import os


device=torch.device("cpu")
random.seed(27)
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
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 3)
    )
        self.t1 = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.n_steps = 100
        self.dt = 0.01

    def step_size_func(self, t0, t1):
        """Computes a variable step size based on the current value of t1."""
        dt = t1 / self.n_steps
        dt = 0.01
        self.n_steps = torch.ceil(t1/dt).int() - 1
        return self.n_steps

    def dot(self, x):
        return self.f(x)

    def forward(self, x):
        y0 = x.view(-1, 3)
        odefunc = lambda t, y: self.f(y)
        self.t1.data.clamp_(0.01001, 1)
        t0, t1 = nn.Parameter(torch.tensor([0.0]), requires_grad=False), self.t1
        # print('terminal time:')
        # print(t1)

        t_seq = torch.arange(t0.item(), t1.item()-1e-6, self.dt)
        t_seq = torch.cat((t_seq, t1.view(1)), dim=0)


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
        # static_state, _ = self.model.init_coordinates(x, self.model.dyn_fun)
        # x_in = static_state[:, None].expand(-1, self.h_sample_size, *((-1,) * (static_state.ndim - 1))).flatten(0, 1)
        # y_hat = y_hat[:, :, :-1].flatten(0,1)
        y_in  = y[None,:].expand(y_hat.size(0),-1).flatten(0,1)
        test = nn.functional.cross_entropy(y_hat[:, :, :-1].flatten(0,1), y_in, reduction='none')
        # y_in = y[:, None].expand(-1, 101).flatten(0, 1)

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
        # self.log("model_gain", self.model.dyn_fun.gain, on_step=True, logger=True)
        # self.log('effective_batch_size', effective_batch_size, on_step=True, logger=True)
        h_sample_in = None
        x_in = None
        y_in = None
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        controller = torch.zeros(len(X), 1)
        x = torch.cat((x,controller),1)
        t_eval, y_hat = self.model(x)
        # print(t_eval[-1])
        y_hat_fin = y_hat[-1][:,:-1] # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_hat_fin, y) + t_eval[-1] + self.compute_loss(x, y_hat, y)
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



        # return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader

#
logger = TensorBoardLogger('tb_logs', name='augmented_0.05')
# #
Net = Net()
learn = Learner(Net)
trainer = pl.Trainer(min_epochs=300, max_epochs=800, logger=logger)
start = time.perf_counter()
trainer.fit(learn)
end = time.perf_counter()
print(end-start)
torch.save(Net.state_dict(), 'augmented_CM_T_rk4fix_combine_from005_.pt')

# Net.load_state_dict(torch.load('augmented_CM_T_rk4fix_combine_from005.pt'))

controller = torch.zeros(len(X_train), 1)
Z_train = torch.cat((X_train,controller),1)
# start = time.perf_counter()
t_eval, trajectory = Net(Z_train)
# end = time.perf_counter()
# print('inference')
# print(end-start)
#

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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch.nn.functional as tf

# x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2)
import pickle

# X, yn = d.generate(n_samples=512, noise=1e-2, dataset_type='spheres', dim=2)

# with open('data.pkl', 'wb') as f:
#     pickle.dump((x0, ys), f)

with open('data.pkl', 'rb') as f:
    x0, ys = pickle.load(f)


def dec_bound(model, x):
    P = [p for p in model[-1].parameters()]
    w1, w2, b = P[0][0][0].cpu().detach(), P[0][0][1].cpu().detach(), P[1][0].cpu().detach().item()
    return (-w1*x - b + .5)/w2

def plot_traj(model, x0, ys, device=torch.device("cpu")):
    # x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2);
    x0 = torch.Tensor(x0).to(device)
    ys = torch.LongTensor(ys.long()).to(device)
    s = torch.linspace(0, 1, 50)
    controller = torch.zeros(len(x0), 1)
    x0 = torch.cat((x0, controller), 1)



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
    ax.set_xlim(-2, 2);
    ax.set_ylim(-2, 2)
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.savefig('zoom_.pdf', bbox_inches='tight')
    ax.set_xlim(xS[:,:,-3].min(), xS[:,:,-3].max()) ; ax.set_ylim(xS[:,:,-2].min(), xS[:,:,-2].max())
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$\mathbf{x_1}$',fontsize=20, fontweight='bold')
    plt.ylabel(r'$\mathbf{x_2}$',fontsize=20, fontweight='bold')
    plt.savefig('fulltraj_MNODE.pdf', bbox_inches='tight')
    plt.show()
    return model


plot_traj(Net, x0, ys)
# torch.save(model.state_dict(), 'CA.pt')
from matplotlib import pyplot
palette = pyplot.get_cmap('Set1')
def plot_lossviatime(model1, x0, ys, device=torch.device("cpu")):
    plt.figure(figsize=(8, 8))
    # x0, ys = d.generate(n_samples=20, noise=1e-2, dataset_type='spheres', dim=2);
    x0 = torch.Tensor(x0).to(device)
    ys = torch.LongTensor(ys.long()).to(device)
    # s = torch.linspace(0, 1, 50)
    controller = torch.zeros(len(x0), 1)
    x0 = torch.cat((x0, controller), 1)
    # t, s = model2(x0)
    # t_eval, xS = model(x0, s)
    t_eval_1, xS_1 = model1(x0)
    t_eval_1 = t_eval_1.detach().cpu()
    xS_1 = xS_1.detach().cpu()
    loss1=[]
    for j in range(20):
        loss_1 = []
        for i in range(len(t_eval_1)):
            y_hat = xS_1[i][j, :-1]  # select last point of solution trajectory
            loss = nn.CrossEntropyLoss()(y_hat, ys[j])
            loss_1.append(loss)
        loss1.append(loss_1)
    color = palette(1)
    avg = np.mean(loss1, axis=0)
    print(avg)
    std = np.std(loss1, axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg,std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(avg,std)))
    l1 = plt.plot(t_eval_1, avg, color=color, linewidth=5)
    plt.fill_between(t_eval_1, r1, r2, color=color,alpha=0.2)

    plt.xlim(0,1)
    # plt.legend(loc='best', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$\mathbf{t}$', fontsize=20, fontweight='bold')
    plt.ylabel('loss', fontsize=20, fontweight='bold')
    # plt.savefig('lossviatime_MNODE.pdf', bbox_inches='tight')
    plt.show()
    return model1

plot_lossviatime(Net, x0, ys)