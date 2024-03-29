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
    cross_entropy_loss = nn.CrossEntropyLoss()  # CE

    while tf <= max_tf:
        model.model.t1 = torch.tensor(tf)  # update tf
        one_epoch_trainer = pl.Trainer(max_epochs=1)
        one_epoch_trainer.fit(model, trainloader)  # train with new tf

        model.eval()  # eval
        total_ce = 0
        total_samples = 0

        with torch.no_grad():
            for batch in trainloader:
                inputs, labels = batch
                controller = torch.zeros(len(inputs), 1)
                inputs = torch.cat((inputs, controller), 1)
                t_eval, outputs = model(inputs) 
                loss = cross_entropy_loss(outputs[-1][:,:-1], labels)
                total_ce += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        average_ce = total_ce / total_samples  # mean ce
        # print(average_ce)
        if average_ce < threshold:
            break  # stop if < threshold

        tf += tf_increment  # increase tf
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




from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch.nn.functional as tf


import pickle

# X, yn = d.generate(n_samples=512, noise=1e-2, dataset_type='spheres', dim=2)

# with open('data.pkl', 'wb') as f:
#     pickle.dump((x0, ys), f)

with open('data.pkl', 'rb') as f:
    x0, ys = pickle.load(f)

from matplotlib import pyplot
palette = pyplot.get_cmap('Set1')
def plot_lossviatime(model, x0, ys, device=torch.device("cpu")):
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
    return model

plot_lossviatime(fullNet, x0, ys)
