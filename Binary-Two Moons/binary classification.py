from torchdyn.core import NeuralODE
from torchdyn.datasets import *
import torchdiffeq
from torchdiffeq import odeint
from pytorch_lightning.loggers import TensorBoardLogger
from torchdyn.nn import DataControl, DepthCat, Augmenter, GalLinear, Fourier
from torchdyn import *
import time
from torch.autograd.functional import jvp

print("ok")
d = ToyDataset()
X, yn = d.generate(n_samples=512, noise=1e-1, dataset_type='moons')
dry_run = False
#
import matplotlib.pyplot as plt
#
# # colors = ['orange', 'blue']
# from matplotlib import pyplot
# # palette = pyplot.get_cmap('Set1')
# # colors=[palette(1), palette(2)]
# # fig = plt.figure(figsize=(3,3))
# # ax = fig.add_subplot(111)
# # for i in range(len(X)):
# #     ax.scatter(X[i,0], X[i,1], s=1, color=colors[yn[i].int()])
# # plt.show()
#
import torch
import torch.utils.data as data
device = torch.device("cpu")
print("ok")
X_train = torch.Tensor(X).to(device)
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)
#
import torch.nn as nn
import pytorch_lightning as pl

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        self.t1 = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.n_steps = 100
        self.dt = 0.01

    def dot(self, x):
        return self.f(x)


    def forward(self, x):
        y0 = x.view(-1, 2)
        odefunc = lambda t, y: self.f(y)
        t0, t1 = nn.Parameter(torch.tensor([0.0]), requires_grad=False), self.t1
        # print('terminal time:')
        # print(t1)


        t_1 = round(t1.item(), 4)
        t_seq = torch.arange(t0.item(), t_1, self.dt)
        t_seq = torch.cat((t_seq, t1.view(1)), dim=0)
        # t_seq = torch.arange(0,0.09,0.01)

        # num_steps = torch.floor(t1 / self.dt).long()


        y = odeint(odefunc, y0, t_seq, method='rk4')
        return t_seq, y

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.order = 1

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y_hat, y):
        # static_state, _ = self.model.init_coordinates(x, self.model.dyn_fun)
        # x_in = static_state[:, None].expand(-1, self.h_sample_size, *((-1,) * (static_state.ndim - 1))).flatten(0, 1)
        # y_hat = y_hat[:, :, :-1].flatten(0,1)
        y_in  = y[None,:].expand(11,-1).flatten(0,1)
        test = nn.functional.cross_entropy(y_hat.flatten(0,1), y_in, reduction='none')
        # y_in = y[:, None].expand(-1, 101).flatten(0, 1)

        def v_ndot(order: int, oc_in):
            assert isinstance(order, int) and order >= 0, \
                f"[ERROR] Order({order}) must be non-negative integer."
            if order == 0:
                return nn.functional.cross_entropy(oc_in.flatten(0,1), y_in, reduction='none')
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

    def training_step(self, batch, batch_index):
        x, y = batch
        t_eval, y_hat = self.model(x)
        # y_hat = y_hat[-1]
        loss = self.compute_loss(x, y_hat, y)
        # loss = self.aug_loss(y_hat, y)
        # loss = nn.CrossEntropyLoss()(y_hat, y) + t_eval[-1]
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.8)
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
        'monitor': 'my_loss'
    }

    def train_dataloader(self):
        return trainloader

logger = TensorBoardLogger('tb_logs', name='binary')
#
Net = Net()
learn = Learner(Net)
trainer = pl.Trainer(min_epochs=200, max_epochs=500, logger=logger)
start = time.perf_counter()
trainer.fit(learn)
end = time.perf_counter()
print(end-start)
torch.save(Net.state_dict(), 'binary_lya_20.pt')
# Net.load_state_dict(torch.load('binary_lya_20.pt'))
# torch.save(model,'test.pt')
start = time.time()
t_eval, trajectory = Net(X_train)
end = time.time()
print(end-start)
print(nn.functional.cross_entropy(trajectory[-1], yn))

trajectory = trajectory.detach().cpu()

# color=['orange', 'blue']

from matplotlib import pyplot
palette = pyplot.get_cmap('Set1')
color=[palette(1), palette(2)]
fig = plt.figure(figsize=(12,4))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for i in range(500):
    ax0.plot(t_eval.detach().numpy(), trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
    ax1.plot(t_eval.detach().numpy(), trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
ax0.set_xlabel(r"$t$ [Depth]") ; ax0.set_ylabel(r"$h_0(t)$")
ax1.set_xlabel(r"$t$ [Depth]") ; ax1.set_ylabel(r"$z_1(t)$")
ax0.set_title("Dimension 0") ; ax1.set_title("Dimension 1")
plt.savefig('moon_lya20.pdf', bbox_inches='tight')
plt.show()

