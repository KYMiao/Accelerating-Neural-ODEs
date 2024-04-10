import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default="./train")
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time = torch.tensor([0, 1]).float()
        self.integration_time = nn.Parameter(torch.tensor(1.), requires_grad=True)
        # self.n_steps = 100

        self.dt = -0.01

    def forward(self, x):
        self.integration_time.data.clamp_(0.01001, 1)
        t1 = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        # t0 = t1 - self.integration_time
        t0 = nn.Parameter(torch.tensor([0.75]), requires_grad=False)

        # self.integration_time = self.integration_time.type_as(x)
        print('terminal time:')
        print(t0)
        t_seq = torch.arange(t1.item(), t0.item(), self.dt)
        t_seq = torch.cat((t_seq, t0.view(1)), dim=0)
        out = odeint(self.odefunc, x, t_seq, method='rk4')
        # out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch(num_samples):
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return (x, logp_diff_t1)


if __name__ == '__main__':
    t0 = 0
    t1 = 0.75
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # model
    func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
    ckpt_path = '/Users/univ5181/Desktop/缪可言博士/YEAR 1/Term 3/Neural ODEs/train/cnf_minimal_circles_35.pth'
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    func.load_state_dict(checkpoint['func_state_dict'])
    model = ODEBlock(func)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
    )
    loss_meter = RunningAverageMeter()



    # if args.train_dir is not None:
    #     if not os.path.exists(args.train_dir):
    #         os.makedirs(args.train_dir)
    #     ckpt_path = os.path.join(args.train_dir, 'cnf_minimal.pth')
        # if os.path.exists(ckpt_path):
        #     checkpoint = torch.load(ckpt_path)
        #     func.load_state_dict(checkpoint['func_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     print('Loaded ckpt from {}'.format(ckpt_path))

    # try:
    #     for itr in range(1, args.niters + 1):
    #         optimizer.zero_grad()
    #
    #         x, logp_diff_t1 = get_batch(args.num_samples)
    #         # t_span = torch.tensor(np.linspace(t1, t0, 100)).type(torch.float32).to(device)
    #         # z_t, logp_diff_t = odeint(
    #         #     func,
    #         #     (x, logp_diff_t1),
    #         #     t_span,
    #         #     atol=1e-5,
    #         #     rtol=1e-5,
    #         #     method='rk4',
    #         # )
    #
    #         z_t, logp_diff_t = model((x, logp_diff_t1))
    #
    #         z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    #
    #         logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
    #         loss = -logp_x.mean(0) + model.integration_time
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         loss_meter.update(loss.item())
    #
    #         print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))
    #
    #     ckpt_path = os.path.join(args.train_dir, 'cnf_minimal.pth')
    #     torch.save({
    #         'func_state_dict': func.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #     }, ckpt_path)
    #
    # except KeyboardInterrupt:
    #     if args.train_dir is not None:
    #         ckpt_path = os.path.join(args.train_dir, 'cnf_minimal.pth')
    #         torch.save({
    #             'func_state_dict': func.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, ckpt_path)
    #         print('Stored ckpt at {}'.format(ckpt_path))
    # print('Training complete after {} iters.'.format(itr))

    # ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
    # torch.save({
    #     'func_state_dict': func.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, ckpt_path)

    if args.viz:
        viz_samples = 30000
        viz_timesteps = 100
        dt = 0.01
        target_sample, _ = get_batch(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            t1 = torch.tensor([1.0])
            t0 = torch.tensor([0.75])
            # t0 = t1 - model.integration_time

            # self.integration_time = self.integration_time.type_as(x)

            t_seq = torch.arange(t0.item(), t1.item(), dt)
            t_seq = torch.cat((t_seq, t1.view(1)), dim=0)
            z_t_samples, _ = odeint(
                model.odefunc,
                (z_t0, logp_diff_t0),
                t_seq.to(device),
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

            z_t_density, logp_diff_t = model((z_t1, logp_diff_t1))

            # z_t_density, logp_diff_t = odeint(
            #     func,
            #     (z_t1, logp_diff_t1),
            #     torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
            #     atol=1e-5,
            #     rtol=1e-5,
            #     method='rk4',
            # )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    t_seq,
                    z_t_samples, z_t_density, logp_diff_t
            ):
                logp_x = p_z0.log_prob(z_density).to(device) - logp_diff.view(-1)
                print(-logp_x.mean(0))
                print(t)
                fig1 = plt.figure(figsize=(4, 4), dpi=200)
                plt.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-terminal-circles-{int(t * 1000):05d}-ICML.jpg"),
                            pad_inches=0.2, bbox_inches='tight')
                fig2 = plt.figure(figsize=(4, 4), dpi=200)
                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                plt.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-terminal-circles-{int(t * 1000):05d}-den-ICML.jpg"),
                            pad_inches=0.2, bbox_inches='tight')
        #         fig = plt.figure(figsize=(12, 4), dpi=200)
        #         plt.tight_layout()
        #         plt.axis('off')
        #         plt.margins(0, 0)
        #         fig.suptitle(f'{t:.2f}s')
        #
        #         ax1 = fig.add_subplot(1, 3, 1)
        #         ax1.set_title('Target')
        #         ax1.get_xaxis().set_ticks([])
        #         ax1.get_yaxis().set_ticks([])
        #         ax2 = fig.add_subplot(1, 3, 2)
        #         ax2.set_title('Samples')
        #         ax2.get_xaxis().set_ticks([])
        #         ax2.get_yaxis().set_ticks([])
        #         ax3 = fig.add_subplot(1, 3, 3)
        #         ax3.set_title('Log Probability')
        #         ax3.get_xaxis().set_ticks([])
        #         ax3.get_yaxis().set_ticks([])
        #
        #         ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
        #                    range=[[-1.5, 1.5], [-1.5, 1.5]])
        #
        #         ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
        #                    range=[[-1.5, 1.5], [-1.5, 1.5]])
        #
        #         logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
        #         ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
        #                         np.exp(logp.detach().cpu().numpy()), 200)
        #
        #         plt.savefig(os.path.join(args.results_dir, f"--cnf-viz-{int(t * 1000):05d}.jpg"),
        #                     pad_inches=0.2, bbox_inches='tight')
        #         plt.close()
        #
        #     img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"--cnf-viz-*.jpg")))]
        #     img.save(fp=os.path.join(args.results_dir, "cnf-viz-terminal__.gif"), format='GIF', append_images=imgs,
        #              save_all=True, duration=250, loop=0)
        #
        # print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz-terminal__.gif")))