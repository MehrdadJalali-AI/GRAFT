"""Conditional WGAN-GP generator of synthetic node features (projected onto the B-ball).

Trained on labelled training nodes only; the condition is the class label. Its outputs are
projected onto the feature ball so that Theorems 1-2 (which need ||x|| <= B) hold."""
import numpy as np
import torch
import torch.nn as nn


class G(nn.Module):
    def __init__(self, z, C, d, h=128):
        super().__init__()
        self.emb = nn.Embedding(C, 16)
        self.net = nn.Sequential(nn.Linear(z + 16, h), nn.LeakyReLU(0.2), nn.Linear(h, h),
                                 nn.LeakyReLU(0.2), nn.Linear(h, d))

    def forward(self, z, c):
        x = self.net(torch.cat([z, self.emb(c)], 1))
        return x / torch.clamp(x.norm(dim=1, keepdim=True), min=1.0)  # projection in the loop


class D(nn.Module):
    def __init__(self, C, d, h=128):
        super().__init__()
        self.emb = nn.Embedding(C, 16)
        self.net = nn.Sequential(nn.Linear(d + 16, h), nn.LeakyReLU(0.2), nn.Linear(h, h),
                                 nn.LeakyReLU(0.2), nn.Linear(h, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, self.emb(c)], 1))


def train_cwgan(X, y, C, seed, steps=1500, zdim=32, n_critic=5, gp=10.0, bs=64):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y).long()
    # class-balanced sampling so minority conditions are trained as often as majority ones
    w = 1.0 / np.bincount(y, minlength=C)[y]
    w = torch.tensor(w / w.sum())
    g, d = G(zdim, C, X.shape[1]).double(), D(C, X.shape[1]).double()
    og = torch.optim.Adam(g.parameters(), 1e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(d.parameters(), 1e-4, betas=(0.5, 0.9))
    for _ in range(steps):
        for _ in range(n_critic):
            i = torch.multinomial(w, bs, replacement=True)
            xr, c = Xt[i], yt[i]
            xf = g(torch.randn(bs, zdim, dtype=torch.float64), c).detach()
            eps = torch.rand(bs, 1, dtype=torch.float64)
            xi = (eps * xr + (1 - eps) * xf).requires_grad_(True)
            gr = torch.autograd.grad(d(xi, c).sum(), xi, create_graph=True)[0]
            loss = d(xf, c).mean() - d(xr, c).mean() + gp * ((gr.norm(dim=1) - 1) ** 2).mean()
            od.zero_grad(); loss.backward(); od.step()
        i = torch.multinomial(w, bs, replacement=True)
        lg = -d(g(torch.randn(bs, zdim, dtype=torch.float64), yt[i]), yt[i]).mean()
        og.zero_grad(); lg.backward(); og.step()
    return g, zdim


def sample(g, zdim, cls, count, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return g(torch.randn(count, zdim, dtype=torch.float64),
                 torch.full((count,), cls, dtype=torch.long)).numpy()
