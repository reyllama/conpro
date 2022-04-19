import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import math


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, k=16, device=torch.device("cuda:0"), n_task=7, **kwargs):
        super().__init__()
        self.size = size
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.z_dim = z_dim
        self.num_layers = int(math.log(size, 2))-1

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_pls = task_ResnetBlock(16*nf//k, 16*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_pls = task_ResnetBlock(16*nf//k, 16*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_pls = task_ResnetBlock(16*nf//k, 8*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_pls = task_ResnetBlock(8*nf//k, 4*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_pls = task_ResnetBlock(4*nf//k, 2*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_pls = task_ResnetBlock(2*nf//k, 1*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.resnet_6_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_6_pls = task_ResnetBlock(1*nf//k, 1*nf//k, k=k, device=device, n_task=n_task) # Plasticity Block
        self.conv_img = nn.Conv2d(nf, 3, 7, padding=3)

        # self.mixing_0_pls = nn.ModuleList([nn.Conv2d(32 * nf, 16 * nf, 1) for _ in range(n_task)])
        # self.mixing_1_pls = nn.ModuleList([nn.Conv2d(32 * nf, 16 * nf, 1) for _ in range(n_task)])
        # self.mixing_2_pls = nn.ModuleList([nn.Conv2d(16 * nf, 8 * nf, 1) for _ in range(n_task)])
        # self.mixing_3_pls = nn.ModuleList([nn.Conv2d(8 * nf, 4 * nf, 1) for _ in range(n_task)])
        # self.mixing_4_pls = nn.ModuleList([nn.Conv2d(4 * nf, 2 * nf, 1) for _ in range(n_task)])
        # self.mixing_5_pls = nn.ModuleList([nn.Conv2d(2 * nf, 1 * nf, 1) for _ in range(n_task)])
        # self.mixing_6_pls = nn.ModuleList([nn.Conv2d(2 * nf, 1 * nf, 1) for _ in range(n_task)])

        self.mixing_0_pls = nn.Parameter(torch.rand([n_task], device=device))
        self.mixing_1_pls = nn.Parameter(torch.rand([n_task], device=device))
        self.mixing_2_pls = nn.Parameter(torch.rand([n_task], device=device))
        self.mixing_3_pls = nn.Parameter(torch.rand([n_task], device=device))
        self.mixing_4_pls = nn.Parameter(torch.rand([n_task], device=device))
        self.mixing_5_pls = nn.Parameter(torch.rand([n_task], device=device))
        self.mixing_6_pls = nn.Parameter(torch.rand([n_task], device=device))


    def forward(self, z, y, return_feats=False):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)
        feats = []
        task_id = y[0]

        yembed = self.embedding(y)
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out_task = self.resnet_0_pls(out, task_id=task_id)
        out = self.resnet_0_0(out)
        out = self.mixing_0_pls[task_id] * out_task + (1-self.mixing_0_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_0_pls[y[0]](out)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)

        out_task = self.resnet_1_pls(out, task_id=task_id)
        out = self.resnet_1_0(out)
        out = self.mixing_1_pls[task_id] * out_task + (1 - self.mixing_1_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_1_pls[y[0]](out)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)

        out_task = self.resnet_2_pls(out, task_id=task_id)
        out = self.resnet_2_0(out)
        out = self.mixing_2_pls[task_id] * out_task + (1 - self.mixing_2_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_2_pls[y[0]](out)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)

        out_task = self.resnet_3_pls(out, task_id=task_id)
        out = self.resnet_3_0(out)
        out = self.mixing_3_pls[task_id] * out_task + (1 - self.mixing_3_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_3_pls[y[0]](out)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)

        out_task = self.resnet_4_pls(out, task_id=task_id)
        out = self.resnet_4_0(out)
        out = self.mixing_4_pls[task_id] * out_task + (1 - self.mixing_4_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_4_pls[y[0]](out)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)

        out_task = self.resnet_5_pls(out, task_id=task_id)
        out = self.resnet_5_0(out)
        out = self.mixing_5_pls[task_id] * out_task + (1 - self.mixing_5_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_5_pls[y[0]](out)
        if return_feats:
            feats.append(out)

        # if self.size == 256:

        out = F.interpolate(out, scale_factor=2)

        out_task = self.resnet_6_pls(out, task_id=task_id)
        out = self.resnet_6_0(out)
        out = self.mixing_6_pls[task_id] * out_task + (1 - self.mixing_6_pls[task_id]) * out
        # out = torch.cat([out, out_task], dim=1)
        # out = self.mixing_6_pls[y[0]](out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out, feats

class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.size = size
        self.num_layers = int(math.log(size, 2))-1

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 7, padding=3)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_1_0 = ResnetBlock(1*nf, 2*nf)
        self.resnet_2_0 = ResnetBlock(2*nf, 4*nf)
        self.resnet_3_0 = ResnetBlock(4*nf, 8*nf)
        self.resnet_4_0 = ResnetBlock(8*nf, 16*nf)
        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_6_0 = ResnetBlock(16*nf, 16*nf)

        self.patch_conv = nn.ModuleList(
            [nn.Conv2d(4*nf, 1, 3),
             nn.Conv2d(8*nf, 1, 3),
             nn.Conv2d(16*nf, 1, 3),
             nn.Conv2d(16*nf, 1, 3)]
        )

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)

    def forward(self, x, y, mdl=False, idx=None):
        if not mdl:
            assert(x.size(0) == y.size(0))
        batch_size = x.size(0)
        feats = []

        out = self.conv_img(x)
        out = self.resnet_0_0(out)
        if mdl:
            feats.append(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        if mdl:
            feats.append(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        if mdl:
            feats.append(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        if mdl:
            feats.append(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        if mdl:
            feats.append(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        if mdl:
            feats.append(out)
            feat = feats[2+idx] # 0<=idx<=3
            feat = feat[:batch_size//2] # only those for interpolated images
            pred = self.patch_conv[idx](feat)
            return pred, feats

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_6_0(out)

        # print(f"component size: {batch_size} / {self.nf} / {self.s0} / {16*self.nf*self.s0*self.s0}")
        out = out.contiguous().view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out, feats

    def evaluate_distance(self):
        pass


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class task_ResnetBlock(nn.Module):
    def __init__(self, fin, fout, n_task=7, k=16, fhidden=None, is_bias=True, device=None):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_bottle = nn.ModuleList([nn.Conv2d(self.fin*k, self.fin, 1, device=device) for _ in range(n_task)])
        self.conv_0 = nn.ModuleList([nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, device=device) for _ in range(n_task)])
        self.conv_1 = nn.ModuleList([nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias, device=device) for _ in range(n_task)])
        if self.learned_shortcut:
            self.conv_s = nn.ModuleList([nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False, device=device) for _ in range(n_task)])
        # self.coefs = [nn.Linear(fout, fout*k, device=device) for _ in range(n_task)] # Rescaling feature maps channel-wise
        # self.coefs = [nn.Parameter(torch.rand(fout, fout*k, device=device)) for _ in range(n_task)]
        # self.coefs = [nn.Parameter(torch.div(torch.rand(fout, fout*k, device=device)-0.5, math.sqrt(fout))) for _ in range(n_task)]
        self.conv_bottle_out = nn.ModuleList([nn.Conv2d(self.fout, fout*k, 1, device=device) for _ in range(n_task)])

    def forward(self, x, task_id):
        x = self.conv_bottle[task_id](x)
        x_s = self._shortcut(x, task_id)
        dx = self.conv_0[task_id](actvn(x))
        dx = self.conv_1[task_id](actvn(dx))
        out = x_s + 0.1 * dx # (N, C, H, W)
        # N, C, H, W = out.size()
        # out = out.view(N, C, -1).permute(0,2,1).view(-1,C) # (NHW, C)
        # out = out.view(N, C, -1)
        # out = out.permute(0,2,1)
        # out = out.view(-1, C)
        # print(out.size())
        # out = torch.matmul(out, self.coefs[task_id]).permute(0,2,1) # (N, K, HW)
        # out = self.coefs[task_id](out) # (NHW, C*K)
        out = self.conv_bottle_out[task_id](actvn(out))
        # out = out.view(N, -1, H, W)

        return out

    def _shortcut(self, x, task_id):
        if self.learned_shortcut:
            x_s = self.conv_s[task_id](x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
