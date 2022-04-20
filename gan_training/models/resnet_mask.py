import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import math

def modulated_conv(x, y, weight, bias, weight_mask_left, weight_mask_right, bias_mask, padding):
    # base_filters = weight
    out_channels, in_channels, kernel_size = weight.size(0), weight.size(1), weight.size(2)
    task_id = y[0]-1
    if task_id >= 0:
        left_matrix = weight_mask_left[task_id]
        right_matrix = weight_mask_right[task_id]
        assert left_matrix.dim()==2
        assert right_matrix.dim()==2
        rank = left_matrix.size(1)
        right_matrix = right_matrix.permute(1,0)
        modulation = torch.mm(left_matrix, right_matrix) / math.sqrt(rank)
        modulation = modulation.view(out_channels, kernel_size, in_channels, kernel_size)
        modulation = modulation.permute(0,2,1,3)
        modulation = modulation.sigmoid()-0.5
        filters = weight * (modulation + 1.0)
    else:
        filters = weight
    filters = filters.to(dtype=x.dtype)
    if task_id >= 0:
        bias_ = bias * (bias_mask[task_id].sigmoid()+0.5)
    else:
        bias_ = bias
    return F.conv2d(x, filters, bias_, 1, padding)

# Seperately define 1x1 conv
class pointConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=4, instance_norm=False, channels_last=False, rank=-1, n_task=-1):
        super().__init__()
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.groups = groups
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # initialization
        weights = torch.cat([nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups).weight.data.unsqueeze(0) for _ in range(n_task-1)], dim=0)
        biases = torch.cat([nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups).bias.data.unsqueeze(0) for _ in range(n_task-1)], dim=0)
        self.weight_mask = nn.Parameter(weights)
        self.bias_mask = nn.Parameter(biases)
        self.n_task = n_task
        self.instance_norm = instance_norm

    def forward(self, x, y):
        task_id = y[0]-1
        if self.instance_norm:
            x = x / (x.std(dim=[2,3], keepdim=True) + 1e-8)
        return F.conv2d(x, self.weight_mask[task_id].to(dtype=x.dtype), self.bias_mask[task_id], self.kernel_size, self.padding, groups=self.groups)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, instance_norm=False, channels_last=False, rank=-1, n_task=-1):
        super().__init__()
        self.padding = kernel_size // 2 # preserve resolution
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(2*torch.rand([out_channels, in_channels, kernel_size, kernel_size])-1).to(memory_format=memory_format) # normal -> uniform(-1,1)
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / math.sqrt(in_channels * (kernel_size ** 2)) # normalize (equivalent to initialization) <- redundant if init. with pretrained weights

        self.n_task = n_task

        self.weight_mask_left = nn.Parameter(torch.randn([self.n_task-1, out_channels*kernel_size, rank]) / math.sqrt(out_channels*kernel_size))
        self.weight_mask_right = nn.Parameter(torch.randn([self.n_task-1, in_channels*kernel_size, rank]) / math.sqrt(in_channels*kernel_size))
        self.bias_mask = nn.Parameter(torch.zeros([self.n_task-1, out_channels]))

        self.instance_norm = instance_norm

    def forward(self, x, y):
        # weight = self.weight * self.weight_gain
        if self.instance_norm:
            x = x / (x.std(dim=[2,3], keepdim=True) + 1e-8)
        x = modulated_conv(x=x, y=y, weight=self.weight.to(x.dtype), bias=self.bias, weight_mask_left=self.weight_mask_left,
                           weight_mask_right=self.weight_mask_right, bias_mask=self.bias_mask, padding=self.padding)
        return x

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

class Gen_ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True, kernel_size=3, instance_norm=False, channels_last=False, rank=-1, n_task=-1):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        # self.learned_shortcut = False # TODO: First turn off 1x1 shortcuts
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        self.conv_0 = ConvLayer(self.fin, self.fhidden, kernel_size, instance_norm, channels_last, rank, n_task)
        self.conv_mix = pointConvLayer(self.fhidden, self.fhidden, 1, 4, instance_norm, channels_last, rank, n_task)
        self.conv_1 = ConvLayer(self.fhidden, self.fout, kernel_size, instance_norm, channels_last, rank, n_task)

        if self.learned_shortcut:
            self.conv_s = ConvLayer(self.fin, self.fout, 1, instance_norm, channels_last, rank, n_task)

    def forward(self, x, y):
        task_id = y[0]
        x_s = self._shortcut(x, y)
        dx = self.conv_0(actvn(x), y)
        if task_id>0:
            dx = self.conv_mix(actvn(dx), y)
        dx = self.conv_1(actvn(dx), y)
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x, y):
        if self.learned_shortcut:
            x_s = self.conv_s(x, y)
        else:
            x_s = x
        return x_s

class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.z_dim = z_dim
        self.num_layers = int(math.log(size, 2)) - 1

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        rank = kwargs['rank']
        n_task = nlabels

        self.resnet_0_0 = Gen_ResnetBlock(16*nf, 16*nf, rank=rank, n_task=n_task)
        self.resnet_1_0 = Gen_ResnetBlock(16*nf, 16*nf, rank=rank, n_task=n_task)
        self.resnet_2_0 = Gen_ResnetBlock(16*nf, 8*nf, rank=rank, n_task=n_task)
        self.resnet_3_0 = Gen_ResnetBlock(8*nf, 4*nf, rank=rank, n_task=n_task)
        self.resnet_4_0 = Gen_ResnetBlock(4*nf, 2*nf, rank=rank, n_task=n_task)
        self.resnet_5_0 = Gen_ResnetBlock(2*nf, 1*nf, rank=rank, n_task=n_task)
        self.resnet_6_0 = Gen_ResnetBlock(1*nf, 1*nf, rank=rank, n_task=n_task)
        self.conv_img = nn.Conv2d(nf, 3, 7, padding=3)


    def forward(self, z, y, return_feats=False):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)
        feats = []

        yembed = self.embedding(y)
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out, y)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out, y)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out, y)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out, y)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out, y)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out, y)
        if return_feats:
            feats.append(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_6_0(out, y)
        if return_feats:
            feats.append(out)

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
        self.num_layers = int(math.log(size, 2)) - 1

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
            [nn.Conv2d(4 * nf, 1, 3),
             nn.Conv2d(8 * nf, 1, 3),
             nn.Conv2d(16 * nf, 1, 3),
             nn.Conv2d(16 * nf, 1, 3)]
        )

        self.fc = nn.Linear(16*nf*s0*s0, nlabels) # (16384, nlabels) for 256x256 images

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
            feat = feats[2 + idx]  # 0<=idx<=3
            feat = feat[:batch_size // 2]  # only those for interpolated images
            pred = self.patch_conv[idx](feat)
            return pred, feats

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_6_0(out)

        out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out, feats

def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out