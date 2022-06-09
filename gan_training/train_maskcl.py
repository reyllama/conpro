# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from torch import nn

# Define auxiliary functions for MDL computation
sfm = nn.Softmax(dim=1)
kl_loss = nn.KLDivLoss()
sim = nn.CosineSimilarity()

def generate_interp(generator, z, distribution):
    batch_size = z.size(0)
    device = z.device()
    alpha = distribution.sample((batch_size, )).to(device)
    z_interp = torch.matmul(alpha, z)
    interp_image = generator(z_interp)

    return interp_image

def mdl_d(discriminator, interp_image, fake_image, alpha): # TODO: alter discriminator so that it returns simlinear features
    """

    Args:
        discriminator: discriminator model
        interp_image: images generated from interpolated latent code (batch_size, 3, H, W)
        fake_image: images generated from original latent code (batch_size, 3, H, W)
        alpha: interpolation coefficients (batch_size, batch_size)

    Returns: adv_loss, mdl_loss

    """
    device = interp_image.device()
    target = sfm(alpha)
    input_image = torch.cat([interp_image, fake_image], dim=0)
    interp_pred, sim_pred = discriminator(input_image, mdl=True)        # TODO: implement discriminator mdl forward pass
    targets = torch.zeros_like(interp_pred, device=device)
    adv_loss = F.binary_cross_entropy_with_logits(interp_pred, targets)
    mdl_loss = kl_loss(torch.log(sim_pred), target)

    return adv_loss, mdl_loss

def mdl_g(generator, interp_feat, fake_feat, alpha):                    # TODO: alter generator so that it returns intermediate feature maps
    """

    Args:
        generator: generator model
        interp_feat: intermediate generator features of interpolated latents (batch_size, feat_dim)
        fake_feat: intermediate generator features of edge latents (batch_size, feat_dim)
        alpha: interpolation coefficients (batch_size, batch_size)

    Returns:

    """

    batch_size = fake_feat.size(0)

    dist_source = sfm(alpha)  # (2, K)

    # (Select layer idx to extract activation from)
    feat_ind = np.random.randint(1, generator.num_layers - 1, size=batch_size) # TODO: Implement generator attribute (generator.num_layers)

    # computing distances among target generations
    dist_target = torch.zeros([batch_size, batch_size]).cuda()

    # iterating over different elements in the batch
    for pair1 in range(batch_size):
        for pair2 in range(batch_size):
            anchor_feat = torch.unsqueeze(
                interp_feat[feat_ind[pair1]][pair1].reshape(-1), 0)
            compare_feat = torch.unsqueeze(
                fake_feat[feat_ind[pair1]][pair2].reshape(-1), 0)
            dist_target[pair1, pair2] = sim(anchor_feat, compare_feat)
    dist_target = sfm(dist_target)
    mdl_loss = kl_loss(torch.log(dist_target), dist_source)  # distance consistency loss

    return mdl_loss


"""
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):

        """

        Args:
            features: features of size (batch_size, n_views, ...)
            labels: labels for the current batch
            mask: employ mask instead of labels if labels not provided
            target_labels: labels for the "positive" samples (supCon)
            reduction: how to compute final loss

        Returns: SupConLoss

        """

        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = logits_mask.to(logits.device)
        mask = mask.to(logits.device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss

def d_logistic_loss(fake_pred):
    fake_loss = F.softplus(fake_pred)
    return fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    fake_loss = F.softplus(-fake_pred)
    return fake_loss.mean()

class Trainer(object):
    # def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
    #              gan_type, reg_type, reg_param, distribution, batch_size, mdl_d_wt, mdl_g_wt):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, batch_size, config):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.batch_size = batch_size
        self.supcon = SupConLoss(temperature=0.1, base_temperature=0.1)

        # self.gan_type = gan_type
        # self.reg_type = reg_type
        # self.reg_param = reg_param
        #
        # self.distribution_name = distribution
        #
        # self.mdl_d_wt = mdl_d_wt
        # self.mdl_g_wt = mdl_g_wt
        self.gan_type = config['training']['gan_type']
        self.reg_type = config['training']['reg_type']
        self.reg_param = config['training']['reg_param']
        self.distribution_name = config['training']['coef_distribution']
        self.mdl_d_wt = config['training']['mdl_d_wt']
        self.mdl_g_wt = config['training']['mdl_g_wt']
        self.supcon_wt = config['training']['supcon_wt']
        self.use_pretrain = config['training']['use_pretrain']

        # Todo: needs to be altered
        self.is_conpro = ('conpro' in config['generator']['name']) or ('mask' in config['generator']['name'])
        print(f"is_conpro: {self.is_conpro}")

        if self.distribution_name == 'dirichlet':
            self.distribution = torch.distributions.dirichlet.Dirichlet(torch.ones(batch_size), validate_args=None)
        elif self.distribution_name == 'uniform':
            self.distribution = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError

    def generator_trainstep(self, y, z):
        assert(y.size(0) == z.size(0))

        if self.use_pretrain:
            toggle_grad(self.generator, True, partial_update=True, task_id=y[0])
        else:
            if y[0]>1:
                toggle_grad(self.generator, True, partial_update=True, task_id=y[0])
            else:
                toggle_grad(self.generator, True)

        # if y[0]>0 and self.is_conpro: # From second task in continual learning setting
        #     toggle_grad(self.generator, True, partial_update=True, task_id=y[0])
        # else: # First task in CL setting
        #     toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        if self.is_conpro:
            x_fake, _ = self.generator(z, y)
            d_fake, _ = self.discriminator(x_fake, y) # TODO: Unconditional discriminator? Projection discriminator? Non-saturating loss?
        # In order to keep the baseline models as-is
        else:
            x_fake = self.generator(z, y)
            d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item()

    def compute_discriminator_supcon(self, images, labels):
        """

        Args:
            images: concatenation of real and generated (replay) images
            labels: class labels
        Returns: supcon loss

        """
        _, feats = self.discriminator(images, labels, mdl=True, idx=None) # feats = [batch_feat_0, ... , batch_feat_k]
        feat_ind = np.random.randint(3, self.discriminator.module.num_layers - 1) # exclusive of the upper bound (1,2,3,4,5)
        feat = feats[feat_ind]
        batch_size = feat.size(0) # this is not actually batch_size, but concatenation with replay samples
        feat = feat.view(batch_size, -1).unsqueeze(1) # (N, 1, feat_dim)
        feat = F.normalize(feat, dim=2)
        # print(f"feat_dim: {feat.size()}")
        # print(f"feat_ind: {feat_ind}")
        # print(f"labels: {labels}")
        # TODO: This can be asymmetric as well (target_labels=[int(y[0])])
        target_labels = list(range(int(labels[0])))
        # print(f"supcon target labels: {target_labels}")
        supcon_loss = self.supcon(feat, labels, target_labels=target_labels)
        return supcon_loss

    def discriminator_supcon(self, x_real, y, zdist):

        # x_real: real images
        # y: labels for x_real

        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        batch_size = x_real.size(0)

        x_real.requires_grad_()

        # On fake data
        past_cls = list(range(int(y[0])))
        # print(f"past_cls: {past_cls}")
        n_sample_past = max(batch_size // len(past_cls), 2)
        past_y = [label for label in past_cls for _ in range(n_sample_past)]
        past_y = torch.Tensor(past_y).to(y)
        # print(f"past_y: {past_y}")
        z = zdist.sample((past_y.size(0),))
        # print(f"z size: {z.size()}")
        with torch.no_grad():
            x_fake, _ = self.generator(z, past_y)

        x_fake.requires_grad_()

        model_device = next(self.generator.module.parameters()).device
        x_real = x_real.to(model_device)
        x_fake = x_fake.to(model_device)
        images = torch.cat([x_real, x_fake], dim=0)
        labels = torch.cat([y, past_y], dim=0)

        supcon_loss = self.compute_discriminator_supcon(images, labels)

        # print(f"supcon_loss: {supcon_loss.item()}")

        dloss_supcon = self.supcon_wt * supcon_loss
        dloss_supcon.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        return dloss_supcon.item()

    def discriminator_trainstep(self, x_real, y, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        if self.is_conpro:
            d_real, _ = self.discriminator(x_real, y)
        else:
            d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            if self.is_conpro:
                x_fake, _ = self.generator(z, y)
            else:
                x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        if self.is_conpro:
            d_fake, _ = self.discriminator(x_fake, y)
        else:
            d_fake = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        elif self.gan_type == 'non_saturating':
            loss = F.softplus(-d_out).mean() if target==1 else F.softplus(d_out).mean() # TODO: mean() aggregation necessary?
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out, _ = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def generate_interp(self, z, y, return_feats=False):
        batch_size = z.size(0)
        device = z.device
        alpha = self.distribution.sample((batch_size,)).to(device)
        z_interp = torch.matmul(alpha, z)
        interp_image, interp_feat = self.generator(z_interp, y, return_feats)
        fake_image, fake_feat = self.generator(z, y, return_feats)

        return fake_image, interp_image, fake_feat, interp_feat, alpha

    def compute_discriminator_mdl(self, interp_image, fake_image, y, alpha):
        batch_size = interp_image.size(0)
        device = interp_image.device
        dist_source = sfm(alpha)
        input_image = torch.cat([interp_image, fake_image], dim=0)

        feat_ind = np.random.randint(0, self.discriminator.module.num_layers - 3) # [1,2,3] -> [0,1,2,3], for patch discrimination

        interp_pred, feat = self.discriminator(input_image, y, mdl=True, idx=feat_ind) # alter this part so that discriminator returns indexed feature (feat_ind + 2)
        # interp_feat, fake_feat = feats[:batch_size], feats[batch_size:]
        # interp_feat, fake_feat = [feat[:batch_size] for feat in feats], [feat[batch_size:] for feat in feats]
        interp_feat, fake_feat = feat[:batch_size], feat[batch_size:]
        # targets = torch.zeros_like(interp_pred, device=device)
        # adv_loss = F.binary_cross_entropy_with_logits(interp_pred, targets)
        # adv_loss = self.compute_loss(interp_pred, 0)
        adv_loss = d_logistic_loss(interp_pred) # TODO: Note I arbitrarily applied logistic loss (patch discrimination) here, might be a problem

        # feat_ind = np.random.randint(1, self.discriminator.module.num_layers - 1, size=batch_size) # [1,2,3,4,5], for MDL (pairwise feature similarity)

        # computing distances among target generations
        # dist_target = torch.zeros([batch_size, batch_size]).cuda()

        interp_feat = interp_feat.view(batch_size, -1).unsqueeze(2)
        fake_feat = fake_feat.view(batch_size, -1).unsqueeze(0)

        dist_target = sim(interp_feat, fake_feat)
        dist_target = sfm(dist_target)

        # for pair1 in range(batch_size):
        #     for pair2 in range(batch_size):
        #         anchor_feat = torch.unsqueeze(
        #             interp_feat[feat_ind[pair1]][pair1].reshape(-1), 0)
        #         compare_feat = torch.unsqueeze(
        #             fake_feat[feat_ind[pair1]][pair2].reshape(-1), 0)
        #         dist_target[pair1, pair2] = sim(anchor_feat, compare_feat)
        #
        # dist_target = sfm(dist_target)
        mdl_loss = kl_loss(torch.log(dist_target), dist_source)

        return adv_loss, mdl_loss

    def compute_generator_mdl(self, interp_image, interp_feat, fake_feat, y, alpha):
        batch_size = fake_feat[0].size(0)
        device = interp_image.device
        dist_source = sfm(alpha)  # (2, K)

        # (Select layer idx to extract activation from)
        feat_ind = np.random.randint(1, self.generator.module.num_layers-2)

        interp_feat, fake_feat = interp_feat[feat_ind], fake_feat[feat_ind]
        interp_feat = interp_feat.view(batch_size, -1).unsqueeze(2)
        fake_feat = fake_feat.view(batch_size, -1).unsqueeze(0)
        dist_target = sim(interp_feat, fake_feat)
        dist_target = sfm(dist_target)

        # computing distances among target generations
        # dist_target = torch.zeros([batch_size, batch_size]).cuda()

        # iterating over different elements in the batch
        # for pair1 in range(batch_size):
        #     for pair2 in range(batch_size):
        #         anchor_feat = torch.unsqueeze(
        #             interp_feat[feat_ind[pair1]][pair1].reshape(-1), 0)
        #         compare_feat = torch.unsqueeze(
        #             fake_feat[feat_ind[pair1]][pair2].reshape(-1), 0)
        #         dist_target[pair1, pair2] = sim(anchor_feat, compare_feat)
        # dist_target = sfm(dist_target)
        mdl_loss = kl_loss(torch.log(dist_target), dist_source)  # distance consistency loss

        feat_ind = np.random.randint(0, self.discriminator.module.num_layers - 3)

        d_fake, _ = self.discriminator(interp_image, y, mdl=True, idx=feat_ind)
        adv_loss = g_nonsaturating_loss(d_fake)
        # targets = torch.zeros_like(d_fake, device=device)
        # adv_loss = F.binary_cross_entropy_with_logits(d_fake, targets)
        # gloss = self.compute_loss(d_fake, 1)                                          # TODO: This can be an option if patch discriminator complies to this class method

        return adv_loss, mdl_loss

    def discriminator_mdl(self, x_real, y, z):

        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real, _ = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        dloss_real.backward()

        # On fake data
        with torch.no_grad():
            x_fake, x_interp, _f, _i, alpha = self.generate_interp(z, y)

        x_fake.requires_grad_()
        x_interp.requires_grad_()

        adv_dloss, mdl_dloss = self.compute_discriminator_mdl(x_interp, x_fake, y, alpha)

        dloss_fake = adv_dloss + self.mdl_d_wt * mdl_dloss
        dloss_fake.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return (dloss_real+adv_dloss).item(), (self.mdl_d_wt*mdl_dloss).item()

    def generator_mdl(self, y, z):
        assert (y.size(0) == z.size(0))
        if self.use_pretrain:
            toggle_grad(self.generator, True, partial_update=True, task_id=y[0])
        else:
            if y[0] > 1:
                toggle_grad(self.generator, True, partial_update=True, task_id=y[0])
            else:
                toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        fake_image, interp_image, fake_feat, interp_feat, alpha = self.generate_interp(z, y, return_feats=True)
        adv_loss, mdl_gloss = self.compute_generator_mdl(interp_image, interp_feat, fake_feat, y, alpha)
        gloss = adv_loss + self.mdl_g_wt * mdl_gloss
        gloss.backward()

        self.g_optimizer.step()

        return adv_loss.item(), (self.mdl_g_wt*mdl_gloss).item()

# Utility functions
def toggle_grad(model, requires_grad, partial_update=False, task_id=None, n_task=7):
    if not partial_update:
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    # update plasticity blocks only
    else:
        for name, param in model.named_parameters():
            if name.find('mask') >= 0:
                param.requires_grad = requires_grad
            # elif name.find('embedding') >= 0:
            #     param.requires_grad = requires_grad
            # elif name.find('fc') >= 0:
            #     param.requires_grad = requires_grad
            else:
                param.requires_grad = False



def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
