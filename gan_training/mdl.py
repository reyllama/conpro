import torch
import torch.nn.functional as F
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
    interp_pred, sim_pred = discriminator(input_image, mdl=True) # TODO: implement discriminator mdl forward pass
    targets = torch.zeros_like(interp_pred, device=device)
    adv_loss = F.binary_cross_entropy_with_logits(interp_pred, targets)
    mdl_loss = kl_loss(torch.log(sim_pred), target)

    return adv_loss, mdl_loss

def mdl_g(generator, interp_feat, fake_feat, alpha): # TODO: alter generator so that it returns intermediate feature maps
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