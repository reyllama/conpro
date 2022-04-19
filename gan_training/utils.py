
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import math

def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    while n < N:
        x_next, y_next = next(iter(data_loader))
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def attach_partial_params(model, dict):
    model_dict = model.state_dict()

    if torch.is_tensor(model_dict.get('embedding.weight', None)):
        online, prtrn = model_dict['embedding.weight'].size(), dict['embedding.weight'].size()
        if prtrn[0] < online[0]:
            assert prtrn[1]==online[1], "the embedding dimension should match"
            n_labels, emb_dim = online[0], online[1]
            prtrn_labels = prtrn[0]
            padding = torch.randn((n_labels-prtrn_labels, emb_dim), dtype=dict['embedding.weight'].dtype, device=dict['embedding.weight'].device)
            dict['embedding.weight'] = torch.cat((dict['embedding.weight'], padding), dim=0)
            assert dict['embedding.weight'].size(0) == model_dict['embedding.weight'].size(0), "n_labels should match for embeddings"

    if torch.is_tensor(model_dict.get('fc.weight', None)):
        online, prtrn = model_dict['fc.weight'].size(), dict['fc.weight'].size()
        # adapting Celeb-A discriminator to continual setting
        if prtrn[0] < online[0]:
            assert prtrn[1]==online[1], "feature dimension for fc layer should match"
            n_labels, feature_dim = online[0], online[1]
            prtrn_labels = prtrn[0]
            # weights (last fc layer)
            padding = torch.rand((n_labels-prtrn_labels, feature_dim), dtype=dict['fc.weight'].dtype, device=dict['fc.weight'].device)
            padding = 2*(padding-0.5) / math.sqrt(online[1]) # properly initialize
            dict['fc.weight'] = torch.cat((dict['fc.weight'], padding), dim=0)
            # bias (last fc layer)
            padding = torch.rand((n_labels-prtrn_labels), dtype=dict['fc.bias'].dtype, device=dict['fc.bias'].device)
            padding = 2 * (padding - 0.5) / math.sqrt(online[1])  # properly initialize
            dict['fc.bias'] = torch.cat((dict['fc.bias'], padding), dim=0)
            assert dict['fc.weight'].size(0) == model_dict['fc.weight'].size(0), "n_labels should match for fc weight"
            assert dict['fc.bias'].size(0) == model_dict['fc.bias'].size(0), "n_labels should match for fc bias"

    model_dict.update(dict)
    model.load_state_dict(model_dict)
    return model
