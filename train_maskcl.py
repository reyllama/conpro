import argparse
import os
import shutil
from os import path
import numpy as np
import time
import copy
import torch
from torch import nn
from tqdm import tqdm
from gan_training import utils
from gan_training.train_maskcl import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)

# Initialize parameters with the most similar previous task
def init_mask_weights(model, target, cur_task_id, rel_task_id):
    for module in model.children():
        if module.__class__.__name__ == target:
            for child in module.children():
                for name, param in child.named_parameters():
                    if name.find("weight_mask") >= 0:
                        with torch.no_grad():
                            if rel_task_id == -1:
                                pass
                            else:
                                getattr(child, name)[cur_task_id] = nn.Parameter(getattr(child, name)[rel_task_id].data)

# Clamp parameters in a given range
def clamp_weights(model, target, min_val, max_val):
    for name, param in model.named_parameters():
            if name.find(target) >= 0:
                name = name.split('.')[-1]
                with torch.no_grad():
                    try:
                        getattr(model, name).clamp_(min_val, max_val)
                    except:
                        getattr(model.module, name).clamp_(min_val, max_val)

# Control / clear gradients under various running_mean optimizations (momentum, etc.)
def control_gradients(model, target, idx=[], requires_grad=False):
    for module in model.module.children():
        if module.__class__.__name__ == target:
            for child in module.children():
                for i in idx:
                    child[i].weight.requires_grad = requires_grad
                    child[i].weight.grad = None
                    try:
                        child[i].bias.requires_grad = requires_grad
                        child[i].bias.grad = None
                    except:
                        pass

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
# print(f"is_cuda: {is_cuda}")

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
g_steps = config['training']['g_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_nlabels = config['training']['sample_nlabels']

###### misc.details ######
exp_setting = config['training']['misc'] # For free form code level changes that are not contained in config.yaml
##########################

# Compose experiment output directory
out_dir = config['training']['out_dir']
exp_name = config['generator']['name']+'_'+ str(config['generator']['kwargs']['nfilter']) +'_' \
            + str(config['generator']['kwargs']['embed_size'])
exp_name += '_img_' + str(config['data']['img_size']) + "_mdl_" + str(config['training']['mdl_every']) + "_supcon_" + str(config['training']['supcon_every']) +\
            f"_{str(config['training']['mdl_g_wt'])}_{str(config['training']['mdl_d_wt'])}" + "_dstep_" + str(config['training']['d_steps']) + \
    "_gstep_" + str(config['training']['g_steps']) + "_rank_" + str(config['generator']['kwargs']['rank'])
if exp_setting:
    exp_name += f"_misc_{exp_setting}"

out_dir = path.join(out_dir, exp_name)
checkpoint_dir = path.join(out_dir, 'ckpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# save config snapshot for reproducibility
shutil.copy(args.config, out_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")

# whether to start from a pretrained weights
try:
    DATA_FIX = config['data']['data_fix']
    load_dir = config['data']['pretrain_dir']
except:
    DATA_FIX = None
    load_dir = None

# Create models
generator, discriminator = build_models(config)

# Start from pretrained model
if DATA_FIX and load_dir:
    print("Loading pretrained weights...!")
    dict_G = torch.load(load_dir + DATA_FIX + 'Pre_generator')
    generator = utils.attach_partial_params(generator, dict_G)
    # generator = load_model_norm(generator)
    dict_D = torch.load(load_dir + DATA_FIX + 'Pre_discriminator')
    discriminator = utils.attach_partial_params(discriminator, dict_D)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, config
)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
)

# Get model file
model_file = config['training']['model_file']

# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)

# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Train
tstart = t0 = time.time()

# Load checkpoint if it exists
try:
    load_dict = checkpoint_io.load(model_file)
except FileNotFoundError:
    it = epoch_idx = -1
else:
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    logger.load_stats('stats.p')

# Reinitialize model average if needed
if (config['training']['take_model_average']
        and config['training']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate annealing
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
# trainer = Trainer(
#     generator, discriminator, g_optimizer, d_optimizer,
#     gan_type=config['training']['gan_type'],
#     reg_type=config['training']['reg_type'],
#     reg_param=config['training']['reg_param'],
#     distribution=config['training']['coef_distribution'],
#     batch_size=batch_size,                                  # TODO: check if this part works well (i.e., drop_last or sth)
#     mdl_d_wt = config['training']['mdl_d_wt'],
#     mdl_g_wt = config['training']['mdl_g_wt']
# )

# simplify arguments
trainer = Trainer(
    generator, discriminator, g_optimizer, d_optimizer,
    batch_size=batch_size,
    config=config
)


# shorthands
n_epoch = config['training']['n_epoch']
n_task = config['training']['n_task']
mdl_every = config['training']['mdl_every']
supcon_every = config['training']['supcon_every']

# caching previous task indices
past_tasks = []

# Task index depends on the pretraining configuration
if config['training']['use_pretrain']:
    task_range = range(1, n_task+1)
else:
    task_range = range(n_task)

for task_id in task_range:
    print(f"Start training for task {task_id}...!")
    # Dataset
    train_dataset, nlabels = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'][f'task_{task_id}'],
        size=config['data']['img_size'],
        lsun_categories=config['data']['lsun_categories_train']
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=False, sampler=None, drop_last=True
    )

    # Number of labels
    # nlabels = min(nlabels, config['data']['nlabels'])
    nlabels = task_id + 1
    sample_nlabels = nlabels

    # Distributions
    ydist = get_ydist(nlabels, device=device)                               # TODO: Implement so that sample from past tasks -> clear
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Save for tests
    ntest = 16
    x_real, ytest = utils.get_nsamples(train_loader, ntest)
    ytest.clamp_(None, nlabels - 1)
    ztest = zdist.sample((ntest,))
    utils.save_images(x_real, path.join(out_dir, f'real_{task_id}.png'))

    # Evaluator
    evaluator = Evaluator(generator_test, zdist, ydist,
                          batch_size=batch_size, config=config, device=device)

    # y_inst = 0
    # x = evaluator.create_samples(ztest, y_inst)
    # print(x[0].min(), x[0].max())
    # logger.add_imgs(x, '%04d' % y_inst, -1)

    # From second task
    if task_id > 0:
        # print(f"past_tasks: {past_tasks}")
        # TODO
        # z = zdist.sample((batch_size,))
        # real_cur, _ = next(train_loader)
        # dists = np.zeros(task_id)
        # for prev_task_id in range(task_id):
        #     y0 = torch.ones([batch_size], dtype=torch.long) * prev_task_id
        #     gen_replay, _ = generator(z, y0)
        #     cross_dist = discriminator.evaluate_distance(gen_replay, real_cur) # TODO: implement evaluate_distance function in our discriminator
        #     dists[prev_task_id] = cross_dist.item()
        # rel_task_id = np.argmin(dists)
        #
        # # initialize model parameters with the most similar previous task
        if config['training']['use_pretrain']:
            init_mask_weights(generator.module, 'Gen_ResnetBlock', task_id-1, task_id-2) # TODO: modify with distance learning
        else:
            init_mask_weights(generator.module, 'Gen_ResnetBlock', task_id, task_id-1) # TODO: modify with distance learning
        # control_gradients(generator, 'task_ResnetBlock', past_tasks, False)

        n_epoch = int(config['training']['n_epoch'] * config['training']['n_epoch_factor']) # For incorporated base training

    for epoch_idx in range(n_epoch):
        # epoch_idx += 1
        # print('Start epoch %d...' % epoch_idx)

        for x_real, _ in train_loader:
            it += 1
            g_scheduler.step()
            d_scheduler.step()

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']
            logger.add('learning_rates', 'discriminator', d_lr, it=it)
            logger.add('learning_rates', 'generator', g_lr, it=it)

            # x_real, y = x_real.to(device), y.to(device)
            x_real = x_real.to(device)

            y = torch.ones([batch_size], dtype=torch.long) * task_id
            y = y.to(device)

            y.clamp_(None, nlabels-1) # nlabels = task_id + 1

            if ((it + 1) % d_steps) == 0:
                z = zdist.sample((batch_size,))

                # Alternate between regular train step and MDL step

                if (it+1) % supcon_every == 0:
                    supcon_loss = trainer.discriminator_supcon(x_real, y, zdist)
                    logger.add('losses', 'supcon', supcon_loss, it=it)

                if (it+1) % mdl_every == 0:
                    dloss, mdl_dloss = trainer.discriminator_mdl(x_real, y, z)
                    logger.add('losses', 'discriminator', dloss, it=it)
                    logger.add('losses', 'mdl-d', mdl_dloss, it=it)

                # Regular discriminator updates
                else:
                    dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
                    logger.add('losses', 'discriminator', dloss, it=it)
                    logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            if ((it + 1) % g_steps) == 0:
                z = zdist.sample((batch_size,))

                if (it+1) % mdl_every == 0:
                    gloss, mdl_gloss = trainer.generator_mdl(y, z)
                    logger.add('losses', 'generator', gloss, it=it)
                    logger.add('losses', 'mdl-g', mdl_gloss, it=it)
                else:
                    gloss = trainer.generator_trainstep(y, z)
                    logger.add('losses', 'generator', gloss, it=it)

                if config['training']['take_model_average']:
                    update_average(generator_test, generator, beta=config['training']['model_average_beta'])

            # Print stats
            g_loss_last = logger.get_last('losses', 'generator')
            mdl_g_loss_last = logger.get_last('losses', 'mdl-g')
            d_loss_last = logger.get_last('losses', 'discriminator')
            mdl_d_loss_last = logger.get_last('losses', 'mdl-d')
            supcon_last = logger.get_last('losses', 'supcon')
            d_reg_last = logger.get_last('losses', 'regularizer')
            if it % 1 == 0:
                print('[epoch %0d, it %4d] g_loss = %.3f, d_loss = %.3f, mdl_g = %.3f, mdl_d = %.3f, supcon = %.3f, reg=%.3f'
                    % (epoch_idx, it, g_loss_last, d_loss_last, mdl_g_loss_last, mdl_d_loss_last, supcon_last, d_reg_last))

            # clamp_weights(generator, 'mixing', 0.2, 0.8)
            # print(getattr(generator.module, 'mixing_0_pls'))
            # print(y)
            # print(generator.module.resnet_1_0.conv_0.weight.sum().item())
            # print(generator.module.resnet_1_pls.conv_0[1].weight.sum().item())
            # print('-'*30)

            # (i) Sample if necessary
            if (it % config['training']['sample_every']) == 0:
                print('Creating samples...')
                x = evaluator.create_samples(ztest, ytest)                  # TODO: modify evaluator.create_samples() so that the output grid is more rectangular
                logger.add_imgs(x, 'all', it)
                for y_inst in range(sample_nlabels):
                    # print(f"task_id: {task_id}")
                    # print(f"sample_nlabels: {sample_nlabels}")
                    # print(f"y_inst: {y_inst}")
                    x = evaluator.create_samples(ztest, y_inst)
                    logger.add_imgs(x, '%04d' % y_inst, it)

            # (ii) Compute inception if necessary
            if inception_every > 0 and ((it + 1) % inception_every) == 0:
                inception_mean, inception_std = evaluator.compute_inception_score()
                logger.add('inception_score', 'mean', inception_mean, it=it)
                logger.add('inception_score', 'stddev', inception_std, it=it)

            # (iii) Backup if necessary
            if ((it + 1) % backup_every) == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it)
                logger.save_stats('stats_%08d.p' % it)

            # (iv) Save checkpoint if necessary
            if time.time() - t0 > save_every:
                print('Saving checkpoint...')
                checkpoint_io.save(model_file, it=it)
                logger.save_stats('stats.p')
                t0 = time.time()

                if (restart_every > 0 and t0 - tstart > restart_every):
                    exit(3)

    past_tasks.append(task_id)
