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
from gan_training.train_cl import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)

# Initialize parameters with the most similar previous task
def init_weights(model, target, cur_task_id, rel_task_id):
    for module in model.children():
        if module.__class__.__name__ == target:
            for child in module.children():
                with torch.no_grad():
                    child[cur_task_id].weight = nn.Parameter(child[rel_task_id].weight.data)
                    try:
                        child[cur_task_id].bias = nn.Parameter(child[rel_task_id].bias.data)
                    except:
                        pass

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
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_nlabels = config['training']['sample_nlabels']

out_dir = config['training']['out_dir']
exp_name = config['generator']['name']+'_'+ str(config['generator']['kwargs']['nfilter']) +'_' \
           + str(config['generator']['kwargs']['nfilter_max']) + '_' + str(config['generator']['kwargs']['embed_size'])
exp_name += '_' + str(config['data']['img_size']) + '_' + str(config['data']['nlabels'])
out_dir = path.join(out_dir, exp_name)
checkpoint_dir = path.join(out_dir, 'ckpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

shutil.copy(args.config, out_dir) # copy config file to the experiment output

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")

# Create models
generator, discriminator = build_models(config)
# print(generator)
# print(discriminator)

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

trainer = Trainer(
    generator, discriminator, g_optimizer, d_optimizer,
    batch_size=batch_size,                                  # TODO: check if this part works well (i.e., drop_last or sth)
    config=config
)

# Training loop
print('Start training...')

n_epoch = config['training']['n_epoch']
n_task = config['training']['n_task']
mdl_every = config['training']['mdl_every']
supcon_every = config['training']['supcon_every']

past_tasks = []

for task_id in range(n_task):
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
    # sample_nlabels =

    # Distributions
    ydist = get_ydist(nlabels, device=device)                               # TODO: Implement so that sample from past tasks
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Save for tests
    ntest = batch_size
    x_real, ytest = utils.get_nsamples(train_loader, ntest)
    ytest.clamp_(None, nlabels - 1)
    ztest = zdist.sample((ntest,))
    utils.save_images(x_real, path.join(out_dir, f'real_{task_id}.png'))

    # Evaluator
    evaluator = Evaluator(generator_test, zdist, ydist,
                          batch_size=batch_size, config=config, device=device)

    # From second task
    print("is_conpro: ", trainer.is_conpro)
    if task_id > 0:
        if trainer.is_conpro:
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
            rel_task_id = 0
            init_weights(generator.module, 'task_ResnetBlock', task_id, rel_task_id) # TODO: modify with distance learning
            # control_gradients(generator, 'task_ResnetBlock', past_tasks, False)

        n_epoch = config['training']['n_epoch'] // 4

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

            y.clamp_(None, nlabels-1)
            z = zdist.sample((batch_size,))

            # Alternate between regular train step and MDL step
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
            if ((it + 1) % d_steps) == 0:
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
            d_reg_last = logger.get_last('losses', 'regularizer')
            print('[epoch %0d, it %4d] g_loss = %.3f, d_loss = %.3f, mdl_g = %.3f, mdl_d = %.3f, reg=%.3f'
                  % (epoch_idx, it, g_loss_last, d_loss_last, mdl_g_loss_last, mdl_d_loss_last, d_reg_last))

            clamp_weights(generator, 'mixing', 0.2, 0.8)
            # print(getattr(generator.module, 'mixing_0_pls'))
            # print(y)
            # print(generator.module.resnet_1_0.conv_0.weight.sum().item())
            # print(generator.module.resnet_1_pls.conv_0[1].weight.sum().item())
            # print('-'*30)

            # (i) Sample if necessary
            if (it % config['training']['sample_every']) == 0:
                print('Creating samples...')
                x = evaluator.create_samples(ztest, ytest)                  # TODO: evaluator.create_samples() so that the output grid is more rectangular
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