import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn
import shutil
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
print(f"is_cuda: {is_cuda}")

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
g_steps = config['training']['g_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_nlabels = config['training']['sample_nlabels']

out_dir = config['training']['out_dir']

exp_name = "mdl_" + str(config['training']['mdl_every']) + f"_{config['training']['mdl_d_wt']}_{config['training']['mdl_g_wt']}" + \
            "_supcon_" + str(config['training']['supcon_every']) + f"_{config['training']['supcon_wt']}"

exp_name += config['training']['misc']
out_dir = path.join(out_dir, exp_name)
checkpoint_dir = path.join(out_dir, 'chkpts')

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

# whether to start from a pretrained weights
try:
    DATA_FIX = config['data']['data_fix']
    load_dir = config['data']['pretrain_dir']
except:
    DATA_FIX = None
    load_dir = None

# Dataset
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    lsun_categories=config['data']['lsun_categories_train']
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=False, sampler=None, drop_last=True
)
try:
    print("CLASS MAPPING: ", train_dataset.class_to_idx)
except:
    pass

# Number of labels
nlabels = min(nlabels, config['data']['nlabels'])
sample_nlabels = min(nlabels, sample_nlabels)

# Create models
generator, discriminator = build_models(config)
# print(generator)
# print(discriminator)
num_params = sum(x.numel() for x in generator.parameters())
print('GENERATOR PARAMETERS: ', num_params)

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

# Distributions
print(f"** N_LABELS: {nlabels} **")
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)

# Save for tests
ntest = batch_size
x_real, ytest = utils.get_nsamples(train_loader, ntest)
ytest.clamp_(None, nlabels-1)
ztest = zdist.sample((ntest,))
utils.save_images(x_real, path.join(out_dir, 'real.png'))

# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Evaluator
evaluator = Evaluator(generator_test, zdist, ydist,
                      batch_size=batch_size, config=config, out_dir=out_dir, device=device)

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

logger.add('Generator', 'num_params', num_params, it=0)

# Training loop
print('Start training...')

for epoch_idx in range(n_epoch):
    # epoch_idx += 1
    # print('Start epoch %d...' % epoch_idx)

    for x_real, y in train_loader:
        it += 1
        g_scheduler.step()
        d_scheduler.step()

        if it == 0:
            print("LABEL example: ", y)

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'discriminator', d_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # Discriminator updates
        if ((it + 1) % d_steps) == 0:
            z = zdist.sample((batch_size,))

            if supcon_every > 0 and (it + 1) % supcon_every == 0:
                supcon_loss = trainer.discriminator_supcon(x_real, y, zdist)
                logger.add('losses', 'supcon', supcon_loss, it=it)

            if mdl_every > 0 and (it + 1) % mdl_every == 0:
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

            if mdl_every > 0 and (it + 1) % mdl_every == 0:
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
            print(
                        '[epoch %0d, it %4d] g_loss = %.3f, d_loss = %.3f, mdl_g = %.3f, mdl_d = %.3f, supcon = %.3f, reg=%.3f'
                        % (epoch_idx, it, g_loss_last, d_loss_last, mdl_g_loss_last, mdl_d_loss_last, supcon_last,
                           d_reg_last))

        # (i) Sample if necessary
        if (it % config['training']['sample_every']) == 0:
            print('Creating samples...')
            # x = evaluator.create_samples(ztest, ytest)
            # logger.add_imgs(x, 'all', it)
            for y_inst in range(sample_nlabels):
                x = evaluator.create_samples(ztest, y_inst)
                logger.add_imgs(x, '%04d' % y_inst, it)

        # (ii) Compute inception if necessary
        if inception_every > 0 and ((it + 1) % inception_every) == 0:
            # inception_mean, inception_std = evaluator.compute_inception_score()
            print("Calculating FID...")
            if "uncon" in config['data']['train_dir']:
                # TODO: this code is pathetic!
                # TODO: Hard coded........
                fid = evaluator.compute_fid(3)
                logger.add('FID', 'score', fid, it=it)
            else:
                fids = []
                for cls in range(0, 7):
                    fid = evaluator.compute_fid(cls)
                    fids.append(fid)
                print(f"FID: {fids}")
            # logger.add('inception_score', 'mean', inception_mean, it=it)
            # logger.add('inception_score', 'stddev', inception_std, it=it)
                logger.add('FID', 'score', fids, it=it)

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

with open(path.join(out_dir, "final_result.txt"), 'w') as f:
    for k, v in evaluator.curBest.items():
        f.write(f"{k}: {v:.2f}")
        f.write('\n')