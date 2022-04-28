import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)

# Arguments
parser = argparse.ArgumentParser(
    description='Test a trained GAN and create visualizations.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
img_dir = path.join(out_dir, 'test', 'img')
img_all_dir = path.join(out_dir, 'test', 'img_all')

# Creat missing directories
if not path.exists(img_dir):
    os.makedirs(img_dir)
if not path.exists(img_all_dir):
    os.makedirs(img_all_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

# Get model file
model_file = config['test']['model_file']

# Models
device = torch.device("cuda:0" if is_cuda else "cpu")

generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)

# Evaluator
evaluator = Evaluator(generator_test, zdist, ydist,
                      batch_size=batch_size, device=device)

# Load checkpoint if existant
load_dict = checkpoint_io.load(model_file)
it = load_dict.get('it', -1)
epoch_idx = load_dict.get('epoch_idx', -1)

# Inception score
if config['test']['compute_inception']:
    print('Computing inception score...')
    inception_mean, inception_std = evaluator.compute_inception_score()
    print('Inception score: %.4f +- %.4f' % (inception_mean, inception_std))


def create_samples(generator, zdist, y, batch_size, n_samples, img_dir=img_dir):
    generator.eval()
    if isinstance(y, int):
        y = torch.full((batch_size,), y, device=generator.device, dtype=torch.int64)
    its = n_samples // batch_size
    res = n_samples - (its * batch_size)
    tick = 1
    for _ in range(its):
        z = zdist.sample((batch_size,))
        z = z.to(device=y.device)

        with torch.no_grad():
            x = generator(z, y)
            if isinstance(x, tuple):
                x = x[0]
            for i in range(len(x)):
                utils.save_images(x[i], path.join(img_dir, str(int(y[0])), '%06d.png' % tick), nrow=1) # TODO; check img_dir
                tick += 1
    if res > 0:
        z = zdist.sample((res,))
        z = z.to(device=y.device)

        with torch.no_grad():
            x = generator(z, y)
            if isinstance(x, tuple):
                x = x[0]
            for i in range(len(x)):
                utils.save_images(x[i], path.join(img_dir, str(int(y[0])), '%06d.png' % tick), nrow=1)  # TODO; check img_dir

# Samples
print('Creating samples...')
for y in range(1, 8):
    create_samples(generator_test, zdist, y, batch_size, 5000)