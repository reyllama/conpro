# Conservative Generator, Progressive Discriminator

This repository provides the code base for ConPro, a framework for incremental few-shot learning of GANs.

<!-- This repository contains the experiments in the supplementary material for the paper [Which Training Methods for GANs do actually Converge?](https://avg.is.tuebingen.mpg.de/publications/meschedericml2018).

To cite this work, please use
```
@INPROCEEDINGS{Mescheder2018ICML,
  author = {Lars Mescheder and Sebastian Nowozin and Andreas Geiger},
  title = {Which Training Methods for GANs do actually Converge?},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2018}
}
```
You can find further details on [our project page](https://avg.is.tuebingen.mpg.de/research_projects/convergence-and-stability-of-gan-training). -->

# Usage
First download your data and put it into the `./data` folder. We mainly use [Animal-Face Dataset](https://data.mendeley.com/datasets/z3x59pv4bz/3).

Note that we simulate incremental few-shot setting by sampling 7 random classes (Bear, Cat, Cow, Deer, Elephant, Lion, Wolf).

Arrange your dataset sub-directories and pass the structure in to the `configs/config.yaml` under `data/train_dir`.

To train a new model, first create a config script similar to the ones provided in the `./configs` folder.  You can then train ConPro model using
```
python train_masckcl.py PATH_TO_CONFIG
```
During the training process, FID is computed against the real images, in which case you are recommended to precompute the inception features using `calc_inception.py`.

To compute `LPIPS` score, run `compute_lpips.py` with PATH to the generated samples and other arguments (e.g., n_sample, n_run) provided.

Finally, you can create nice latent space interpolations using
```
python interpolate.py PATH_TO_CONFIG
```
or
```
python interpolate_class.py PATH_TO_CONFIG
```

Our main configuration is provided in `configs/conpro_animalface.yaml`, where important hyperparameters such as AFM factorization rank, mdl and supcon frequency, relative loss weights can be controlled.
