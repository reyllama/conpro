import torch
import os
import numpy
from gan_training.metrics import inception_score, FID

class Evaluator(object):
    def __init__(self, generator, zdist, ydist=None, batch_size=64, config=None, out_dir=None, device=None):
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = config['test']['inception_nsamples']
        self.batch_size = batch_size
        self.device = device
        self.is_conpro = ('conpro' in config['generator']['name'])
        self.config = config
        self.out_dir = out_dir
        self.iteration = 0
        self.inception_every = config['training']['inception_every']
        self.is_joint = ("joint" in config['data']['train_dir'])
        self.curBest = dict()
        for i in range(1, config['data']['nlabels']+1):
            self.curBest[i] = 10000

    def compute_fid(self, task_id):
        self.generator.eval()
        if self.is_joint:
            if task_id==0:
                self.iteration += self.inception_every
        else:
            self.iteration += self.inception_every
        imgs = []
        num_its = self.inception_nsamples // self.batch_size
        res = self.inception_nsamples - ( num_its * self.batch_size )
        for _ in range(num_its):
            ztest = self.zdist.sample((self.batch_size,))
            # ytest = self.ydist.sample((self.batch_size,))
            ytest = torch.ones([self.batch_size], dtype=torch.long) * task_id
            ytest = ytest.to(ztest.device)

            samples = self.generator(ztest, ytest)
            if isinstance(samples, tuple):
                samples = samples[0]
            # samples = [s.data.cpu().numpy() for s in samples]
            # imgs.extend(samples)
            imgs.append(samples.data) # batched images


        if res > 0:
            ztest = self.zdist.sample((res, ))
            ytest = torch.ones([res], dtype=torch.long) * task_id
            ytest = ytest.to(ztest.device)
            samples = self.generator(ztest, ytest)
            if isinstance(samples, tuple):
                samples = samples[0]
            imgs.append(samples.data)

        # imgs = imgs[:self.inception_nsamples]
        score = FID(
            imgs, task_id, self.config, device=self.device, resize=True, splits=10
        )

        if "joint" in self.config['data']['train_dir']:
            task_id += 1
        # if score < self.curBest[task_id]:
        #     out_dir = os.path.join(self.out_dir, 'imgs/samples', f"{task_id}")
        #     if not os.path.isdir(out_dir):
        #         os.makedirs(out_dir)
        #     arrz = torch.cat(imgs, dim=0).data.cpu().numpy()
        #     numpy.savez_compressed(os.path.join(out_dir, '%08d' % self.iteration), arrz)
            # self.curBest[task_id] = score

        self.curBest[task_id] = min(self.curBest[task_id], score)

        return score

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            # if self.is_conpro:
            #     x, _ = self.generator(z, y)
            # else:
            #     x = self.generator(z, y)
            x = self.generator(z, y)
            if isinstance(x, tuple):
                return x[0]
        return x
