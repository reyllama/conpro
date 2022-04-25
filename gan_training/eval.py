import torch
from gan_training.metrics import inception_score, FID

class Evaluator(object):
    def __init__(self, generator, zdist, ydist, batch_size=64, config=None, device=None):
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = config['test']['inception_nsamples']
        self.batch_size = batch_size
        self.device = device
        self.is_conpro = ('conpro' in config['generator']['name'])
        self.config = config

    def compute_fid(self, task_id):
        self.generator.eval()
        imgs = []
        num_its = self.inception_nsamples // self.batch_size
        res = self.inception_nsamples - ( num_its * self.batch_size )
        for _ in range(num_its):
            ztest = self.zdist.sample((self.batch_size,))
            # ytest = self.ydist.sample((self.batch_size,))
            ytest = torch.ones([self.batch_size], dtype=torch.long) * task_id
            ytest = ytest.to(ztest.device)

            samples = self.generator(ztest, ytest)
            # samples = [s.data.cpu().numpy() for s in samples]
            # imgs.extend(samples)
            imgs.append(samples.data.cpu().numpy()) # batched images

        ztest = self.zdist.sample((res, ))
        ytest = torch.ones([res], dtype=torch.long) * task_id
        ytest = ytest.to(ztest.device)
        samples = self.generator(ztest, ytest)
        imgs.append(samples.data.cpu().numpy())

        # imgs = imgs[:self.inception_nsamples]
        score = FID(
            imgs, task_id, self.config, device=self.device, resize=True, splits=10
        )

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
