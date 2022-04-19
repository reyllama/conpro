from gan_training.models import (
    resnet, resnet2, resnet3, resnet4, resnet_conpro, resnet4_uncond, resnet_mask, resnet4_cam
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet3': resnet3.Generator,
    'resnet4': resnet4.Generator,
    'resnet_conpro': resnet_conpro.Generator,
    'resnet4_uncond': resnet4_uncond.Generator,
    'resnet_mask': resnet_mask.Generator,
    'resnet4_cam': resnet4_cam.Generator
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet3': resnet3.Discriminator,
    'resnet4': resnet4.Discriminator,
    'resnet_conpro': resnet_conpro.Discriminator,
    'resnet4_uncond': resnet4_uncond.Discriminator,
    'resnet_mask': resnet_mask.Discriminator,
    'resnet4_cam': resnet4_cam.Discriminator
}
