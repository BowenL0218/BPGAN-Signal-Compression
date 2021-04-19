### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .Bpgan_VGG_Extractor import Bpgan_VGGLoss


class Bpgan_GAN_DIS_Model(BaseModel):
    def name(self):
        return 'Bpgan_GAN_Distillation_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc

        ##### define networks
        # Generator network
        netE_input_nc = input_nc
        self.netE = networks.define_E(input_nc=netE_input_nc, ngf=opt.ngf, n_downsample=opt.n_downsample_global,
                                      C_channel=opt.C_channel, norm=opt.norm, gpu_ids=self.gpu_ids,
                                      one_D_conv=opt.OneDConv, one_D_conv_size=opt.OneDConv_size, max_ngf=opt.max_ngf,
                                      Conv_type=opt.Conv_type)
        self.netDecoder = networks.define_Decoder(output_nc=opt.output_nc, ngf=opt.ngf,
                                                  n_downsample=opt.n_downsample_global, C_channel=opt.C_channel,
                                                  n_blocks_global=opt.n_blocks_global, norm=opt.norm,
                                                  gpu_ids=self.gpu_ids, one_D_conv=opt.OneDConv,
                                                  one_D_conv_size=opt.OneDConv_size, max_ngf=opt.max_ngf,
                                                  Conv_type=opt.Conv_type, Dw_Index=opt.Dw_Index)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids,
                                          one_D_conv=opt.OneDConv, one_D_conv_size=opt.OneDConv_size)

        print('---------- Networks initialized -------------')

        # load networks
        pretrained_path = '' if not self.isTrain else opt.load_pretrain
        self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
        if self.isTrain:
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criteraion_mse = torch.nn.MSELoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = Bpgan_VGGLoss()

            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'MSE_Loss', 'Feature', 'D_real', 'D_fake']

            params = self.netDecoder.parameters()
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate(self, test_image, use_pool=False):
        input_concat = test_image.detach()
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def encode_input(self, label_map, real_image=None, infer=False):

        input_label = label_map.data.cuda()

        # get edges from instance map

        input_label = Variable(input_label, requires_grad=not infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding

        return input_label, real_image

    def inference(self, label):
        # Encode Inputs
        input_label, image = self.encode_input(Variable(label), infer=True)

        # Fake Generation
        input_concat = input_label
        Compressed_p = self.netE.forward(input_concat)
        fake_image = self.netDecoder.forward(Compressed_p)
        return fake_image, Compressed_p

    def forward(self, label, image, infer=False, ADMM=False):
        # Encode Inputs
        input_label, real_image = self.encode_input(label, image)

        # Fake Generation

        input_concat = input_label
        Compressed_p = self.netE.forward(input_concat)
        fake_image = self.netDecoder.forward(Compressed_p)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(fake_image)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        loss_mse = 0
        if not self.opt.no_mse_loss:
            loss_mse = self.criteraion_mse(fake_image, real_image) * self.opt.lambda_mse
        # Only return the fake_B image if necessary to save BW
        if ADMM == False:
            return [[loss_G_GAN, loss_G_GAN_Feat, loss_mse, loss_G_VGG, loss_D_real, loss_D_fake],
                    None if not infer else fake_image]
        else:
            return [[loss_G_GAN, loss_G_GAN_Feat, loss_mse, loss_G_VGG, loss_D_real, loss_D_fake],
                    None if not infer else fake_image, Compressed_p]

    def save(self, which_epoch):
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netDecoder, 'Decoder_Dis', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netE.parameters()) + list(self.netDecoder.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
