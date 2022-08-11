import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='timit_8k_mel', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='Bpgan_GAN', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='cus', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--image_bit_num',type=int,default=16)

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=90, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=64, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./dataset/timit_8k_mel/')
        self.parser.add_argument('--resize_or_crop', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_false', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--max_ngf',type=int,default=256)
        self.parser.add_argument('--n_downsample_global', type=int, default=3, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=3, help='number of residual blocks in the generator network')

        self.parser.add_argument('--C_channel', type=int, default=4,help='number of compressed channels')
        self.parser.add_argument('--Color_Input',type =str, default="gray",help="the color channel of the input, gray and RGB")
        self.parser.add_argument('--Color_Output',type=str,default="gray",help="the color channel of the output")
        self.parser.add_argument("--OneDConv",type=bool,default=False,help="if apply 1D conv on the first layer, just for spectrum")
        self.parser.add_argument("--OneDConv_size",type=int,default=63)
        self.parser.add_argument('--n_cluster',type=int,default=16,help='cluster number of quantization')
        self.parser.add_argument('--quantize_type',type=str,default='scalar',help='scalar or vector')
        self.parser.add_argument('--sampling_ratio',type=int,default=8000,help='sampling ratio')
        self.parser.add_argument('--n_fft', type=int, default=256, help='fft size')
        self.parser.add_argument('--n_mels', type=int, default=64, help='mel size')
        self.parser.add_argument('--Conv_type', type=str, default="C", help="C for conventional conv, E for efficient Conv")
        self.parser.add_argument('--Dw_Index', type=str, default="0,1,2", help="index for layer to do DW deconvolution")
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        if self.opt.Dw_Index is None:
            Dw_Index = None
        else:
            Dw_Index = self.opt.Dw_Index.split(',')
        self.opt.gpu_ids = []
        self.opt.Dw_Index = None
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if Dw_Index is not None:
            self.opt.Dw_Index = []
            for index in Dw_Index:
                self.opt.Dw_Index.append(int(index))
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
