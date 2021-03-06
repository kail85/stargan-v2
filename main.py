"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch
from torch.utils import data

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver

from PIL import Image
import numpy as np
import cv2

'''
data
    |-[exp_name]
        |-train
            |-class01
                | images....png
            |-class02
                | images....png
            |-...
        |-val
            |-class01
                | images....png
            |-class02
                | images....png
            |-...

expr
    |-checkpoints
        |-[exp_name]
            | model....ckpt
            |-samples
                | debug images
                | log.txt
'''

#%%
mode = 'test' # 'train' or 'test'
exp_name = 'ccvdg' # which checkpoint to load

def set_train(args):
    args.mode = 'train'

    args.total_iters = 3000*4*20 # num of images to process in total
    args.batch_size = 4  # 2 for M60
    args.num_workers = 0 # must set to 0 maybe windows issue

    args.w_hpf = 0 # alignment

    args.lambda_reg = 1
    args.lambda_sty = 1
    args.lambda_ds = 2
    args.lambda_cyc = 1

    args.num_domains = 4
    args.img_size = 256 # we don't want to resize in the fly
    args.train_img_dir = os.path.join('data', exp_name, 'train')
    args.val_img_dir = os.path.join('data', exp_name, 'val') # note all images are evaluated

    exp_ind = 0
    args.checkpoint_dir = os.path.join('expr', 'checkpoints', exp_name + f'_%02d'%exp_ind)
    if os.path.exists(args.checkpoint_dir):
        args.checkpoint_dir = os.path.join('expr', 'checkpoints', exp_name + f'_%02d'%(exp_ind+1))

    args.sample_dir = os.path.join(args.checkpoint_dir, 'samples')

    # print/save log every n images being processed
    args.print_every = 1000
    args.sample_every = 5000
    args.save_every   = 5000
    args.eval_every = np.iinfo(np.int).max # never evaluate

    return args


def set_inference(args):
    args.mode = 'sample'

    args.num_domains = 4
    args.w_hpf = 0

    # args.batch_size = 2  # 2 for M60
    args.num_workers = 0 # must set to 0 maybe windows issue

    args.make_video = True

    args.resume_iter = 40000  # to determine ckpt file to load
    args.checkpoint_dir = os.path.join('expr', 'checkpoints', exp_name + '_00')

    args.src_dir = os.path.join('data', exp_name, 'test', 'test2')
    args.ref_dir = os.path.join('data', exp_name, 'test', 'test2')
    args.result_dir = os.path.join('expr', 'checkpoints', exp_name + '_00', 'result')

    return args


#%%
def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def find_img_datatype(image_root_path):
    domain_names = os.listdir(image_root_path)
    image_path = os.path.join(image_root_path, domain_names[0])

    image = Image.open(os.path.join(image_path, os.listdir(image_path)[0]))
    if image.mode is 'L':
        img_datatype = cv2.CV_8UC1
    elif image.mode is 'I':
        img_datatype = cv2.CV_16UC1
    elif image.mode is 'RGB':
        img_datatype = cv2.CV_8UC3
    else:
        raise ValueError('Unknown image data type.')

    return img_datatype


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.mode == "train":
        args.img_datatype = find_img_datatype(args.train_img_dir)
    elif args.mode == "sample":
        args.img_datatype = find_img_datatype(args.src_dir)

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_type = args.img_datatype,
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_type = args.img_datatype,
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_type = args.img_datatype,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_type = args.img_datatype,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_type = args.img_datatype,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=False, # was True
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data\\celeba_hq\\train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data\\celeba_hq\\val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr\\samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr\\checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr\\eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr\\results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets\\representative\\celeba_hq\\src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets\\representative\\celeba_hq\\ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets\\representative\\custom\\female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets\\representative\\celeba_hq\\src\\female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr\\checkpoints\\wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr\\checkpoints\\celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)

    args = parser.parse_args()

    if mode == 'train':
        args = set_train(args)
    else:
        args = set_inference(args)

    main(args)
