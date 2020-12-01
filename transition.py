#%% image transition demo
from os.path import join as ospj
import torch
from torch.backends import cudnn
import cv2
import numpy as np
from munch import Munch
from matplotlib import pyplot as plt

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.utils import save_image, get_alphas

cudnn.benchmark = True
torch.manual_seed(777)


#%%
args = Munch(
    img_size=256,
    img_datatype=cv2.CV_16UC1,
    style_dim=64,
    w_hpf=0,
    latent_dim=16,
    num_domains=4,
    resume_iter=40000,
    checkpoint_dir = ospj('expr', 'checkpoints', 'ccvdg' + '_00')
)


#%%
def image_to_tensor(image_file):
    image = cv2.imread(image_file, -1)    
    # normalise
    image = image / 65535
    image = (image - 0.5) / 0.5
    # to tensor
    tensorImg = torch.zeros([1, 1, image.shape[0], image.shape[1]], dtype = torch.float32)    
    tensorImg[0, 0, :, :] = torch.from_numpy(image)
    return tensorImg


image_content = image_to_tensor(ospj('data', 'ccvdg', 'test', 'test_transition', 'cl0073_HS_D_rc_20080321.png'))
image_style   = image_to_tensor(ospj('data', 'ccvdg', 'test', 'test_transition', 'cl0067_HS_D_rc_20070624.png'))
content_label   = torch.tensor([0])
style_label     = torch.tensor([3])

image_content = image_content.cuda()
image_style   = image_style.cuda()
content_label = content_label.cuda()
style_label   = style_label.cuda()


#%%
def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def tensor_to_image(image_tensor):
    image_tensor = denormalize(image_tensor)

    image = image_tensor.mul(65535).add_(0.5).clamp_(0, 65535).to('cpu').numpy()
    image = image[0, 0, :].astype(np.uint16)

    return image


def load_model(args):
    _, nets_ema = build_model(args)

    ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{0:06d}_nets_ema.ckpt'), **nets_ema)] # compatible with Windows
    for ckptio in ckptios:
        ckptio.load(args.resume_iter)

    return nets_ema


@torch.no_grad()
def inference(net, image_content, image_style, style_label):
    s_ref = net.style_encoder(image_style, style_label)
    x_fake = net.generator(image_content, s_ref, masks=None)

    return x_fake


net = load_model(args)


#%%
image_tensor_fake = inference(net, image_content, image_style, style_label)

image_fake = tensor_to_image(image_tensor_fake)

plt.imshow(image_fake, cmap='gray')
plt.show()


#%% interplation
@torch.no_grad()
def interplation(net, image_content, image_style, content_label, style_label):
    s_con = net.style_encoder(image_content, content_label)
    s_ref = net.style_encoder(image_style, style_label)

    alphas = torch.linspace(0, 1, 10).cuda()
    # alphas = torch.FloatTensor(get_alphas(start=-1, end=1, step=0.5, len_tail=5)).cuda()

    x_fakes = []
    for alpha in alphas:
        s_inter_ref = torch.lerp(s_con, s_ref, alpha)
        x_fake = net.generator(image_content, s_inter_ref, masks=None)
        x_fakes.append(x_fake)
    
    x_fakes = torch.cat(x_fakes, dim=0)

    return x_fakes


image_inters = interplation(net, image_content, image_style, content_label, style_label)
save_image(image_inters, image_inters.shape[0], ospj('expr', 'checkpoints', 'ccvdg_00', 'result', 'vdg0to3.jpg'))
