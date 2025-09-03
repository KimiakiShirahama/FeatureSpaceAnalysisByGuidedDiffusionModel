import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.optim import SGD, Adam, AdamW
import PIL
from torch.utils import data
from pathlib import Path
from PIL import Image
import random
from helper import OptimizerDetails
import clip
import os
import inspect
import torchvision.transforms.functional as TF

import pdb

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    print(config.model)
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img

def cycle(dl):
    while True:
        for data in dl:
            yield data

import errno
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


class Clip(nn.Module):
    def __init__(self, log_filename=None):
        super(Clip, self).__init__()

        clip_model, clip_preprocess = clip.load("RN50")
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        self.model = clip_model
        self.preprocess = clip_preprocess
        print(self.preprocess)
        self.trans = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.log_filename = log_filename

    def forward(self, x, y, log_info=None):
        x = (x + 1) * 0.5

        # Virtual save image to precisely simulated that x (image) is loaded and encoded into a Clip's embedding
        x = x.mul(255)
        x = x + x.round().detach() - x.detach()  # Straight through estimator to backpropagte across round function 

        # Processes needed to complete the preprocessing of CLIP
        x = TF.resize(x, (224, 224), interpolation=TF.InterpolationMode.BICUBIC)
        x = x + x.round().detach() - x.detach()  # Straight through estimator to backpropagte across round function 
        x = x.div(255)  # div is only needed to complete ToTensor
        x = self.trans(x)

        # Cast xfeats and yfeats to float32 to improve the computational precision
        xfeats = self.model.encode_image(x).to(torch.float32)
        yfeats = y.to(torch.float32)

        # Batch computation of the squared Euclidian distances for rows in xfeats and those in yfeats
        # (the resulting matrix is (# of xfeat's rows x # of yfeat's row)), although it is not needed
        x_sq = torch.sum(xfeats**2, dim=1, keepdim=True)
        y_sq = torch.sum(yfeats**2, dim=1, keepdim=True).T
        xy = torch.mm(xfeats, yfeats.T)
        xy_dists = x_sq + y_sq - 2.0 * xy

        if self.log_filename != None and log_info != None:
            # FYI: Check the cosine similarity between xfeats and yfeats
            # (Normalisation is performed according to CLIP's forward in src/clip/clip/model.py)
            xfeats2 = xfeats / xfeats.norm(dim=1, keepdim=True)
            yfeats2 = yfeats / yfeats.norm(dim=1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            xy_cossim = logit_scale * xfeats2 @ yfeats2.t()
            with open(self.log_filename, mode='a') as f:
                if "bk_id" in log_info:  # Backward guidance where rec_iter_id is assumed to be specified
                    print(f'{log_info["t_id"]}th timestep, {log_info["rec_id"]}th self-recurrence, {log_info["bk_id"]}th backward iteration', end="", file=f)
                elif log_info["comment"] == "best image":   # Image evaluation before and after forward guidance
                    print(f'{log_info["t_id"]}th timestep, {log_info["rec_id"]}th self-recurrence', end="", file=f)
                elif log_info["comment"] == "best z_prev":  # Image evaluation for z_prev selection
                    print(f'{log_info["t_id"]}th timestep, {log_info["rec_id"]}th self-recurrence (z_prev)', end="", file=f)
                else:  # For the finally created image
                    print(f'{log_info["t_id"]}th timestep (FINAL)', end="", file=f)
                print(f" -> xy_dists:{xy_dists.item()}, xy_cossim:{xy_cossim.item()}", file=f)

        return xy_dists

    def get_image_embedding(self, img):
        return self.model.encode_image(img)

    def get_text_embedding(self, text):
        return self.model.encode_text(text)


def get_optimation_details(args):

    l_func = Clip(args.optim_log_filename)
    l_func.eval()
    for param in l_func.parameters():
        param.requires_grad = False
    l_func = torch.nn.DataParallel(l_func).cuda()

    operation = OptimizerDetails()

    operation.image_H = args.H
    operation.image_W = args.W
    
    operation.num_steps = args.optim_num_steps
    operation.early_emp_end = args.optim_early_emp_end
    operation.select_z_prev = args.optim_select_prev
    operation.operation_func = None
    operation.other_guidance_func = None

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l_func
    operation.other_criterion = None

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff
    operation.tv_loss = args.optim_tv_loss

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.original_guidance = args.optim_original_conditioning
    operation.mask_type = args.optim_mask_type

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.do_guidance_3_norm = args.optim_do_forward_guidance_norm
    operation.grad_clip_threshold = args.optim_grad_clip_threshold

    operation.avg_sim_threshold = args.optim_avg_sim_to_start_backward
    operation.sim_history_size = args.optim_sim_queue_size

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 10  # 500
    operation.folder = args.optim_folder

    return operation, l_func

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./models/ldm/stable-diffusion/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="epochs",
    )
    parser.add_argument(
        "--save_image_folder",
        type=str,
        default='./sanity_check/',
        help="folder to save",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="data/segmentation_data/Walker/og_img_0.png",
        help="png image filename or text prompt"
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument('--optim_do_forward_guidance_norm', action='store_true', default=False)
    parser.add_argument(
        "--optim_grad_clip_threshold",
        type=float,
        default=-1.0,
        help="Gradient clipping is done if this threshold is positive"
    )
    parser.add_argument(
        "--optim_avg_sim_to_start_backward",
        type=float,
        default=0.1,
        help="Average similarity to start backward guidance"
    )
    parser.add_argument(
        "--optim_sim_queue_size", type=int, default=80,
        help="Size of a queue of similarities to compute their average"
    )
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_aug', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./results/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--optim_early_emp_end", default=-1, type=int)
    parser.add_argument("--optim_select_prev", action='store_true', default=False)
    parser.add_argument("--optim_mask_fraction", default=0.5, type=float)
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batches", default=0, type=int)

    opt = parser.parse_args()
    # seed_everything(opt.seed)
    results_folder = opt.optim_folder
    create_folder(results_folder)
    # Add the path of a log file
    opt.optim_log_filename = os.path.join(opt.optim_folder, "log.txt")

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # model.requires_grad_(False)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()
    # model.requires_grad_(False)
    operation, l_func = get_optimation_details(opt)

    # Get the goal embedding
    if opt.prompt.endswith(".png") or opt.prompt.endswith(".jpg"):
        img_filename = opt.prompt
        print(f"Goal feature is defined by the image, {img_filename}")
        img = l_func.module.preprocess(Image.open(img_filename)).unsqueeze(0).cuda()
        goal_emb = l_func.module.get_image_embedding(img)
    else:
        text_desc = opt.prompt
        print(f"Goal feature is defined by the text, {text_desc}")
        text = clip.tokenize(text_desc).cuda()
        goal_emb = l_func.module.get_text_embedding(text)
    print(f"Goal embedding: {goal_emb}")

    # A fixed embedding (condition) to make the image generation uncoditional 
    cond = model.module.get_learned_conditioning([""])

    # Generate an image whose embedding is very close to the goal embeddig
    start_time = time.perf_counter()
    output = model.module.operation_diffusion(og_img=None, operated_image=goal_emb, cond=cond, operation=operation)
    end_time = time.perf_counter()
    with open(opt.optim_log_filename, mode='a') as f:
        print(f"Elapsed time = {end_time - start_time} sec", file=f)
        print(f"Run options: {opt}", file=f)
    return_cv2(output[0], f'{results_folder}/out_img.png')


if __name__ == "__main__":
    main()
