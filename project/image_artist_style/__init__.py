"""Image/Video Artist Style Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import todos
from . import artist_style

import pdb


def get_model():
    """Create model."""

    model_path = "models/image_artist_style.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = artist_style.StyleModel()
    todos.model.load(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_artist_style.torch"):
        model.save("output/image_artist_style.torch")

    return model, device


def image_predict(content_files, style_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    content_filenames = todos.data.load_files(content_files)
    style_filenames = todos.data.load_files(style_files)

    # start predict
    progress_bar = tqdm(total=len(content_filenames) * len(style_filenames))
    for content_filename in content_filenames:
        content_tensor = todos.data.load_tensor(content_filename)
        B, C, H, W = content_tensor.shape

        for style_filename in style_filenames:
            progress_bar.update(1)
            style_tensor = todos.data.load_tensor(style_filename)
            predict_tensor = todos.model.two_forward(model, device, content_tensor, style_tensor)

            content_base_filename = os.path.basename(content_filename).split(".")[0]
            output_file = f"{output_dir}/{content_base_filename}_{os.path.basename(style_filename)}"

            SB, SC, SH, SW = style_tensor.shape
            if SH != H or SW != W:
                style_tensor = F.interpolate(style_tensor, size=(H, W), mode="bilinear", align_corners=False)
            todos.data.save_tensor([content_tensor, style_tensor, predict_tensor], output_file)

    todos.model.reset_device()
