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
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F

import redos
import todos
from . import artist_style

import pdb

ARTIST_STYLE_MULTI_TIMES = 8


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


def model_forward(model, device, content_tensor, style_tensor):
    return todos.model.two_forward(model, device, content_tensor, style_tensor)


def image_client(name, content_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(content_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.artist_style(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  artist_style {input_file} ...")
        try:
            content_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, content_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_artist_style", do_service, host, port)


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
        Hnew = int(ARTIST_STYLE_MULTI_TIMES * math.ceil(H / ARTIST_STYLE_MULTI_TIMES))
        Wnew = int(ARTIST_STYLE_MULTI_TIMES * math.ceil(W / ARTIST_STYLE_MULTI_TIMES))

        if Hnew != H or Wnew != W:
            content_tensor = F.interpolate(content_tensor, size=(Hnew, Wnew), mode="bilinear", align_corners=False)

        for style_filename in style_filenames:
            progress_bar.update(1)
            style_tensor = todos.data.load_tensor(style_filename)
            B, C, H, W = style_tensor.shape
            if Hnew != H or Wnew != W:
                style_tensor = F.interpolate(style_tensor, size=(Hnew, Wnew), mode="bilinear", align_corners=False)

            predict_tensor = model_forward(model, device, content_tensor, style_tensor)

            content_base_filename = os.path.basename(content_filename).split(".")[0]
            output_file = f"{output_dir}/{content_base_filename}_{os.path.basename(style_filename)}"
            todos.data.save_tensor([content_tensor, style_tensor, predict_tensor], output_file)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  artist_style {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def artist_style_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        content_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        content_tensor = content_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, content_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=artist_style_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.artist_style(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_artist_style", video_service, host, port)
