from .gemini import example_transcript

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")


path = "OpenGVLab/InternVideo2-Chat-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).cuda()
tokenizer =  AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


def get_prompts(transcript: str = None) -> dict[str,str]:
    if transcript is None: transcript = example_transcript
    
    responses = {}

    video_path = "videos/a01c.mp4" # TODO insert correct video (path)
    # sample uniformly 8 frames from the video
    video_tensor = load_video(video_path, num_segments=8, return_msg=False)
    video_tensor = video_tensor.to(model.device)
    
    generation_config = dict(do_sample=True, pad_token_id = tokenizer.eos_token_id)

    chat_history= []

    prompt_positive = "Generate a positive prompt for stable diffusion of the following text with no more than 77 tokens.\nThe text: " + transcript
    # prompt_negative = "Generate a negative prompt for stable diffusion of the following text with no more than 77 tokens.\nThe text: " + transcript
    
    responses["positive"] = model.chat(tokenizer, '', prompt_positive, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config=generation_config)[0]
    # responses["negative"] = model.chat(tokenizer, '', prompt_negative, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True, generation_config=generation_config)[0]
    
    return responses



#####################################
# video preprocessing for InternVideo
#####################################

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames