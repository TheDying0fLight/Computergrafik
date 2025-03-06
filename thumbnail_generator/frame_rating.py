import google.generativeai as genai
import numpy as np
from pyyoutube import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import models, transforms
from moviepy.video.io.VideoFileClip import VideoFileClip
from IPython.display import clear_output, display

#from .datacollection import extract_frames

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_LAUNCH_BLOCKING = 1
DTYPE = torch.float32 #if DEVICE == "cpu" else torch.float16  # CPU doesn't support float16
MD_REVISION = "2024-07-23"
genai.configure(api_key="AIzaSyAOz2kX5yf8Sd3M5JcmARXZoY2GECYpmxw")
model = genai.GenerativeModel('gemini-1.5-flash')
prompt = "How well does the image illustrate the following text on a scale from 1 (very poor) to 100 (very good)? Name only the number.\nThe text: "


class FrameRating():
    def gemini(keysentence: str, vid_path: str) -> list[list, list]:
        frames = video_to_frames(vid_path)
        frame_rating = []
        for frame in frames:
          response = model.generate_content([prompt + keysentence, Image.fromarray(np.uint8(frame)).convert('RGB')]).text
          frame_rating.append(response)

        return [frame_rating, frames]

    def moondream(keysentence: str, vid_path: str, path="vikhyatk/moondream2", ft_path = None) -> list[list, list]:
        frames = video_to_frames(vid_path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            revision=MD_REVISION,
            trust_remote_code=True,
            attn_implementation=None if DEVICE == "cuda" else None,
            torch_dtype=DTYPE,
            device_map={"": DEVICE})
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

        if ft_path:
          model.load_state_dict(torch.load(ft_path, weights_only=True))

        responses = {}
        prompt = prompt + keysentence
        frame_rating = []

        for frame in frames:
          prompt_frame = [frame, prompt]
          response = model.answer_question(*prompt_frame, tokenizer=tokenizer)
          frame_rating.append(response)

        return [frame_rating, frames]


def video_to_frames(video_path):

  frame_num = 10
  video = VideoFileClip(video_path)
  duration = video.duration
  frames = []
  step = duration/frame_num
  time = random_number = np.random.uniform(0, step) #step/frame_num
  while time < duration:
    frames.append(video.get_frame(time))
    time += step
  clear_output(wait=True)
  video.close()
  
  return frames
