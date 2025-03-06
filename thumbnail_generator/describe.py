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
prompt = "Generate a cartoonish description of the following image."


class Describe():
  def gemini(image) -> str:
    description = model.generate_content([prompt, Image.fromarray(np.uint8(image)).convert('RGB')]).text
    return description