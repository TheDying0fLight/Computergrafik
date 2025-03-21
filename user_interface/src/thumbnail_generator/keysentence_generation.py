import google.generativeai as genai
import numpy as np
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
prompt = "Generate one sentence that summarizes the content of the following text. \nThe text: "


class KeySentenceGenerator():
    def gemini(transcript: str) -> str:
        responses = {}
        response = model.generate_content([prompt + transcript]).text
        return response

    def moondream(transcript: str, path="vikhyatk/moondream2", ft_path = None) -> str:
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
        prompt = prompt + transcript
        prompt = [prompt]
        response = model.answer_question(*prompt, tokenizer=tokenizer)

        return response
