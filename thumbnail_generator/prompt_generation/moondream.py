from .gemini import example_transcript

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image


path = "vikhyatk/moondream2"
DEVICE = "cuda"
CUDA_LAUNCH_BLOCKING=1
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-07-23"

model = AutoModelForCausalLM.from_pretrained(
    path,
    revision=MD_REVISION,
    trust_remote_code=True,
    attn_implementation=None if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    device_map={"": DEVICE})
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

def get_prompts(transcript: str = None) -> dict[str,str]:
    if transcript is None: transcript = example_transcript
    
    responses = {}

    image = None # TODO insert relevant image(s) from video | Image.open(image_path).convert('RGB')
    
    prompt_positive = "Generate a positive prompt for stable diffusion of the following text with no more than 77 tokens.\nThe text: " + transcript
    prompt_negative = "Generate a negative prompt for stable diffusion of the following text with no more than 77 tokens.\nThe text: " + transcript
    
    responses["positive"] = model.answer_question(model.encode_image(Image.open(image)), prompt_positive, tokenizer=tokenizer)
    responses["negative"] = model.answer_question(model.encode_image(Image.open(image)), prompt_negative, tokenizer=tokenizer)
    
    return responses