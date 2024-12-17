# from transformers import AutoModelForCausalLM, AutoTokenizer
from .gemini import example_transcript

# code to download and save the model from hugging face
# Thanks to Gustavosta: https://huggingface.co/Gustavosta
# tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
# model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16)

def get_prompts(transcript: str = None) -> dict[str,str]:
    if transcript is None: transcript = example_transcript
    
    responses = {}

    prompt_positive = "Generate a positive prompt for stable diffusion of the following text with no more than 77 tokens.\nThe text: " + transcript
    inputs_positive = tokenizer(prompt_positive, return_tensors="pt").to(model.device)

    tokens_positive = model.generate(
        **inputs_positive,
        max_new_tokens=77,
        temperature=0.5,
        top_p=0.95,
        do_sample=True)
    responses["positive"] = tokenizer.decode(tokens_positive[0], skip_special_tokens=True)

    prompt_negative = "Generate a negative prompt for stable diffusion of the following text with no more than 77 tokens.\nThe text: " + transcript
    inputs_negative = tokenizer(prompt_negative, return_tensors="pt").to(model.device)

    tokens_negative = model.generate(
        **inputs_negative,
        max_new_tokens=77,
        temperature=0.5,
        top_p=0.95,
        do_sample=True)
    responses["negative"] = tokenizer.decode(tokens_negative[0], skip_special_tokens=True)
    
    return responses
