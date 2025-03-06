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
prompt = "Generate one {} stable diffusion prompt for a thumbnail for the following text and provided images.\nThe text: "


class PromptGenerator():
    def gemini(transcript: str, images: list[np.ndarray] = []) -> dict[str, str]:
        responses = {}
        responses["positive"] = model.generate_content([prompt.format("positive") + transcript, *images]).text
        responses["negative"] = model.generate_content([prompt.format("negative") + transcript, *images]).text
        return responses

    def moondream(transcript: str, vid_path: str, path="vikhyatk/moondream2", ft_path = None) -> dict[str, str]:
        vid_emb = video_to_emb(vid_path)
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
        prompt_positive = prompt.format("positive") + transcript
        prompt_negative = prompt.format("negative") + transcript

        # responses["positive"] = model.answer_question(model.encode_image(Image.open(image)), prompt_positive,
        #                                              tokenizer=tokenizer)
        # responses["negative"] = model.answer_question(model.encode_image(Image.open(image)), prompt_negative,
        #                                              tokenizer=tokenizer)
        pos = [vid_emb, prompt_positive]
        neg = [vid_emb, prompt_negative]
        responses["positive"] = model.answer_question(*pos, tokenizer=tokenizer)
        responses["negative"] = model.answer_question(*neg, tokenizer=tokenizer)

        return responses


def video_to_emb(video_path):

      # Video in frames
    video = VideoFileClip(video_path)
    duration = video.duration
    frames = []
    step = duration/2
    time = step/2
    while time < duration:
      frames.append(video.get_frame(time))
      time += step
    clear_output(wait=True)
    video.close()
    
      # Frames in embeddings
    embeddings = []
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((2048, 2048)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = AutoModelForCausalLM.from_pretrained(
      "vikhyatk/moondream2",
      revision="2025-01-09",
      trust_remote_code=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    for frame in frames:
        #print("frame", frame.shape)
        # Fix input tensor shape
        input_tensor = preprocess(frame)
        #print("input tensor", input_tensor.shape)
        with torch.no_grad():
            embedding = model(input_tensor)
            #print("mode inp tens", model(input_tensor).shape)
            #print("embedding", embedding.shape)
            embeddings.append(embedding)

    # Mean of embeddings
    #embeddings_mean = torch.mean(torch.stack(embeddings), dim=0)
    #print("embedding mean", embeddings_mean.shape)

    # Mean of embeddings
    #if combine_type == 'mul':
    #embedding_frames = 1
    #for emb in embeddings:
    #  embedding_frames = embedding_frames * emb
    #if combine_type == 'add':
    embedding_frames = 0
    for emb in embeddings:
      embedding_frames = embedding_frames + emb
    #if combine_type == 'mean':
    #embedding_frames = torch.mean(torch.stack(embeddings), dim=0)

    return embedding_frames

    #return embeddings_mean
