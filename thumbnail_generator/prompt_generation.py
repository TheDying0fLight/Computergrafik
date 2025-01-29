import google.generativeai as genai
import numpy as np
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import models, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_LAUNCH_BLOCKING = 1
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16  # CPU doesn't support float16
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

    def moondream(transcript: str, vid_path: str, path="vikhyatk/moondream2") -> dict[str, str]:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            revision=MD_REVISION,
            trust_remote_code=True,
            attn_implementation=None if DEVICE == "cuda" else None,
            torch_dtype=DTYPE,
            device_map={"": DEVICE})
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

        responses = {}
        prompt_positive = prompt.format("positive") + transcript
        prompt_negative = prompt.format("negative") + transcript

        # responses["positive"] = model.answer_question(model.encode_image(Image.open(image)), prompt_positive,
        #                                              tokenizer=tokenizer)
        # responses["negative"] = model.answer_question(model.encode_image(Image.open(image)), prompt_negative,
        #                                              tokenizer=tokenizer)
        vid_emb = video_to_emb(vid_path)
        pos = [vid_emb, prompt_positive]
        neg = [vid_emb, prompt_negative]
        responses["positive"] = model.answer_question(*pos, tokenizer=tokenizer)
        responses["negative"] = model.answer_question(*neg, tokenizer=tokenizer)

        return responses


def video_to_emb(video_path):

    # Video in frames
    video = VideoFileClip(video_path)
    duration = video.duration
    frame_times = sorted(random.sample(range(int(duration * 1000)), 10))  # timesteps in ms
    frames = [video.get_frame(t / 1000.0) for t in frame_times]  # frame of these 10 random timesteps
    video.close()

    # Frames in embeddings
    embeddings = []
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet50(weights="IMAGENET1K_V1")
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    for frame in frames:
        input_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            embedding = model(input_tensor).squeeze()
            embeddings.append(embedding)

    # Mean of embeddings
    embeddings_mean = torch.mean(embeddings, dim=0)
    return embeddings_mean
