import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moviepy.video.io.VideoFileClip import VideoFileClip
from IPython.display import clear_output
from enum import Enum
from PIL.Image import Image
import numpy as np
import concurrent.futures
import multiprocessing
from transformers import CLIPProcessor, CLIPModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_LAUNCH_BLOCKING = 1
DTYPE = torch.float32  # if DEVICE == "cpu" else torch.float16  # CPU doesn't support float16
MD_REVISION = "2024-07-23"
genai.configure(api_key="AIzaSyAOz2kX5yf8Sd3M5JcmARXZoY2GECYpmxw")


class Prompts(Enum):
    VideoToPrompt = "Generate a stable diffusion prompt for a thumbnail for the following text and provided images.\nThe text: "
    SumText = "Generate one sentence that summarizes the content of the following text. \nThe text: "
    Rating = "How well does the image illustrate the following text on a scale from 1 (very poor) to 100 (very good)? Name only the number.\nThe text: "


class PromptGenerator():
    def gemini(transcript: str, images: list[Image] = [], prompt=Prompts.VideoToPrompt) -> dict[str, str]:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content([prompt.value + transcript, *images]).text

    def moondream(transcript: str, path="vikhyatk/moondream2", ft_path=None, prompt=Prompts.VideoToPrompt) -> dict[str, str]:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            revision=MD_REVISION,
            trust_remote_code=True,
            attn_implementation=None if DEVICE == "cuda" else None,
            torch_dtype=DTYPE,
            device_map={"": DEVICE})
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

        if ft_path: model.load_state_dict(torch.load(ft_path, weights_only=True))

        pos = [prompt.value + transcript]

        return model.answer_question(*pos, tokenizer=tokenizer)


class Describe():
    prompt = "Generate only a stable diffusion prompt for a thumbnail in a {} style of the following image."

    def gemini(self, image, style: str) -> str:
        model = genai.GenerativeModel('gemini-1.5-flash')
        description = model.generate_content(
            [self.prompt.format(style), Image.open(image)]).text
        return description

    def moondream(self, image, style: str, path="vikhyatk/moondream2", ft_path=None) -> str:
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

        prompt = self.prompt.format(style)
        prompt = [image, prompt]
        description = model.answer_question(*prompt, tokenizer=tokenizer)

        return description


class FrameRating():
    def gemini(keysentence: str, vid_path: str) -> list[list, list]:
        model = genai.GenerativeModel('gemini-1.5-flash')
        frames = video_to_frames(vid_path)
        frame_rating = []
        for frame in frames:
            response = model.generate_content([Prompts.Rating.value + keysentence,
                                              Image.fromarray(np.uint8(frame)).convert('RGB')]).text
            frame_rating.append(response)

        return [frame_rating, frames]

    def moondream(keysentence: str, vid_path: str, path="vikhyatk/moondream2", ft_path=None) -> list[list, list]:
        frames = video_to_frames(vid_path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            revision=MD_REVISION,
            trust_remote_code=True,
            attn_implementation=None if DEVICE == "cuda" else None,
            torch_dtype=DTYPE,
            device_map={"": DEVICE})
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

        if ft_path: model.load_state_dict(torch.load(ft_path, weights_only=True))

        prompt = Prompts.Rating.value + keysentence
        frame_rating = []

        for frame in frames:
            prompt_frame = [frame, prompt]
            response = model.answer_question(*prompt_frame, tokenizer=tokenizer)
            frame_rating.append(response)

        return [frame_rating, frames]

    def clip(keysentence: str, vid_path: str, frames: int) -> list[list, list]:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        frames = video_to_frames(vid_path, frames)
        text_inputs = processor(text=keysentence, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad(): text_features = model.get_text_features(**text_inputs)

        def encode_image_and_calculate_similarity(image):
            image_input = processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad(): image_features = model.get_image_features(**image_input)
            similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
            return similarity.item()

        max_threads = multiprocessing.cpu_count()
        used_threads = min(2, max_threads // 2)
        with concurrent.futures.ThreadPoolExecutor(used_threads) as executor:
            results = list(executor.map(encode_image_and_calculate_similarity, frames))
        return [results, frames]


def video_to_frames(video_path, frame_amt=10):
    video = VideoFileClip(video_path)
    duration = video.duration
    frames = []
    step = duration / frame_amt
    time = np.random.uniform(0, step)  # step/frame_num
    while time < duration:
        frames.append(video.get_frame(time))
        time += step
    clear_output(wait=True)
    video.close()

    return frames
