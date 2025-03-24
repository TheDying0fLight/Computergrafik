import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moviepy.video.io.VideoFileClip import VideoFileClip
from IPython.display import clear_output
from enum import Enum
import PIL.Image
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
    def gemini(transcript: str, images: list[PIL.Image.Image] = [], prompt=Prompts.VideoToPrompt) -> dict[str, str]:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content([prompt.value + transcript, *images]).text

    def moondream(transcript: str, path="vikhyatk/moondream2", ft_path=None, prompt=Prompts.VideoToPrompt) -> dict[str, str]:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": DEVICE}
        )

        if ft_path: model.load_state_dict(torch.load(ft_path, weights_only=True))

        pos = [prompt.value + transcript]

        return model.answer_question(*pos)


class Describe():
    def gemini(self, img: PIL.Image.Image, style: str) -> str:
        prompt = "Generate a stable diffusion prompt for a thumbnail in a {} style of the following image. The answer should only contain the prompt"
        model = genai.GenerativeModel('gemini-1.5-flash')
        description = model.generate_content(
            [prompt.format(style), img]).text
        return description

    def moondream(self, img: PIL.Image.Image, style: str, path="vikhyatk/moondream2", ft_path=None) -> str:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": DEVICE}
        )
        if ft_path: model.load_state_dict(torch.load(ft_path, weights_only=True))

        encoded_image = model.encode_image(img)
        description = model.caption(encoded_image)["caption"]
        description = " " + style + ", " + description

        return description


class FrameRating():
    def gemini(keysentence: str, vid_path: str, frames: int) -> list[list[tuple[float, PIL.Image.Image]]]:
        model = genai.GenerativeModel('gemini-1.5-flash')
        frames = video_to_frames(vid_path, frames)
        groups = []
        for frame in frames:
            response = model.generate_content([Prompts.Rating.value + keysentence,
                                              PIL.Image.fromarray(np.uint8(frame)).convert('RGB')]).text
            try: rating = float(response)
            except ValueError: rating = 0.0
            groups.append([(rating, frame)])
        return sorted(groups, key=lambda group: group[0][0], reverse=True)

    def moondream(keysentence: str, vid_path: str, frames: int, path="vikhyatk/moondream2", ft_path=None
                  ) -> list[list[tuple[float, PIL.Image.Image]]]:
        frames = video_to_frames(vid_path, frames)
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": DEVICE}
        )
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
        if ft_path:
            model.load_state_dict(torch.load(ft_path, weights_only=True))

        prompt = Prompts.Rating.value + keysentence
        groups = []
        for frame in frames:
            image = PIL.Image.fromarray(np.uint8(frame)).convert('RGB')
            response = model.answer_question(frame, prompt, tokenizer=tokenizer)
            try: rating = float(response)
            except ValueError: rating = 0.0
            groups.append([(rating, image)])
        return groups

    def clip(keysentence: str, vid_path: str, frames: int) -> list[list[tuple[float, PIL.Image.Image]]]:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        frames_list = video_to_frames(vid_path, frames)

        text_inputs = processor(text=keysentence, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)

        def encode_image(frame):
            image = PIL.Image.fromarray(np.uint8(frame)).convert('RGB')
            image_input = processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                image_features = model.get_image_features(**image_input)
            image_features = image_features.squeeze(0)
            text_sim = torch.nn.functional.cosine_similarity(text_features, image_features.unsqueeze(0)).item()
            return image_features, text_sim, image

        max_threads = multiprocessing.cpu_count()
        used_threads = min(2, max_threads // 2)
        with concurrent.futures.ThreadPoolExecutor(used_threads) as executor:
            results = list(executor.map(encode_image, frames_list))

        image_features_list, text_sims, images_list = zip(*results)
        features_tensor = torch.stack(image_features_list)
        norm_features = features_tensor / features_tensor.norm(dim=1, keepdim=True)
        similarity_matrix = (norm_features @ norm_features.T).cpu().tolist()

        IMAGE_GROUP_THRESHOLD = 0.9
        N = len(images_list)
        visited = [False] * N
        groups_indices = []

        for i in range(N):
            if not visited[i]:
                component = []
                queue = [i]
                while queue:
                    curr = queue.pop(0)
                    if visited[curr]:
                        continue
                    visited[curr] = True
                    component.append(curr)
                    for j in range(N):
                        if not visited[j] and similarity_matrix[curr][j] >= IMAGE_GROUP_THRESHOLD:
                            queue.append(j)
                groups_indices.append(component)

        grouped_results = []
        for comp in groups_indices:
            group_items = sorted([(text_sims[idx], images_list[idx]) for idx in comp],
                                 key=lambda x: x[0], reverse=True)
            grouped_results.append(group_items)

        return sorted(grouped_results, key=lambda group: group[0][0], reverse=True)


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
