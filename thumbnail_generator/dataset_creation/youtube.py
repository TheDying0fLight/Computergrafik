from pyyoutube import Api
from youtube_transcript_api import YouTubeTranscriptApi
import json
import requests
import shutil
import os
import time
from PIL import Image
from pathlib import Path
from pytube.innertube import _default_clients
from IPython.display import clear_output, display
import google.generativeai as genai
from google.api_core.exceptions import InternalServerError

import torch
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


_default_clients["ANDROID_MUSIC"] = _default_clients["WEB"]
yt_str = "https://www.youtube.com/watch?v={}"

class Youtube():
    def __init__(self, path: str = None) -> None:
        self.path = path
        self.videos = []
        if path is not None:
            try:
                with open(f"{path}.json", "r") as f: self.videos = json.load(f)
            except Exception: pass

    def to_json(self, path = None):
        if path is not None: self.path = path
        if self.path is None: raise ValueError("A path has to be defined")
        with open(f"{self.path}.json", "w") as f: json.dump(self.videos, f)

    def get_popular(self, api_key = None, amount = 5) -> list:
        added = []
        if api_key is not None: self.api = Api(api_key=api_key)
        try: self.api
        except Exception: raise ValueError("A api_key has to be given at least once")
        videos = self.api.get_videos_by_chart(chart="mostPopular", count=amount).to_dict()["items"]
        for video in videos:
            exists = False
            for v in self.videos:
                if video["id"] == v["id"]:
                    exists = True
                    break
            if not exists:
                self.videos.append(video)
                added.append(video)
        return added

    def add_transcripts(self, amount = 10, verbose = False):
        added = []
        for v in self.videos:
            if amount is not None and len(added) >= amount: break
            try: v["caption"]
            except Exception:
                try:
                    v["caption"] = YouTubeTranscriptApi.get_transcript(v["id"])
                    added.append(v)
                    try:
                        print(v["caption"], end='\r')
                    except Exception: pass
                except Exception: v["caption"] = None
        return added

    # attempt to use pytube instead of YoutubeTranscriptApi
    # def add_transcripts(self, amount = 10, verbose = False):
    #     added = []
    #     for v in self.videos:
    #         if amount <= 0: break
    #         try: v["caption"]
    #         except:
    #             vid = pytube.YouTube(yt_str.format(v["id"]))
    #             tracks = [k.__dict__["code"] for k in list(vid.captions.keys()) if "en" in k.__dict__["code"]]
    #             if len(tracks) > 0: print(vid.captions[tracks[-1]].json_captions)
    #             amount -= 1
    #     return added

    def add_thumbnails(self, amount = 10, show = False):
        thumbnails = []
        for v in self.videos:
            if amount is not None and len(thumbnails) >= amount: break
            id = v["id"]
            path = Path(f"{self.path}/{id}.webp")
            if os.path.exists(path): continue
            urls = v["snippet"]["thumbnails"]
            try: url = urls["standard"]["url"]
            except Exception:
                try: url = urls["maxres"]["url"]
                except Exception: url = urls["high"]["url"]
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                img = Image.open(path)
                thumbnails.append(img)
                if show:
                    try:
                        clear_output(wait=True)
                        display(img)
                    except Exception: pass
        return thumbnails

    def add_gemini_thumbnail_description(self, api_key, amount=15, show=False, requests_per_min=10, overwrite=False):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        interval = 60 / requests_per_min

        for v in self.videos:
            if amount is not None and amount < 0: break
            ret = self.generate_thumbnail_description(v, show, overwrite, model, tokenizer=None, keyname="gemini")
            if ret != -1: time.sleep(interval)
            if amount is not None: amount -= 1

    def add_internvl2_thumbnail_description(self, show=False, overwrite=False):
        path = "OpenGVLab/InternVL2_5-1B"
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        for v in self.videos:
            self.generate_thumbnail_description(v, show, overwrite, model, tokenizer, keyname="internvl2")

    def generate_thumbnail_description(self, video, show, overwrite, model, tokenizer, keyname):
        try: video["thumbnail_descriptions"]
        except Exception: video["thumbnail_descriptions"] = {}
        try:
            video["thumbnail_descriptions"][keyname]
            if not overwrite: return -1
        except Exception: pass
        id = video['id']
        image_path = os.path.join(self.path, f"{id}.webp")

        try:
            if keyname == "gemini":
                image = Image.open(image_path)
            if keyname == "internvl2":
                image = Image.open(image_path).convert('RGB')
                pixel_values = Youtube.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        except FileNotFoundError:
            print(f"Video {id} has no thumbnail")
            return -1

        prompt = "Generate a positive prompt for Stable Diffusion for the given thumbnail with no more than 77 tokens. The response should only include the prompt."

        if keyname == "gemini":
            for i in range(10):
                try: video["thumbnail_descriptions"][keyname] = model.generate_content([image, prompt]).text
                except InternalServerError: continue
                break
        if keyname == "internvl2":
            generation_config = dict(max_new_tokens=1024, do_sample=True, pad_token_id = tokenizer.eos_token_id)
            video["thumbnail_descriptions"][keyname] = model.chat(tokenizer, pixel_values, '<image>\n' + prompt, generation_config)
        if show:
            try:
                clear_output(wait=True)
                display(image)
                print(video["thumbnail_descriptions"][keyname])
            except Exception: pass


    ###################################
    # image preprocessing for InternVL2
    ###################################

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        MEAN, STD = Youtube.IMAGENET_MEAN, Youtube.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = Youtube.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = Youtube.build_transform(input_size=input_size)
        images = Youtube.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
