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
                        clear_output(wait=True)
                        print(v["caption"])
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
        prompt = "Generate a positive prompt for Stable Diffusion for the given thumbnail with no more than 77 tokens. The response should only include the prompt"
        for v in self.videos:
            if amount is not None and amount < 0: break
            ret = self.generate_thumbnail_description(v, show, overwrite, model, keyname="gemini", prompt=prompt)
            if ret != -1: time.sleep(interval)
            if amount is not None: amount -= 1

    def generate_thumbnail_description(self, video, show, overwrite, model, keyname, prompt):
        try: video["thumbnail_descriptions"]
        except Exception: video["thumbnail_descriptions"] = {}
        try:
            video["thumbnail_descriptions"][keyname]
            if not overwrite: return -1
        except Exception: pass
        id = video['id']
        image_path = os.path.join(self.path, f"{id}.webp")
        try: image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Video {id} has no thumbnail")
            return -1

        for i in range(10):
            try: video["thumbnail_descriptions"][keyname] = model.generate_content([image, prompt]).text
            except InternalServerError: continue
            break
        if show:
            try:
                clear_output(wait=True)
                display(image)
                print(id)
            except Exception: pass
