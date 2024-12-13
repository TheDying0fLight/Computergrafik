from pyyoutube import Api
from youtube_transcript_api import YouTubeTranscriptApi
import json
import requests, shutil
import os
from PIL import Image
from pathlib import Path
from pytube.innertube import _default_clients
from IPython.display import clear_output

_default_clients["ANDROID_MUSIC"] = _default_clients["WEB"]
yt_str = "https://www.youtube.com/watch?v={}"

class Youtube():
    def __init__(self, path: str = None) -> None:
        self.path = path
        self.videos = []
        if path is not None:
            try:
                with open(f"{path}.json", "r") as f: self.videos = json.load(f)
            except: pass

    def to_json(self, path = None):
        if not path is None: self.path = path
        if self.path is None: raise ValueError("A path has to be defined")
        with open(f"{self.path}.json", "w") as f: json.dump(self.videos, f)

    def get_popular(self, api_key = None, amount = 5) -> list:
        if api_key is not None: self.api = Api(api_key=api_key)
        try: self.api
        except: raise ValueError("A api_key has to be given at least once")
        videos = self.api.get_videos_by_chart(chart="mostPopular", count=amount).to_dict()["items"]
        for video in videos:
            exists = False
            for v in self.videos:
                if video["id"] == v["id"]: exists = True
            if not exists: self.videos.append(video)
        return self.videos

    def add_transcripts(self, amount = 10, verbose = False):
        added = []
        for v in self.videos:
            if amount is not None and len(added) >= amount: break
            try: v["caption"]
            except:
                try:
                    v["caption"] = YouTubeTranscriptApi.get_transcript(v["id"])
                    added.append(v)
                except Exception as e:
                    if verbose: print(e)
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
            try: url = v["snippet"]["thumbnails"]["standard"]["url"]
            except:
                try: url = v["snippet"]["thumbnails"]["standard"]["url"]
                except: url = v["snippet"]["thumbnails"]["high"]["url"]
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
                    except: pass
        return thumbnails