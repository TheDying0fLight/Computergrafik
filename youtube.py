import pytube
from pyyoutube import Api
from youtube_transcript_api import YouTubeTranscriptApi
import json

class Youtube():
    def __init__(self, path: str = None) -> None:
        self.path = path
        if path is not None:
            try:
                with open(path, "r") as f:
                    self.videos = json.load(f)
            except: pass
        else: self.videos = []

    def to_json(self, path = None):
        if path is not None: self.path = path
        if self.path is None: raise ValueError("A path has to be defined")
        with open(self.path, "w") as f:
            json.dump(self.videos, f)

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
            if amount <= 0: break
            try: v["caption"]
            except:
                try:
                    v["caption"] = YouTubeTranscriptApi.get_transcript(v["id"])
                    added.append(v)
                    amount -= 1
                except Exception as e:
                    if verbose: print(e)
        return added