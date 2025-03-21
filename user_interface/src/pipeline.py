from youtube_transcript_api import YouTubeTranscriptApi
from thumbnail_generator import video_id, PromptGenerator, Prompts, Describe, FrameRating

from yt_dlp import YoutubeDL
import os
from thumbnail_generator import Diffuser

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt_tab')


class Pipeline():

    def preprocess_url(video_url):
        return video_id(video_url)

    def generate_key_sentence(video_url, LLM_type, frame_amt=20, combine_emb="mul"):

        id = Pipeline.preprocess_url(video_url)

        print("transcript download and preprocessing...")
        stop_words = set(stopwords.words('english'))
        words = set(nltk.corpus.words.words())

        transcript_list = YouTubeTranscriptApi.get_transcript(id)
        transcript = ' '.join([section['text'] for section in transcript_list])

        transcript = word_tokenize(transcript)
        transcript = [w.lower() for w in transcript if w not in stop_words and w in words]
        transcript = ' '.join(transcript)

        print("model generating key sentence from transcript...")
        sentence = ""
        try:
            if LLM_type == 'Moondream':
                sentence = PromptGenerator.moondream(transcript, prompt=Prompts.SumText)
            elif LLM_type == 'Finetuned Moondream':
                sentence = PromptGenerator.moondream(transcript,
                                                     ft_path="/content/drive/MyDrive/moondream_ft_moon_mean_eps10_bs8_1frame",
                                                     prompt=Prompts.SumText)
            elif LLM_type == 'Gemini':
                sentence = PromptGenerator.gemini(transcript, prompt=Prompts.SumText)
            else:
                sentence = 'wrong LLM_type'
        except Exception as e:
            print(e)

        return sentence

    def rate_frames(key_sentence, video_url, LLM_type, frame_amt, ext="webm", res=480):

        print(f"downloading the video and extracting {frame_amt} frames...")

        id = Pipeline.preprocess_url(video_url)

        opts = {
            'format': f'bestvideo[ext={ext}][height={res}]',
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'paths': {'home': ''},
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': True
        }

        frame_rating = [0]
        frames = []

        yt_str = "https://www.youtube.com/watch?v="
        with YoutubeDL(opts) as ydl: ydl.download(yt_str + id)

        print("model rating frames...")

        if LLM_type == 'CLIP':
            [frame_rating, frames] = FrameRating.clip(key_sentence, f"{id}.{ext}", frame_amt)
            print("FRAME RATING:", list(sorted(frame_rating)))

        # while max(frame_rating) < 80:
        try:
            if LLM_type == 'Moondream':
                [frame_rating, frames] = FrameRating.moondream(key_sentence, f"{id}.{ext}", frame_amt)
            elif LLM_type == 'Finetuned Moondream':
                [frame_rating, frames] = FrameRating.moondream(key_sentence, f"{id}.{ext}", frame_amt,
                                                               ft_path="/content/drive/MyDrive/moondream_ft_moon_mean_eps10_bs8_1frame")
            elif LLM_type == 'Gemini':
                [frame_rating, frames] = FrameRating.gemini(key_sentence, f"{id}.{ext}", frame_amt)
            else:
                print('wrong LLM_type')
        except Exception as e:
            print(e)

        frame_rating = [int(x.strip()) for x in frame_rating if isinstance(x, str) and x.strip().isdigit()]
        # print("FRAME RATING IN PIPELINE.py:", frame_rating)

        os.remove(f"{id}.{ext}")
        return frame_rating, frames

    def get_best_frame(frame_rating, frames):
        zipped = list(zip(frame_rating, frames))
        s = list(sorted(zipped, key=lambda tup: tup[0], reverse=True))
        best_frame = s[0][1]
        return best_frame

    def describe_best_frame(best_frame, style, LLM_type):
        print("model describing the frame...")
        try:
            if LLM_type == 'Moondream':
                describe_best_image = Describe().moondream(best_frame, style)
            elif LLM_type == 'Finetuned Moondream':
                describe_best_image = Describe().moondream(best_frame,
                                                           style,
                                                           ft_path="/content/drive/MyDrive/moondream_ft_moon_mean_eps10_bs8_1frame")
            elif LLM_type == 'Gemini':
                describe_best_image = Describe().gemini(best_frame, style)
            else:
                raise Exception('wrong LLM_type')
        except Exception as e:
            print(e)

        return describe_best_image

    def generate_thumbnail(prompt):
        print("Stable Diffusion generating the cartoonish thumbnail...")
        gen_res = (1344, 768)
        diff = Diffuser()

        # option to use trained lora
        # lora = "sdxl/1344x768-200-1600-500-cats-no"
        # diff.pipe.load_lora_weights(f"loras/{lora}.safetensors")

        images = diff.generate(prompt, batch_size=1, width=gen_res[0], height=gen_res[1], seed=42)
        return images[0]
