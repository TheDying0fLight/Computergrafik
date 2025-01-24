import google.generativeai as genai
import numpy as np

genai.configure(api_key="AIzaSyAOz2kX5yf8Sd3M5JcmARXZoY2GECYpmxw")
model = genai.GenerativeModel('gemini-1.5-flash')
prompt = "Generate one {} stable diffusion prompt for a thumbnail for the following text and provided images.\nThe text: "

class PromptGenerator():
    def gemini(transcript: str, images: list[np.ndarray] = []) -> dict[str,str]:
        responses = {}
        responses["positive"] = model.generate_content([prompt.format("positive") + transcript, *images]).text
        responses["negative"] = model.generate_content([prompt.format("negative") + transcript, *images]).text
        return responses