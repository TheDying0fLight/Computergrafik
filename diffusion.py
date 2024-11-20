import torch
from diffusers import AutoPipelineForText2Image

class diffuser():
    def __init__(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, steps = 25, guidance = 7):
        return self.pipe(prompt,
                         num_inference_steps = steps,
                         guidance_scale = guidance,
                        ).images