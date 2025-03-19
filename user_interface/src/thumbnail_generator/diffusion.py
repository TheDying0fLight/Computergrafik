from diffusers import StableDiffusion3Pipeline, AutoencoderTiny, SanaPipeline, DiffusionPipeline
from PIL import Image
from pathlib import Path
import numpy as np
import os
import pandas
import torch
from IPython.display import clear_output, display
from compel import Compel, ReturnedEmbeddingsType

class Diffuser():
    def __init__(self):
        self.set_model("stabilityai/stable-diffusion-xl-base-1.0")

    def generate(self,
                 prompt,
                 negative_prompt = "",
                 steps = 25,
                 guidance = 7,
                 batch_size = 1,
                 height = None,
                 width = None,
                 seed = None):
        generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)
        isSana = issubclass(type(self.pipe), SanaPipeline)
        compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                        text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        prompt_embeds, pooled = compel([prompt] * batch_size)
        # compel = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)
        # prompt_embeds = compel([prompt] * batch_size)

        self.images = self.pipe(
            prompt_embeds = prompt_embeds,
            pooled_prompt_embeds = pooled,
            num_inference_steps = steps,
            guidance_scale = guidance,
            height = height if not isSana else 1024,
            width = width if not isSana else 1024,
            generator = generator,
            negative_prompt = [negative_prompt] * batch_size
        ).images
        return self.images

    def set_model(self, model):
        if issubclass(type(model), DiffusionPipeline): self.pipe = model
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16)

        try: self.pipe.enable_vae_slicing()
        except Exception: pass
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

    def optimized_sd3pipeline(self, path):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            path,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.float16
        )
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=torch.float16)
        return pipe

    def sana_sd3pipeline(self, path):
        pipe = SanaPipeline.from_pretrained(
            path,
            torch_dtype=torch.float16
        )
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        pipe.transformer = pipe.transformer.to(torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    def get_grid(self):
        size = int(np.ceil(np.sqrt(len(self.images))))
        return self._image_grid_(self.images, size, size)

    def _image_grid_(self, imgs, rows, cols):
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid

    def generate_from_csv(self, csv_name: str, prompt_idx: str, name_idx: str, replace = False):
        split = str.partition(csv_name, ".")
        if split[-1] == "csv": csv_name = ".".join(csv_name[0:-1])
        frame = pandas.read_csv(csv_name + ".csv")
        idx_prompt = list(zip(frame[name_idx].to_list(), frame[prompt_idx].to_list()))
        img_path = os.path.join(csv_name, str.split(self.pipe.config._name_or_path,"/")[-1])
        Path(img_path).mkdir(parents=True, exist_ok=True)
        for idx,prompt in idx_prompt:
            path = os.path.join(img_path, f"{idx}.webp")
            if not replace and os.path.exists(path): continue
            self.generate(prompt, batch_size=4)
            img = self.get_grid()
            try:
                clear_output(wait=True)
                display(img)
            except Exception: pass
            img.save(path)