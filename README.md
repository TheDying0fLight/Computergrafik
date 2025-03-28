# Computergraphics Practical Course: Thumbtastic Thumbnail Generator


## Overview
This repository contains resources and scripts for dataset structuring, image generation, and prompt generation for computer graphics projects.


## Usage
Install dependencies using the `requirements.txt` file:
```
  pip install -r requirements.txt
```

 Run the user interface:
```
python src/main.py
```


## Code

### Pipeline and User Interface
- The `src/` directory contains the main scripts for running the user interface.
- The `thumbnail_generator/` subdirectory includes pipeline's modules for dataset creation, image generation, and prompt generation.

### Finetuning and Dataset

- The `notebooks/` directory contains Jupyter notebooks for dataset structuring, pipeline examples, and fine-tuning Moondream.
- The `dataset/` directory contains pre-structured datasets (`.pkl` files) and a JSON file (`first.json`) with various information of the used Youtube videos, and a zipped image dataset containing the downloaded  frames from the Youtube videos (`imgs.zip`).
- The `first/` subdirectory contains the unzipped frames.


## Resources
- [Moondream](https://github.com/vikhyat/moondream)
- [Gemini](https://gemini.google.com/)
- [CLIP](https://github.com/openai/CLIP/)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
