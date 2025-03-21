import flet as ft
from PIL import Image
from pipeline import Pipeline
import numpy as np

def main(page: ft.Page):
    page.pipeline_status = "ready"
    page.title = "Thumbtastic Thumbnail Selector and Generator"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    page.best_frame = None
    page.own_frame = None

    heading = ft.Text("Thumbtastic Thumbnail Selector and Generator", size=40)
    instruction = ft.Text("Create a cartoonish thumbnail by selecting a video frame.", size=20)
    heading_frame_selection = ft.Text("Step 1: Select a suitable frame", size=30)
    page.add(heading, instruction, heading_frame_selection)

    own_frame_heading = ft.Text("Option 1", size=20)
    own_frame_text = ft.Text("You can upload your own frame here.", size=15)
    selected_files = ft.Text()
    selected_img = ft.Container(content=ft.Text("No image selected", size=15))

    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_files.value = ", ".join(f.name for f in e.files)
            selected_files.update()
            selected_img.content = ft.Image(src=e.files[0].path, width=200, height=200, fit=ft.ImageFit.CONTAIN)
            selected_img.update()
            page.own_frame = e.files[0].path
        else:
            selected_files.value = "Cancelled!"
            selected_files.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(pick_files_dialog)

    own_frame_button = ft.ElevatedButton(
        "Upload own frame",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=True)
    )
    own_frame_column = ft.Column(
        controls=[own_frame_heading, own_frame_text, own_frame_button, selected_files, selected_img],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    model_frame_heading = ft.Text("Option 2", size=20)
    model_frame_text = ft.Text("You can enter the video URL and a key sentence below \n and choose a model which will select the best frame.", size=15)
    video_url = ft.TextField(label="Video URL", width=400)
    key_sentence = ft.TextField(label="Key Sentence to describe the video (optional)", width=400)
    frame_amount_text = ft.Text("Amount of frames taken from the video (default: 10)", size=15)
    frame_amount = ft.Slider(min=1, max=50, divisions=49, value=10, label="{value}", width=500)
    model_selection = ft.Dropdown(
        label="Select Model to rate the frames",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream"),
            ft.dropdown.Option("CLIP")
        ],
        width=400
    )
    submit_button = ft.ElevatedButton(text="Start frame selection", on_click=lambda e: call_frame_selection())
    best_frame_column = ft.Column(
        controls=[model_frame_heading, model_frame_text, video_url, key_sentence, frame_amount_text, frame_amount, model_selection, submit_button],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    page.add(ft.Row(
        controls=[own_frame_column, best_frame_column],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START
    ))

    def call_frame_selection():
        if not video_url.value:
            best_frame_column.controls.append(ft.Text("Video URL is required", color="red"))
            best_frame_column.update()
            return
        if not model_selection.value:
            best_frame_column.controls.append(ft.Text("Model selection is required", color="red"))
            best_frame_column.update()
            return
        best_frame_src = frame_selection(key_sentence.value, video_url.value, model_selection.value, frame_amount.value)
        page.best_frame = best_frame_src
        best_frame_img = ft.Image(src=best_frame_src, width=200, height=200, fit=ft.ImageFit.CONTAIN)
        best_frame_column.controls.extend([best_frame_img, ft.Text("This is the frame selected from the video.", size=20)])
        best_frame_column.update()

    def frame_selection(ks, url, model, amount):
        best_frame_column.controls.append(ft.Text("Model generating key sentence from transcript...", size=15))
        best_frame_column.controls.append(ft.Text(f"Downloading the video and extracting {amount} frames...", size=15))
        best_frame_column.controls.append(ft.Text("Model rating the frames...", size=15))
        best_frame_column.update()
        if not ks:
            ks = Pipeline.generate_key_sentence(url, "Gemini" if model == "CLIP" else model, amount)
        frame_rating, frames = Pipeline.rate_frames(ks, url, model, amount)
        if frames is None or frame_rating is None:
            raise ValueError("Failed to generate frame ratings or frames. Please check the input parameters.")
        selected_frame = Pipeline.get_best_frame(frame_rating, frames)
        selected_frame_src = "assets/best_frame.png"
        Image.fromarray(np.uint8(selected_frame)).convert('RGB').save(selected_frame_src)
        return selected_frame_src

    heading_sd = ft.Text("Step 2: Create Cartoonish Thumbnail", size=30)
    instruction_sd = ft.Text("Insert Stable Diffusion styles, separated by commata, and choose the model to generate the prompt for Stable Diffusion.", size=20)
    style_field = ft.TextField(label="Styles, comma-separated, e.g.: photorealistic, gothic, cyberpunk", width=400)
    model_selection_2 = ft.Dropdown(
        label="Select Model to describe the frame",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream")
        ],
        width=400
    )
    submit_button_2 = ft.ElevatedButton(text="Submit", on_click=lambda e: call_thumbnail_generation(style_field.value, model_selection_2.value))
    page.add(heading_sd, instruction_sd, style_field, model_selection_2, submit_button_2)
    sd_out = ft.Column(horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    page.add(sd_out)
    page.update()

    def call_thumbnail_generation(style, model):
        if page.own_frame is not None:
            page.best_frame = page.own_frame
        thumbnail_src = thumbnail_generation(style, model, page.best_frame)
        thumbnail = ft.Image(src=thumbnail_src, width=800, height=450, fit=ft.ImageFit.CONTAIN)
        sd_out.controls.append(thumbnail)
        page.update()
        def save_file_result(e: ft.FilePickerResultEvent):
            save_file_path.value = e.path if e.path else "Cancelled!"
            if save_file_path.value != "Cancelled!":
                try:
                    Image.open(thumbnail_src).save(save_file_path.value)
                except Exception as e:
                    print("error saving the files", e)
            page.update()
        save_file_dialog = ft.FilePicker(on_result=save_file_result)
        save_file_path = ft.Text()
        page.overlay.append(save_file_dialog)
        sd_out.controls.extend([
            ft.ElevatedButton("Save cartoonish thumbnail", icon=ft.Icons.SAVE, on_click=lambda _: save_file_dialog.save_file(), disabled=page.web),
            save_file_path
        ])
        page.update()

    def thumbnail_generation(style, model, best_frame):
        sd_out.controls.clear()
        best_frame_img = Image.open(best_frame)
        sd_out.controls.append(ft.Text("Model describing the frame...", size=15))
        page.update()
        frame_prompt = Pipeline.describe_best_frame(best_frame_img, style, model)
        sd_out.controls[-1].value = f"Model describing the frame: {frame_prompt}"
        page.update()
        thumbnail = Pipeline.generate_thumbnail(frame_prompt)
        thumbnail_src = "assets/thumbnail.png"
        Image.fromarray(np.uint8(thumbnail)).convert('RGB').save(thumbnail_src)
        return thumbnail_src

ft.app(main)