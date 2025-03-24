import flet as ft
from PIL import Image
from pipeline import Pipeline
import numpy as np
import os
import io
import base64

def main(page: ft.Page):
    page.pipeline_status = "ready"
    page.title = "Thumbtastic Thumbnail Selector and Generator"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.scroll = ft.ScrollMode.AUTO
    page.best_frame = None
    page.own_frame = None
    page.uploaded_frames = []

    # Lists to store container options for selection
    uploaded_frames = []
    rated_frames = []

    # --- Header ---
    header = ft.Column(
        controls=[
            ft.Text("Thumbtastic Thumbnail Selector and Generator", size=40, weight="bold"),
            ft.Text("Create a cartoonish thumbnail by selecting a video frame.", size=20)
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10
    )
    page.add(header)

    # --- Step 1: Frame Selection ---
    step1_header = ft.Text("Step 1: Select a Suitable Frame", size=30, weight="bold")
    step1_divider = ft.Divider(height=2, thickness=2)

    # Option 1: Upload your own frame
    own_frame_heading = ft.Text("Option 1: Upload Your Own Frame", size=20, weight="bold")
    own_frame_instruction = ft.Text("You can upload one or more frames here.", size=15)
    # Row to display uploaded images
    uploaded_images_row = ft.Row(controls=[], wrap=True, width=1000, scroll=ft.ScrollMode.AUTO, height=400)
    own_frame_button = ft.ElevatedButton(
        "Upload Own Frame",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=True)
    )
    own_frame_column = ft.Column(
        controls=[own_frame_heading, own_frame_instruction, own_frame_button, uploaded_images_row],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    uploaded_container = ft.Container(
        content=own_frame_column,
        padding=20,
        border=ft.border.all(1, "lightgray"),
        border_radius=10,
        margin=10,
        expand=True
    )

    # Option 2: Automatic Frame Selection
    model_frame_heading = ft.Text("Option 2: Automatic Frame Selection", size=20, weight="bold")
    model_frame_instruction = ft.Text(
        "Enter the video URL, key sentence (optional) and choose a model which will select the best frame.",
        size=15
    )
    video_url = ft.TextField(label="Video URL", width=400)
    key_sentence = ft.TextField(label="Key Sentence (optional)", width=400)

    # Slider using logarithmic scale
    frame_amt_text = "Amount of frames (>100 only recommended with CLIP): "
    frame_amount_text = ft.Text(f"{frame_amt_text}10", size=15)
    frame_amount = ft.Slider(
        min=0,
        max=4,
        divisions=400,
        value=1,
        label="{value}",
        width=500,
        on_change=lambda _: update_frame_amount_text(frame_amount, frame_amount_text)
    )

    def update_frame_amount_text(slider, text_widget):
        effective_value = int(10 ** slider.value)
        text_widget.value = f"{frame_amt_text}{effective_value}"
        text_widget.update()

    model_selection = ft.Dropdown(
        label="Select Model to Rate the Frames",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream"),
            ft.dropdown.Option("CLIP")
        ],
        width=400,
        key="model_selection"
    )
    submit_button = ft.ElevatedButton(text="Rate Frames", on_click=lambda _: call_frame_selection())

    # Column to display the best frame (from either uploaded images or rated frames)
    best_frame_column = ft.Column(
        controls=[],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10
    )
    model_frame_column = ft.Column(
        controls=[model_frame_heading, model_frame_instruction, video_url, key_sentence,
                  frame_amount_text, frame_amount, model_selection, submit_button, best_frame_column],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    rated_container = ft.Container(
        content=model_frame_column,
        padding=20,
        border=ft.border.all(1, "lightgray"),
        border_radius=10,
        margin=10,
        expand=True
    )

    # Place both options side-by-side
    options_row = ft.Row(
        controls=[uploaded_container, rated_container],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START
    )

    selected_column = ft.Column(
        controls=[],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10
    )

    page.add(step1_header, step1_divider, options_row, selected_column)

    # File picker for uploading frames
    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            # Process each uploaded file
            for file in e.files:
                # Create a container that shows the image and a remove button
                container = ft.Container(
                    content = ft.Column(
                        controls=[
                            ft.Image(src=file.path, width=160, height=90, fit=ft.ImageFit.CONTAIN)
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                    padding=5,
                    border=ft.border.all(1, "lightgray"),
                    border_radius=5,
                    key=f"uploaded_{file.name}"
                )
                container.content.controls.append(
                    ft.IconButton(
                        icon=ft.icons.CLOSE,
                        tooltip="Remove this image",
                        on_click=lambda _, cont=container, f=file: remove_uploaded_image(cont, f)
                    )
                )
                container.on_click = lambda e, file=file, cont=container: select_image(file.path, cont)
                uploaded_frames.append(container)
                uploaded_images_row.controls.append(container)
            uploaded_images_row.update()
        else: pass

    def remove_uploaded_image(container, file):
        nonlocal uploaded_frames
        if container in uploaded_frames:
            uploaded_frames.remove(container)
        if container in uploaded_images_row.controls:
            uploaded_images_row.controls.remove(container)
        if page.best_frame == file.path: page.best_frame = None
        page.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(pick_files_dialog)

    def select_image(file_path, selected_container):
        page.best_frame = file_path
        for cont in rated_frames + uploaded_frames:
            cont.border = ft.border.all(3, "blue") if cont.key == selected_container.key else ft.border.all(1, "lightgray")
            cont.update()

        # Check if preview exists; if not, add one; otherwise, update it
        if not any(isinstance(ctrl, ft.Column) and ctrl.key == "selected_preview" for ctrl in best_frame_column.controls):
            preview = ft.Column(
                controls=[
                    ft.Image(src=file_path, width=160*3, height=90*3, fit=ft.ImageFit.CONTAIN),
                    ft.Text("This is your selected frame.", size=20)
                ],
                key="selected_preview",
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
            selected_column.controls = [preview]
        else:
            for ctrl in best_frame_column.controls:
                if isinstance(ctrl, ft.Column) and ctrl.key == "selected_preview":
                    ctrl.controls[0].src = file_path
                    ctrl.controls[0].update()
                    break
        page.update()

    def call_frame_selection():
        nonlocal rated_frames
        best_frame_column.controls.clear()
        rated_frames.clear()

        if not video_url.value:
            best_frame_column.controls.append(ft.Text("Video URL is required", color="red"))
            best_frame_column.update()
            return
        if not model_selection.value:
            best_frame_column.controls.append(ft.Text("Model selection is required", color="red"))
            best_frame_column.update()
            return

        groups = frame_selection(
            key_sentence.value, video_url.value, model_selection.value, int(10 ** frame_amount.value), best_frame_column
        )
        if not groups:
            best_frame_column.controls.append(ft.Text("No frame groups returned.", color="red"))
            best_frame_column.update()
            return

        best_frame_column.controls.clear()
        best_frame_column.controls.append(ft.Text("Select a frame from the options below:", size=20))
        rated_frames = []
        rated_row = ft.Row(controls=[], wrap=True, width=1000, scroll=ft.ScrollMode.AUTO, height=400)
        for idx, group in enumerate(groups):
            if not group: continue
            score, img = group[0]
            temp_file = f"assets/group_{idx}.png"
            os.makedirs("assets", exist_ok=True)
            Image.fromarray(np.uint8(img)).convert('RGB').save(temp_file)
            frame_image = ft.Image(src=temp_file, width=160, height=90, fit=ft.ImageFit.CONTAIN)
            rating_text = ft.Text(f"Rating: {score:.2f}", size=15)
            frame_column = ft.Column(
                controls=[frame_image, rating_text],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
            container = ft.Container(
                content=frame_column,
                padding=5,
                border=ft.border.all(1, "lightgray"),
                border_radius=5,
                key=f"rated_{idx}"
            )
            container.on_click = lambda _, file=temp_file, cont=container: select_image(file, cont)
            rated_frames.append(container)
            rated_row.controls.append(container)
        best_frame_column.controls.append(rated_row)
        best_frame_column.update()

    def frame_selection(ks, url, model, amount, feedback_column):
        feedback_column.controls.append(ft.Text("Generating key sentence...", size=15))
        feedback_column.update()

        if not ks:
            ks = Pipeline.generate_key_sentence(url, "Gemini" if model == "CLIP" else model, amount)
        feedback_column.controls.append(ft.Text(f"Downloading video, extracting, and rating {amount} frames...", size=15))
        feedback_column.update()
        groups = Pipeline.rate_frames(ks, url, model, amount)
        if groups is None or not groups:
            raise ValueError("Failed to generate frame groups. Please check the input parameters.")
        return groups

    # --- Step 2: Thumbnail Generation ---
    step2_header = ft.Text("Step 2: Create Stylized Thumbnail", size=30, weight="bold")
    step2_divider = ft.Divider(height=2, thickness=2)
    step2_instruction = ft.Text(
        "Insert Stable Diffusion styles (comma-separated) and choose the model to generate the prompt.", size=20
    )
    style_field = ft.TextField(label="Styles (e.g.: photorealistic, gothic, cyberpunk)", width=400)
    model_selection_2 = ft.Dropdown(
        label="Select Model to Describe the Frame",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream")
        ],
        width=400,
        key="model_selection_2"
    )
    submit_button_2 = ft.ElevatedButton(text="Submit", on_click=lambda e: call_thumbnail_generation(
        style_field.value, model_selection_2.value))
    thumbnail_section = ft.Column(
        controls=[step2_instruction, style_field, model_selection_2, submit_button_2],
        spacing=15,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    thumbnail_container = ft.Container(
        content=thumbnail_section,
        padding=20,
        border=ft.border.all(1, "lightgray"),
        border_radius=10,
        margin=20,
        expand=True
    )
    page.add(step2_header, step2_divider, thumbnail_container)

    sd_out = ft.Column(horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)
    page.add(sd_out)
    page.update()

    def call_thumbnail_generation(style, model):
        if page.own_frame is not None and page.best_frame is None:
            page.best_frame = page.own_frame
        if page.best_frame is None:
            sd_out.controls.clear()
            sd_out.controls.append(ft.Text("Please select a frame first.", color="red"))
            sd_out.update()
            return
        thumb_b64 = thumbnail_generation(style, model, page.best_frame)
        sd_out.controls.clear()
        sd_out.controls.append(ft.Image(src_base64=thumb_b64, width=800, height=450, fit=ft.ImageFit.CONTAIN))
        page.update()

        def save_file_result(e: ft.FilePickerResultEvent):
            if e.path:
                try:
                    Image.open(io.BytesIO(base64.b64decode(thumb_b64))).save(e.path)
                except Exception as e:
                    print("Error saving file:", e)
            page.update()

        save_file_dialog = ft.FilePicker(on_result=save_file_result)
        page.overlay.append(save_file_dialog)
        sd_out.controls.extend([
            ft.ElevatedButton("Save Thumbnail", icon=ft.Icons.SAVE,
                              on_click=lambda _: save_file_dialog.save_file(), disabled=page.web)
        ])
        page.update()

    def thumbnail_generation(style, model, frame):
        sd_out.controls.clear()
        pil_frame = Image.open(frame)
        sd_out.controls.append(ft.Text("Model describing the frame...", size=15))
        page.update()
        frame_prompt = Pipeline.describe_frame(pil_frame, style, model)
        sd_out.controls[-1].value = f"Frame description: {frame_prompt}"
        page.update()
        thumbnail = Pipeline.generate_thumbnail(frame_prompt)
        pil_thumbnail = Image.fromarray(np.uint8(thumbnail)).convert('RGB')
        thumbnail_b64 = pil_to_base64(pil_thumbnail)
        return thumbnail_b64

def pil_to_base64(pil_image, format="PNG"):
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

ft.app(main)
