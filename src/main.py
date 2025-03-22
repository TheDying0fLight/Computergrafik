import flet as ft
from PIL import Image
from pipeline import Pipeline
import numpy as np
import os

def main(page: ft.Page):
    page.pipeline_status = "ready"
    page.title = "Thumbtastic Thumbnail Selector and Generator"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.scroll = ft.ScrollMode.AUTO
    page.best_frame = None
    page.own_frame = None

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
    own_frame_instruction = ft.Text("You can upload your own frame here.", size=15)
    selected_files = ft.Text()
    selected_img = ft.Container(content=ft.Text("No image selected", size=15), padding=10)
    own_frame_button = ft.ElevatedButton(
        "Upload Own Frame",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=False)
    )
    own_frame_column = ft.Column(
        controls=[own_frame_heading, own_frame_instruction, own_frame_button, selected_files, selected_img],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    own_frame_container = ft.Container(
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
    frame_amount_text = ft.Text("Amount of frames (default: 10)", size=15)
    frame_amount = ft.Slider(min=0, max=1000, divisions=100, value=20, label="{value}", width=500)
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
    submit_button = ft.ElevatedButton(text="Start Frame Selection", on_click=lambda e: call_frame_selection())
    model_frame_column = ft.Column(
        controls=[model_frame_heading, model_frame_instruction, video_url, key_sentence,
                  frame_amount_text, frame_amount, model_selection, submit_button],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    model_frame_container = ft.Container(
        content=model_frame_column,
        padding=20,
        border=ft.border.all(1, "lightgray"),
        border_radius=10,
        margin=10,
        expand=True
    )

    # Place both options side-by-side
    options_row = ft.Row(
        controls=[own_frame_container, model_frame_container],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START
    )

    page.add(step1_header, step1_divider, options_row)

    # File picker for uploading frame
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

    # Column to display the best frame selected by the model (if applicable)
    best_frame_column = ft.Column(
        controls=[],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10
    )
    page.add(best_frame_column)

    def call_frame_selection():
        best_frame_column.controls.clear()

        if not video_url.value:
            best_frame_column.controls.append(ft.Text("Video URL is required", color="red"))
            best_frame_column.update()
            return
        if not model_selection.value:
            best_frame_column.controls.append(ft.Text("Model selection is required", color="red"))
            best_frame_column.update()
            return

        # Call the updated frame selection which now returns groups of frames
        groups = frame_selection(
            key_sentence.value, video_url.value, model_selection.value, frame_amount.value, best_frame_column
        )
        if not groups:
            best_frame_column.controls.append(ft.Text("No frame groups returned.", color="red"))
            best_frame_column.update()
            return

        best_frame_column.controls.clear()
        best_frame_column.controls.append(ft.Text("Select a frame from the options below:", size=20))
        # For each group, display the first image as a selectable button.
        group_buttons = []
        print(groups)
        for idx, group in enumerate(groups):
            if not group:
                continue
            print(group)
            score, img = group[0]  # Use the first image from each group
            temp_file = f"assets/group_{idx}.png"
            # Ensure assets directory exists
            os.makedirs("assets", exist_ok=True)
            Image.fromarray(np.uint8(img)).convert('RGB').save(temp_file)
            btn = ft.ElevatedButton(
                content=ft.Image(src=temp_file, width=200, height=200, fit=ft.ImageFit.CONTAIN),
                on_click=lambda e, file=temp_file: select_group_frame(file)
            )
            group_buttons.append(btn)
        best_frame_column.controls.append(ft.Row(controls=group_buttons, wrap=True))
        best_frame_column.update()

    def select_group_frame(selected_file):
        # Save the selected frame as the best frame
        page.best_frame = selected_file
        best_frame_column.controls.clear()
        best_frame_column.controls.extend([
            ft.Image(src=selected_file, width=200, height=200, fit=ft.ImageFit.CONTAIN),
            ft.Text("This is the selected frame from the video.", size=20)
        ])
        best_frame_column.update()

    def frame_selection(ks, url, model, amount, feedback_column):
        # Provide progress updates to the feedback_column
        feedback_column.controls.append(ft.Text("Generating key sentence...", size=15))
        feedback_column.update()

        # If no key sentence provided, generate one.
        if not ks:
            ks = Pipeline.generate_key_sentence(url, "Gemini" if model == "CLIP" else model, amount)
        # Now, rate_frames returns groups of frames (list of lists of (score, image) tuples)
        feedback_column.controls.append(ft.Text(f"Downloading video and extracting {amount} frames...", size=15))
        feedback_column.controls.append(ft.Text("Rating frames...", size=15))
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

    # Column to display the generated thumbnail and save option
    sd_out = ft.Column(horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)
    page.add(sd_out)
    page.update()

    def call_thumbnail_generation(style, model):
        if page.own_frame is not None:
            page.best_frame = page.own_frame
        if page.best_frame is None:
            sd_out.controls.clear()
            sd_out.controls.append(ft.Text("Please select a frame first.", color="red"))
            sd_out.update()
            return
        thumbnail_src = thumbnail_generation(style, model, page.best_frame)
        sd_out.controls.clear()
        sd_out.controls.append(ft.Image(src=thumbnail_src, width=800, height=450, fit=ft.ImageFit.CONTAIN))
        sd_out.controls.append(ft.Text("Thumbnail generated.", size=20))
        page.update()

        def save_file_result(e: ft.FilePickerResultEvent):
            if e.path:
                try:
                    Image.open(thumbnail_src).save(e.path)
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

    def thumbnail_generation(style, model, best_frame):
        sd_out.controls.clear()
        best_frame_img = Image.open(best_frame)
        sd_out.controls.append(ft.Text("Model describing the frame...", size=15))
        page.update()
        frame_prompt = Pipeline.describe_frame(best_frame_img, style, model)
        sd_out.controls[-1].value = f"Model describing the frame: {frame_prompt}"
        page.update()
        thumbnail = Pipeline.generate_thumbnail(frame_prompt)
        thumbnail_src = "assets/thumbnail.png"
        Image.fromarray(np.uint8(thumbnail)).convert('RGB').save(thumbnail_src)
        return thumbnail_src

ft.app(main)
