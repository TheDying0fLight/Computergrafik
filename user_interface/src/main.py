import flet as ft
from PIL import Image 
import PIL
from pipeline import Pipeline
import numpy as np

# example url to test with, deadpool trailer: https://www.youtube.com/watch?v=Xithigfg7dA

def main(page: ft.Page):
    page.pipeline_status = "ready"
    page.title = "Thumbtastic Thumbnail Selector and Generator"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO

    page.best_frame = None
    page.own_frame = None

    # heading
    heading = ft.Text("Thumbtastic Thumbnail Selector and Generator", size=40)
    heading_frame_selection = ft.Text("Step 1: Select a suitable frame", size=30)
    instruction = ft.Text("Create a cartoonish thumbnail by selecting a video frame.", size=20)
    page.add(heading, instruction, heading_frame_selection)
    
    #########################
    # step 1: select frame
    #########################

    # option 1: select own frame
    own_frame_heading = ft.Text("Option 1", size=20)
    own_frame_text = ft.Text("You can upload your own frame here.", size=15)
                
    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            # Update the selected files text
            selected_files.value = ", ".join(map(lambda f: f.name, e.files))
            selected_files.update()

            # Display the selected image
            selected_img.src = e.files[0].path  # Set the image source to the selected file path
            selected_img.update()
        else:
            selected_files.value = "Cancelled!"
            selected_files.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text()
    selected_img = ft.Image(src = "assets/placeholder.png", width=300, height=300, fit=ft.ImageFit.CONTAIN)

    page.overlay.append(pick_files_dialog)

    own_frame_button = ft.ElevatedButton(
                    "Upload own frame",
                    icon=ft.Icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=True))

    # Create columns for own frame selection and best frame selection
    own_frame_column = ft.Column(
        controls=[
            own_frame_heading,
            own_frame_text,
            own_frame_button,
            selected_files,
            selected_img
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    # option 2: give video url and key sentence
    model_frame_heading = ft.Text("Option 2", size=20)
    model_frame_text = ft.Text("You can enter the video URL and a key sentence below \n and choose a model which will select the best frame.", size=15)
    
    video_url = ft.TextField(label="Video URL", width=400)
    key_sentence = ft.TextField(label="Key Sentence to describe the video (optional)", width=400)
    frame_amount_text = ft.Text("Amount of frames taken from the video (default: 10)", size=15)
    frame_amount = ft.Slider(
        min=1,
        max=50,
        divisions=49,
        value=10,
        label="{value}",
        width=500,
        #on_change=spacing_slider_change,
    )
    
    model_selection = ft.Dropdown(
        label="Select Model to rate the frames",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream"),
            ft.dropdown.Option("CLIP"),
        ],
        width=400
    )
    
    submit_button = ft.ElevatedButton(text="Start frame selection", on_click=lambda e: call_frame_selection())
    

    # Create a column for the model frame selection
    best_frame_column = ft.Column(
        controls=[
            model_frame_heading,
            model_frame_text,
            video_url,
            key_sentence,
            frame_amount_text,
            frame_amount,
            model_selection,
            submit_button
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    # Add the columns to a row
    page.add(ft.Row(
        controls=[
            own_frame_column,
            best_frame_column
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START
    ))

    ##################################################
    # step 2: create cartoonish version
    ##################################################

    heading_SD = ft.Text("Step 2: Create Cartoonish Thumbnail", size=30)
    instruction_SD = ft.Text("Optional: Insert Stable Diffusion styles, separated by commata, and/or choose the model.", size=20)
    style = ft.TextField(label="Styles, comma-separated, e.g.: photorealistic, gothic, cyberpunk", width=400)
    model_selection_2 = ft.Dropdown(
        label="Select Model to describe the frame",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream"),
        ],
        width=400
    )
    submit_button_2 = ft.ElevatedButton(text="Submit", on_click=lambda e: call_thumbnail_generation(style.value, model_selection_2.value))
    
    page.add(heading_SD, instruction_SD, style, model_selection_2, submit_button_2)
    
    page.update()
    

    def call_frame_selection():

        if not video_url.value:
            best_frame_column.controls.append(ft.Text("Video URL is required", color="red"))
            best_frame_column.update()
            return
            
        if not model_selection.value:
            best_frame_column.controls.append(ft.Text("Model selection is required", color="red"))
            best_frame_column.update()
            return
        
        progress_status = ft.Text("Selecting frames...", size=20, data=0)
        best_frame_column.controls.append(progress_status)
        best_frame_column.update()

        # Here you can handle the submission logic
        print(f"Key Sentence: {key_sentence.value}")
        print(f"Video URL: {video_url.value}")
        print(f"Selected Model: {model_selection.value}")
        
        best_frame_src = frame_selection(key_sentence.value, video_url.value, model_selection.value, frame_amount.value)
        page.best_frame = best_frame_src
        best_frame = ft.Image(src=best_frame_src, width=100, height=100, fit=ft.ImageFit.CONTAIN)
        
        best_frame_text = ft.Text("This is the frame selected from the video.", size=20)
        best_frame_column.controls.extend([best_frame, best_frame_text])
        best_frame_column.update()
        
        return

    def frame_selection(key_sentence, video_url, model_selection, frame_amount):
        
        if model_selection == "CLIP":
            # clip only uses embedding comparisons for ratings
            frame_rating, frames = Pipeline.rate_frames(key_sentence, video_url, model_selection, frame_amount)
            print("FRAME RATING:", frame_rating)
            print("FRAMES:", frames)

        if model_selection != "CLIP" and not key_sentence:
            # other models (LLMs) need a key sentence with which they can compare and rate the frames
            key_sentence = Pipeline.generate_key_sentence(video_url, model_selection, frame_amount)
            frame_rating, frames = Pipeline.rate_frames(key_sentence, video_url, model_selection, frame_amount)
            
            print("KEY SENTENCE:", key_sentence)
            print("FRAME RATING:", frame_rating)
            print("FRAMES:", len(frames))
        
        if frames is None or frame_rating is None:
            raise ValueError("Failed to generate frame ratings or frames. Please check the input parameters.")

        selected_frame = Pipeline.get_best_frame(frame_rating, frames)
        #print(selected_frame)
        selected_frame_src = "assets/best_frame.png"
        with Image.fromarray(np.uint8(selected_frame)).convert('RGB') as f:
            f = f.save(selected_frame_src)

        return selected_frame_src


    def call_thumbnail_generation(style, model_selection):
                
        if page.own_frame is not None:
            page.best_frame = page.own_frame

        explanation_SD = ft.Text("Prompt is created and Stable Diffusion creates the cartoonish version...", size=20)
        page.add(explanation_SD)

        thumbnail_src = generate_thumbnail(style, model_selection, page.best_frame)
        thumbnail = ft.Image(src=thumbnail_src, width=100, height=100, fit=ft.ImageFit.CONTAIN)
        thumbnail_text = ft.Text("This is the cartoonish thumbnail generated by Stable Diffusion.", size=20)
        page.add(thumbnail, thumbnail_text)

        # Save the thumbnail
        def save_file_result(e: ft.FilePickerResultEvent):
            save_file_path.value = e.path if e.path else "Cancelled!"
            if save_file_path.value != "Cancelled!":
                try:
                    with Image.open(thumbnail.src) as f:
                        f = f.save(save_file_path.value)
                except Exception as e:
                    print("error saving the files", e)
            page.update()

        save_file_dialog = ft.FilePicker(on_result=save_file_result)
        save_file_path = ft.Text()    
        page.overlay.append(save_file_dialog)
        page.add(ft.ElevatedButton(
                    "Save cartoonish thumbnail",
                    icon=ft.Icons.SAVE,
                    on_click=lambda _: save_file_dialog.save_file(),
                    disabled=page.web,
                ),
                save_file_path)

        return
    
    def generate_thumbnail(style, model_selection, best_frame):
        frame_prompt = Pipeline.describe_best_frame(style, model_selection, best_frame)
        thumbnail = Pipeline.generate_thumbnail(frame_prompt)
        thumbnail_src = "assets/thumbnail.png"
        with Image.fromarray(np.uint8(thumbnail)).convert('RGB') as f:
            f = f.save(thumbnail_src)

        return thumbnail_src
    
ft.app(main, assets_dir="assets", upload_dir="assets/uploads")
