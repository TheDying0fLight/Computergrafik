import flet as ft
from PIL import Image 
import PIL

def main(page: ft.Page):
    page.pipeline_status = "ready"
    page.title = "Thumbtastic Thumbnail Selector and Generator"

    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    heading = ft.Text("Welcome to Thumbtastic Thumbnail Selector and Generator", size=40)

    heading_frame_selection = ft.Text("Select a suitable frame", size=30)
    instruction = ft.Text("Please enter the key sentence and the video URL below and choose a model which will select the best frame.", size=20)
    page.add(heading, instruction)
    key_sentence = ft.TextField(label="Key Sentence to describe the video (optional)", width=400)
    video_url = ft.TextField(label="Video URL", width=400)
    
    model_selection = ft.Dropdown(
        label="Select Model",
        options=[
            ft.dropdown.Option("Gemini"),
            ft.dropdown.Option("Moondream"),
            ft.dropdown.Option("Finetuned Moondream"),
        ],
        width=400
    )
    
    submit_button = ft.ElevatedButton(text="Submit", on_click=lambda e: submit_action())
    
    page.add(heading_frame_selection, key_sentence, video_url, model_selection, submit_button)


    def submit_action():

        def pick_files_result(e: ft.FilePickerResultEvent):
            selected_files.value = (
                ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
            )
            selected_files.update()

        if not video_url.value:
            page.add(ft.Text("Video URL is required", color="red"))
            return
        if not model_selection.value:
            page.add(ft.Text("Model selection is required", color="red"))
            return
        
        progress_status = ft.Text("Selecting frames...", size=20, data=0)
        page.add(progress_status)

        # Here you can handle the submission logic
        print(f"Key Sentence: {key_sentence.value}")
        print(f"Video URL: {video_url.value}")
        print(f"Selected Model: {model_selection.value}")
        
        call_pipeline(key_sentence.value, video_url.value, model_selection.value)
        
        if page.pipeline_status == "ready":
            # show selected frame
            best_frame = ft.Image(src="assets/best_frame.png", width=100, height=100, fit=ft.ImageFit.CONTAIN,)
            best_frame_text = ft.Text("This is the frame selected from the video. You can generate a cartoonish version of it below.", size=20)
            page.add(best_frame, best_frame_text)
            
            # opportunity to select a different frame
            page.add(ft.Text("If you don't like the selected frame, you can upload your own frame to create a cartoonish version of it.", size=20))
                       
            pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
            selected_files = ft.Text()

            page.overlay.append(pick_files_dialog)

            page.add(ft.ElevatedButton(
                            "Pick files",
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=lambda _: pick_files_dialog.pick_files(
                                allow_multiple=True
                            ),
                        ),
                        selected_files)
            
            #page.add(ft.Image(src=...)# TODO: show selected file
            
            heading_SD = ft.Text("Create Cartoonish Version", size=30)
            instruction_SD = ft.Text("Optional: Insert Stable Diffusion styles, separated by commata, and/or choose the model.", size=20)
            style = ft.TextField(label="Styles, comma-separated, e.g.: photorealistic, gothic, cyberpunk", width=400)
            model_selection_2 = ft.Dropdown(
                label="Select Model",
                options=[
                    ft.dropdown.Option("Gemini"),
                    ft.dropdown.Option("Moondream"),
                    ft.dropdown.Option("Finetuned Moondream"),
                ],
                width=400
            )
            submit_button_2 = ft.ElevatedButton(text="Submit", on_click=lambda e: generate_artificial_thumbnail())
            
            page.add(heading_SD, instruction_SD, style, model_selection_2, submit_button_2)
            
            if page.pipeline_status == "ready":
                # show generated thumbnail
                page.update()            

            page.update()


    def generate_artificial_thumbnail():
        pass
        # Here you can call the pipeline
        page.pipeline_status = "busy"
        explanation_SD = ft.Text("Prompt is created and Stable Diffusion creates the cartoonish version...", size=20)
        page.add(explanation_SD)

        generated_thumbnail = ft.Image(src="assets/generated_thumbnail.png", width=100, height=100, fit=ft.ImageFit.CONTAIN) # TODO
        explanation_cartoonish = ft.Text("This is the cartoonish thumbnail generated by Stabe Diffusion.", size=20)
        page.add(generated_thumbnail, explanation_cartoonish)

        explanation_saving = ft.Text("You can save the cartoonish thumbnail below.", size=20)
        page.add(explanation_saving)

        # Save file dialog
        def save_file_result(e: ft.FilePickerResultEvent):
            save_file_path.value = e.path if e.path else "Cancelled!"
            if save_file_path.value != "Cancelled!":
                try:
                    with Image.open(generated_thumbnail.src) as f:
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

        page.pipeline_status = "ready"

    def call_pipeline(key_sentence, video_url, model_selection):
        # Here we can call the pipeline
        page.pipeline_status = "busy"
        # ...
        page.pipeline_status = "ready"

ft.app(main, assets_dir="assets", upload_dir="assets/uploads")
