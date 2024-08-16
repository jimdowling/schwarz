import gradio as gr
from PIL import Image
import hopsworks

name="prophet"
img="fig2.png"
project = hopsworks.login()

dataset_api = project.get_dataset_api()
mr = project.get_model_registry()

#gets the latest version number for the model
version = mr.get_models(name)[-1].version

def show_image(img):
    img_path = f"Models/{name}/{version}/images/{img}.png"

    downloaded_file_path = dataset_api.download(img_path, overwrite=True)
    image = Image.open(f"{img}.png")
    return image

iface = gr.Interface(
    fn=show_image,
    inputs="textbox",
    outputs="image",
    title="Display PNG Image",
)

iface.launch(share=True)
