import spaces
import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import random
import uuid
from typing import Tuple
import numpy as np

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max

if not torch.cuda.is_available():
    DESCRIPTIONz += "\n<p>⚠️Running on CPU, This may not work on CPU.</p>"

base_model = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

lora_repo = "strangerzonehf/Sketch-Paint"
trigger_word = "Sketch paint"  # Leave trigger_word blank if not used.

pipe.load_lora_weights(lora_repo)
pipe.to("cuda")

style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
    },
    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
    },
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
    },
]

styles = {k["name"]: k["prompt"] for k in style_list}

DEFAULT_STYLE_NAME = "3840 x 2160"
STYLE_NAMES = list(styles.keys())

def apply_style(style_name: str, positive: str) -> str:
    return styles.get(style_name, styles[DEFAULT_STYLE_NAME]).replace("{prompt}", positive)

@spaces.GPU(duration=60, enable_queue=True)
def generate(
    prompt: str,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_name: str = DEFAULT_STYLE_NAME,
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))

    positive_prompt = apply_style(style_name, prompt)
    
    if trigger_word:
        positive_prompt = f"{trigger_word} {positive_prompt}"

    images = pipe(
        prompt=positive_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=30,
        num_images_per_prompt=1,
        output_type="pil",
    ).images
    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed

examples = [

    "midjourney mix, a tiny astronaut hatching from an egg on the moon",
    "midjourney mix, intense Red, a black cat is facing the left side of the frame. The cats head is tilted upward, with its eyes closed. Its whiskers are protruding from its mouth, adding a touch of warmth to the scene. The background is a vibrant red, creating a striking contrast with the cats fur.",
    "midjourney mix, a close-up shot of a womans face, the womans hair is wet, and she is wearing a cream-colored sweater. The background is blurred, and there are red and white signs visible in the background. The womans eyebrows are wet, adding a touch of color to her face. Her lips are a vibrant shade of pink, and her eyes are a darker shade of brown.",
    "midjourney mix, woman in a red jacket, snowy, in the style of hyper-realistic portraiture, caninecore, mountainous vistas, timeless beauty, palewave, iconic, distinctive noses --ar 72:101 --stylize 750 --v 6",
    "midjourney mix, an anime-style illustration of a delicious, golden-brown wiener schnitzel on a plate, served with fresh lemon slices, parsley --style raw5"
]

css = '''
.gradio-container{max-width: 888px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
.submit-btn {
    background-color: #d73333 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #ff0000 !important;
}
'''

with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Generate as : ( 1280 x 832 )🤗", scale=0, elem_classes="submit-btn")
            
            with gr.Accordion("Advanced options", open=True, visible=True):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                    visible=True
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                
                with gr.Row(visible=True):
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1280,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=832,
                    )
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=3.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=40,
                        step=1,
                        value=30,
                    )

                style_selection = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                    label="Quality Style",
                )
        
        with gr.Column(scale=2):
            result = gr.Gallery(label="Result", columns=1, show_label=False)
            
            gr.Examples(
                examples=examples,
                inputs=prompt,
                outputs=[result, seed],
                fn=generate,
                cache_examples=False,
            )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            style_selection,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=40).launch()