import torch
from diffusers import FluxKontextPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.to('cuda')

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

prompt = "Make Pikachu hold a sign that says 'Black Forest Labs is awesome', yarn art style, detailed, vibrant colors"
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png").convert("RGB")

images = pipe(
    prompt,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-flux-dev.png')

##### 2. Process and Save attention map #####
# set unconditional to False for flux
save_attention_maps(attn_maps, pipe.tokenizer, [prompt], base_dir='attn_maps-flux-dev', unconditional=False)
#############################################
