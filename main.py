import time
from diffusers import DiffusionPipeline
import torch
import os

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

object_list = ["cube", "sphere", "cylinder", "cone", "torus"]
material_list = ["cardboard", "pvc", "plaster", "rubber", "stone", "cloth", "wood"]


# Create directory if it doesn't exist
folder_name = "images_{}".format(time())
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Create 100 images
for index in range(100):
    for object_type in object_list:
        for material_type in material_list:

            prompt = f"even flat {material_type} diffuse ground surface, one solid {object_type} made out of an even flat {material_type} diffuse material"

            # Run both experts
            image = base(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
            ).images
            image = refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images[0]

            file_name = "image_{}_{}_{}.png".format(object_type, material_type, index)
            full_path = os.path.join(folder_name, file_name)
            
            # Save image
            image.save(full_path, format="PNG")
