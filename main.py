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

# Settings
prompt = "light and shadow configuration, complexity structure, illumination, artistic artstyle"
number_of_images = 100
folder_path = "generated_images/"

# Check if the folder already exists
if not os.path.exists(folder_path):
    # If it does not exist, create it
    os.makedirs(folder_path)

# Create 100 images
for index in range(100):

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

    file_name = "image_{}.png".format(index)
    full_path = os.path.join(folder_path, file_name)
    # Save image
    image.save(full_path, format="PNG")
