### create image generation pipeline ###

from diffusers import DiffusionPipeline
import torch
import random
import os
import matplotlib.pyplot as plt

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

### generate images ###
# Creating directories for new emotions
os.makedirs('./content/faces/fear', exist_ok=True)  # For faces showing disgust
os.makedirs('./content/faces/confused', exist_ok=True)  # For faces showing disgust

ethnicities = ['a Latino', 'a White', 'a Black', 'a Middle Eastern', 'an Indian', 'an Asian', 'a Southeast Asian', 'a North African']
genders = ['male', 'female']

# Updated emotion_prompts dictionary with added emotions
emotion_prompts = {
    'fear': 'face showing fear: wide open eyes, eyebrows raised and curved, mouth open slightly, facial muscles tense, subtle trembling, cold sweat visible',
    'confused': 'face showing confusion: eyebrows furrowed and drawn together, slight frown, eyes looking off to the side, mouth parted as if mid-question'
}

counter = 0
for j in range(400):
    counter += 1
    for emotion, emotion_prompt in emotion_prompts.items():
        ethnicity = random.choice(ethnicities)
        gender = random.choice(genders)

        prompt = f'Medium-shot portrait of {ethnicity} {gender}, {emotion_prompt}, front view, looking at the camera, ' \
                 f'color photography, ' + \
                 'photorealistic, hyperrealistic, realistic, incredibly detailed, crisp focus, digital art, ' \
                 'depth of field, 50mm, 8k'
        negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), ' \
                          'lowres, normal quality, ((monochrome)), ' + \
                          '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, ' \
                          'bad anatomy, blurred, watermark, grainy, signature'

        img = pipeline(prompt, negative_prompt=negative_prompt).images[0]
        img.save(f'./content/faces/{emotion}/{str(j).zfill(4)}.png')

    print(counter)
