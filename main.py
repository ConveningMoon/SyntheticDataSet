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
os.makedirs('./content/faces/happy', exist_ok=True)
os.makedirs('./content/faces/sad', exist_ok=True)
os.makedirs('./content/faces/angry', exist_ok=True)
os.makedirs('./content/faces/surprised', exist_ok=True)
os.makedirs('./content/faces/bored', exist_ok=True)  # For bored faces
os.makedirs('./content/faces/tired', exist_ok=True)  # For tired faces
os.makedirs('./content/faces/neutral', exist_ok=True)  # For neutral faces
os.makedirs('./content/faces/disgust', exist_ok=True)  # For faces showing disgust

ethnicities = ['a latino', 'a white', 'a black', 'a middle eastern', 'an indian', 'an asian']
genders = ['male', 'female']

# Updated emotion_prompts dictionary with added emotions
emotion_prompts = {'happy': 'smiling',
                   'sad': 'frowning, sad face expression, crying, emotional, depressed',
                   'surprised': 'surprised, opened mouth, raised eyebrows',
                   'angry': 'angry',
                   'bored': 'bored, lack of interest, dull expression, downturned mouth and half-eyed gaze',
                   'tired': 'tired, yawning, droopy eyes',
                   'neutral': 'neutral expression, straight face, looking directly at the camera',
                   'disgust': 'disgusted, scrunched nose, The lips curling into a snarling frown, the nostrils '
                              'flaring, the cheeks pushing up to squinting lower lids and a furrowed brow'}
counter = 0
for j in range(2000):
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
