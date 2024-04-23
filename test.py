### create image generation pipeline ###

from diffusers import DiffusionPipeline
import torch
import random
import os

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

### generate images ###
# Ensuring directories for all emotions exist
emotions = ['happy', 'smiling', 'neutral', 'surprised', 'angry', 'sad']
for emotion in emotions:
    os.makedirs(f'./content/faces/{emotion}', exist_ok=True)

# Expanding the ethnicities to cover a more global representation
ethnicities = ['a latino', 'a white', 'a black', 'a middle eastern', 'an indian', 'an asian', 'an african',
               'an eastern european', 'a southeast asian', 'a pacific islander']
genders = ['male', 'female']

# Correcting the emotion_prompt dictionary to ensure clear representation of emotions
emotion_prompts = {
    'happy': 'smiling with joy, a happy face expression, eyes crinkling, wide beaming smile, visible teeth',
    'smiling': 'content, slight smile with no visible teeth, soft eyes, closed mouth',
    'neutral': 'neutral, straight face, relaxed features, looking directly at the camera',
    'surprised': 'surprised, mouth wide open, raised eyebrows, eyes wide open',
    'angry': 'angry, furious, frowning, narrowed eyes, tight lips, flared nostrils',
    'sad': 'sad face expression, crying, emotional, depressed, intense sadness'
}

# Define the age range for the generated images
age_range = "15-35 years old"

counter = 0
for j in range(10):
    counter += 1
    for emotion, emotion_prompt in emotion_prompts.items():
        ethnicity = random.choice(ethnicities)
        gender = random.choice(genders)

        # Include the age range condition in the prompt
        prompt = f'Medium-shot portrait of {ethnicity} {gender}, age {age_range}, {emotion_prompt}, front view, ' \
                 'looking at the camera, color photography, photorealistic, hyperrealistic, ' \
                 'incredibly detailed, crisp focus, digital art, depth of field, 50mm, 8k'

        negative_prompt = '3d, cartoon, anime, sketches, worst quality:2, low quality:2, ' \
                          'normal quality:2, lowres, monochrome, grayscale, plastic, fake, ' \
                          'disfigured, deformed, blurry, bad anatomy, blurred, watermark, ' \
                          'grainy, signature, elderly, infant, child, old, young child'

        # Generate the image
        img = pipeline(prompt, negative_prompt=negative_prompt).images[0]
        img_path = f'./content/faces/{emotion}/{str(j).zfill(5)}.png'
        img.save(img_path)

    print(f'Generated {counter} sets of images')
