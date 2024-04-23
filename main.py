### create image generation pipeline ###

from diffusers import DiffusionPipeline
import torch
import random
import os

pipeline = DiffusionPipeline.from_pretrained("SG161222/RealVisXL_V4.0", torch_dtype=torch.float16)
pipeline.to("cuda")

### generate images ###
# Ensuring directories for all emotions exist
emotions = ['smiling', 'surprised']
for emotion in emotions:
    os.makedirs(f'./content/faces/{emotion}', exist_ok=True)

# Expanding the ethnicities to cover a more global representation
ethnicities = ['a latino', 'a white', 'a black', 'a middle eastern', 'an indian', 'an asian', 'an african',
               'an eastern european', 'a southeast asian', 'a pacific islander']
genders = ['male', 'female']

# Correcting the emotion_prompt dictionary to ensure clear representation of emotions
emotion_prompts = {
    'smiling': 'slight smile with no visible teeth, soft eyes, closed mouth, no teeth, no open mouth, mouth closed, none teeth',
    'surprised': 'surprised, mouth wide open, raised eyebrows, eyes wide open, natural expression, perfect expression, natural face'
}

# Define the age range for the generated images
age_range = "15-35 years old"

counter = 0
for j in range(5):
    counter += 1
    for emotion, emotion_prompt in emotion_prompts.items():
        ethnicity = random.choice(ethnicities)
        gender = random.choice(genders)

        # Include the age range condition in the prompt
        prompt = f'Medium-shot portrait of {ethnicity} {gender}, age {age_range}, {emotion_prompt}, front view, ' \
                 'looking at the camera, color photography, photorealistic, hyperrealistic, ' \
                 'incredibly detailed, crisp focus, digital art, depth of field, 50mm'

        negative_prompt = '3d, cartoon, anime, sketches, worst quality:2, low quality:2, ' \
                          'normal quality:2, low resolution, lowres, monochrome, grayscale, plastic, fake, ' \
                          'disfigured, mutant,  defective anatomy, deformed, blurry, bad anatomy, blurred, watermark, ' \
                          'grainy, signature, elderly, infant, child, old, young child, deformed, two mouths, mutant, robot, strange, monster'

        # Generate the image
        img = pipeline(prompt, negative_prompt=negative_prompt).images[0]
        img_path = f'./content/faces/{emotion}/{str(j).zfill(5)}.png'
        img.save(img_path)

    print(f'Generated {counter} sets of images')
