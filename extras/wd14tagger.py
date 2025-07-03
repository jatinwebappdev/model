# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags
# https://github.com/pythongosssss/ComfyUI-WD14-Tagger/blob/main/wd14tagger.py

# {
#     "wd-v1-4-moat-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2",
#     "wd-v1-4-convnextv2-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
#     "wd-v1-4-convnext-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2",
#     "wd-v1-4-convnext-tagger": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger",
#     "wd-v1-4-vit-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2"
# }


import numpy as np
import csv
import onnxruntime as ort
import torch
from PIL import Image
from onnxruntime import InferenceSession
from repositories.Fooocus.modules.config import path_clip_vision
from repositories.Fooocus.modules.model_loader import load_file_from_url
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import tempfile
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
 )
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="cuda:0", quantization_config=quantization_config
#     )
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast = True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

exclude_tags="Realistic, hyperrealism, photorealistic, animerealistic, realism "

def default_interrogator(image_rgb, exclude_tags=exclude_tags):
   
    pil_image = Image.fromarray(image_rgb) 
    temp_path = tempfile.mktemp(suffix=".jpg",prefix='temp')
    pil_image.save(temp_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{temp_path}",
                },
                {"type": "text", "text": """I will be providing you with an image. You need to give anime-style tags for that image.
                                            You should only provide tags, no other text should be included.
                                            ###Important rules###

                                            1. Do not repeat tags or give different tags for the same feature multiple times
                                            2.Do not use words like 'realistic' or 'photorealistic'
                                            3.Be extremely specific about all visual features (hair type, clothing details, accessories, etc.)
                                            4.Include tags for ALL visual elements to ensure accurate anime generation
                                            5.if possible mention the location of somes features.
                                            6.Specify exact hair styles (straight, wavy, etc.) if slightly curly hair give it as wavy and colors accurately
                                            7.Include detailed clothing descriptions (colors, styles, patterns)
                                            8.Describe facial expressions and poses precisely
                                            9.Note any accessories or distinguishing features
                                            10.Tag the background and setting elements
                                            11.Don't give Negative Promts such as ' no hair , no ring etc..'
                                            12.Also describe the image at the end so that the model is aware about the positions of face , hand and features.

                                            For example, good tags might include: 1boy, male focus, solo, jewelry, facial hair, black hair, straight hair, pants, shirt, sitting, black shirt, denim, blurry background, necklace, jacket, looking at viewer, earrings, short hair, outdoors, stubble, jeans, open clothes, bracelet, arm hair, wristwatch, grey jacket, asian, beard, watch, open jacket, head tilt, closed mouth, brown eyes, chest hair, depth of field, day, safe"""},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    remove = [s.strip().lower() for s in exclude_tags.lower().split(",")]
# First filter: keep tags that aren't in the remove list
    filtered_tags = []
    for tag in output_text[0].split(','):
        tag = tag.strip()
        if tag and all(exclude_word not in tag.lower() for exclude_word in remove):
            filtered_tags.append(tag)
            
    unique_tags = list(dict.fromkeys(filtered_tags))  # Remove duplicates while preserving order
    res = "Anime 4k, " + ", ".join(tag.replace("(", "\\(").replace(")", "\\)").replace('_', ' ') for tag in unique_tags)

    return res

