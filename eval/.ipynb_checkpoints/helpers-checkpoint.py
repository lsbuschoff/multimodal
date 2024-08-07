import requests
import mimetypes
import cv2
import random
import string
from PIL import Image
from typing import Union
from torchvision import transforms

# ------------------- Utility Functions -------------------
def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type

def configure_options(options, letters=False, with_a=True):
    """Options has to be a list of strings"""
    out = ""
    random.shuffle(options)
    for num, opt in enumerate(options):
        if letters:
            out += f"\n {string.ascii_uppercase[num]}: {opt}"
        else:
            out += f"\n {num+1}: {opt}"
    
    # Add A: only for some models
    if with_a:
        out += "\n A:"
    else:
        out += "\n"

    return out


# ------------------- Image and Video Handling Functions -------------------
def extract_frames(video_path, num_frames=100, black=False):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = 1  # total_frames // num_frames
    frames = []

    # Loop through frames
    for i in range(min(num_frames, total_frames)):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()

        # Make frame all black
        if black:
            frame = frame * 0

        # Convert to RGB if necessary
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            if i == 0:
                print("Converting to RGB")

        # Squeeze frame
        frame_squeeze = frame.resize((224, 224))
        frames.append(frame_squeeze)

    video.release()
    return frames


# def get_image(url: str, black: bool) -> Union[Image.Image, list]:
#     if "://" not in url:  # Local file
#         content_type = get_content_type(url)
#     else:  # Remote URL
#         content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")
#     print(f"Input is type {content_type}")

#     if "image" in content_type:
#         if "://" not in url:  # Local file
#             return Image.open(url)
#         else:  # Remote URL
#             return Image.open(requests.get(url, stream=True, verify=False).content)  #.raw)
#     elif "video" in content_type:
#         video_path = "temp_video.mp4"
#         if "://" not in url:  # Local file
#             video_path = url
#         else:  # Remote URL
#             with open(video_path, "wb") as f:
#                 f.write(requests.get(url, stream=True, verify=False).content)
#         frames = extract_frames(video_path, black=black)
#         if "://" in url:  # Only remove the temporary video file if it was downloaded
#             os.remove(video_path)
#         return frames
#     else:
#         raise ValueError("Invalid content type. Expected image or video.")

def get_image(url: str) -> Union[Image.Image, list]:
    if not url.strip():  # Blank input, return a blank Image
        return Image.new("RGB", (224, 224))  # Assuming 224x224 is the default size for the model. Adjust if needed.
    elif "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image.")
    

# ------------------- OTTER Prompt and Response Functions -------------------
def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(image, prompt: str, model=None, image_processor=None, max_tokens=512) -> str:
    input_data = image

    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype)
        else:
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype

    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=max_tokens,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output

