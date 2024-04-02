import argparse
import json
import os
import helpers
import torch
import glob
import numpy as np
import jsonlines
from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor
from PIL import Image
from tqdm import tqdm
from natsort import natsorted

# Get access to otter folder
pretrained_path = "../fuyu-8b"
model_id = "adept/fuyu-8b"

# Init parser and set defaults
parser = argparse.ArgumentParser(description='MMLLM')
parser.add_argument('--model', type=str, default="FUYU")
parser.add_argument('--dataset', type=str, default="CUBES")
parser.add_argument('--seed', type=int, default=18)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
args = parser.parse_args()

# Set seeds and device
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
precision = {"torch_dtype": torch.float32}
print(precision)

# Create directory if it does not yet exist and save args
model_str = f"results/{args.model}_{args.dataset}_seed_{args.seed}"
os.makedirs(model_str, exist_ok=True)
with open(f'{model_str}/args', 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# Model settings
tokenizer = AutoTokenizer.from_pretrained(model_id)
image_processor = FuyuImageProcessor()
processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
model.eval()
prompts = 'Experiment start \n\n'

# Input shape: (batch_size, len_sequence_train, num_input_channels, 64, 64)
for epoch in tqdm(range(args.epochs)):

    # Disable gradient calculation for test run in order to save memory
    with (torch.no_grad()):

        # For Lerer cubes
        if args.dataset == "CUBES":

            # Set directory and get list of all images
            directory = "images/lerer"
            images = glob.glob(f'{directory}/*.png')
            images = natsorted(images)

            # Loop through experiments
            for exp in range(3):

                # Loop through images
                for i, seq in enumerate(images):

                    # Set prompts for relevant experiment
                    if exp == 0:
                        instruction = "Q: What is the background color?" # You are only allowed to answer with a number corresponding to the correct background color. No words are allowed!
                        # options = ["White", "Black", "Red", "Blue", "Yellow", "Green"]
                        # instruction += helpers.configure_options(options, with_a=False)
                    
                    elif exp == 1:
                        # instruction = "Q: How many blocks are in the image? You are only allowed to answer with the letter corresponding to the correct number of blocks. No words are allowed!"
                        # options = ["1", "2", "3", "4", "5"]
                        # instruction += helpers.configure_options(options, letters=True, with_a=False)
                        instruction = "Q: How many blocks are in the image?" # You are only allowed to answer with a number between 1 and 5 corresponding to the correct number of blocks. No words are allowed!
                        
                    elif exp == 2:
                        instruction = "Q: Will this block tower fall? Give a boolean answer."

                    # Load image
                    prompts += (f'\nEXP_{exp+1}, {seq}: ')
                    image_pil = Image.open(seq)

                    # Pass input to model
                    model_inputs = processor(text=instruction, images=image_pil, return_tensors="pt").to("cuda:0")

                    # Model output
                    generation_output = model.generate(**model_inputs, max_new_tokens=7)
                    response = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
                    prompts += (f'{response}\n')
        
        # For Balance stimuli
        elif args.dataset == "BALANCE":

            # Set directory and get list of all images
            directory = "images/balance"
            images = glob.glob(f'{directory}/*.png')
            images = natsorted(images)

            # Loop through experiments
            for exp in range(3):

                # Loop through images
                for i, seq in enumerate(images):

                    # Set prompts for relevant experiment
                    if exp == 0:
                        instruction = "Q: Consider the following scenario: a friend with a poor sense of balance has built a sculpture using a computer program. He asks you to help him decide whether it will fall, and if so, which way. Please respond with the background color!"
                    elif exp == 1:
                        instruction = "Q: Consider the following scenario: a friend with a poor sense of balance has built a sculpture using a computer program. He asks you to help him decide whether it will fall, and if so, which way. Please respond with the number of blocks in the image!"
                    elif exp == 2:
                        instruction = "Q: Consider the following scenario: a friend with a poor sense of balance has built a sculpture using a computer program. He asks you to help him decide whether it will fall, and if so, which way. Please respond by answering whether the beam would “tip left”, “stay balanced”, or “tip right”!"  
                    
                    # Load image
                    prompts += (f'\nEXP_{exp+1}, {seq}: ')
                    image_pil = Image.open(seq)

                    # Pass input to model
                    model_inputs = processor(text=instruction, images=image_pil, return_tensors="pt").to("cuda:0")

                    # Model output
                    generation_output = model.generate(**model_inputs, max_new_tokens=7)
                    response = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
                    prompts += (f'{response}\n')
                    
        # For Gerstenberg Jenga
        elif args.dataset == "JENGA":

            # Add first frame of sequence to list
            directory = "images/jenga"
            images = glob.glob(f'{directory}/*.png')
            print(f"{len(images)} images.", flush=True)
            num_blocks_gt = np.load("jenga_exp1_gt.npy", allow_pickle=True)

            # Loop through experiments
            for exp in range(3):

                # Loop through images
                for i, seq in enumerate(images):

                    # Get sequence number out of image string
                    seq_nr = int(seq.split("trial_")[1].split("_")[0])

                    # Set prompts for relevant experiment
                    if exp == 0:
                        instruction = "Q: How many blocks are there in the image? You are only allowed to answer with a single number. No words allowed!"
                    elif exp == 1:
                        num_blocks = num_blocks_gt[seq_nr-1, 1]
                        instruction = f"Q: How many of the red bricks would fall off the table if the dark grey brick wasn't there? You are only allowed to answer with a single number between 0 and {num_blocks} corresponding to how many blocks would fall. No words allowed!"
                    elif exp == 2:
                        instruction = "Q: How responsible is the dark grey brick for the red bricks staying on the table? You are only allowed to answer with a number on a scale from 0% (not at all responsible) to 100% (fully responsible). No words allowed!"

                    # Load image
                    prompts += (f'\nEXP_{exp+1}, SEQ_{seq_nr}: ')
                    url = os.path.join(seq)
                    image_pil = Image.open(url).convert("RGB")

                    # Pass input to model
                    model_inputs = processor(text=instruction, images=[image_pil], device="cuda:0")
                    for k, v in model_inputs.items():
                        model_inputs[k] = v.to("cuda:0")

                    # Model output
                    generation_output = model.generate(**model_inputs, max_new_tokens=7)
                    response = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
                    prompts += (f'{response}\n')

        # For Balance stimuli
        elif args.dataset == "MICHOTTE":

             # Set directory and get list of all images
            directory = "images/michotte_edit"
            images = glob.glob(f'{directory}/*.png')
            images = natsorted(images)

            # Loop through experiments
            for exp in range(3):

                # Loop through images
                for i, seq in enumerate(images):

                    # Set prompts for relevant experiment
                    if exp == 0:
                        instruction = "Q: The scene shows two balls labeled “A”, and “B”. On the left side there is a pink gate. The solid arrows show the trajectories of the balls. What is the background color?" 
                    elif exp == 1:
                        instruction = "Q: The scene shows two balls labeled “A”, and “B”. On the left side there is a pink gate. The solid arrows show the trajectories of the balls. Does ball B enter the gate after it has been hit by ball?"
                    elif exp == 2:
                        instruction = "Q: The scene shows two balls labeled “A”, and “B”. On the left side there is a pink gate. The solid arrows show the trajectories of the balls. Would ball B have entered the gate if it had not been hit by ball A?" 
                        
                    # Load image
                    prompts += (f'\nEXP_{exp+1}, {seq}: ')
                    image_pil = Image.open(seq).convert("RGB")

                    # Pass input to model
                    model_inputs = processor(text=instruction, images=image_pil, return_tensors="pt").to("cuda:0")

                    # Model output
                    generation_output = model.generate(**model_inputs, max_new_tokens=7)
                    response = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
                    prompts += (f'{response}\n')

        # For Balance stimuli
        elif args.dataset == "BTOM":

             # Set directory and get list of all images
            directory = "images/btom_stimuli"
            images = glob.glob(f'{directory}/*.png')
            images = natsorted(images)

            # Loop through experiments
            for exp in range(3):

                # Loop through images
                for i, seq in enumerate(images):

                    # Set prompts for relevant experiment
                    if exp == 0:
                        instruction = "Q: Consider the following scenario: a hungry student leaves his office looking for lunch from one of three food trucks: Korean, Lebanese or Mexican. The university provides only two parking spots, so at most two trucks can be on campus on any given day. The student is marked by a triangle, the three trucks are marked by letters (K, Korean; L, Lebanese; M, Mexican), parking spaces are marked by yellow regions, and buildings (which block movement and line of sight visibility) are marked by black rectangles. In the image you see the trajectory of the student as he is looking for lunch. What is the background color?"
                    elif exp == 1:
                        instruction = "Q: Consider the following scenario: a hungry student leaves his office looking for lunch from one of three food trucks: Korean, Lebanese or Mexican. The university provides only two parking spots, so at most two trucks can be on campus on any given day. The student is marked by a triangle, the three trucks are marked by letters (K, Korean; L, Lebanese; M, Mexican), parking spaces are marked by yellow regions, and buildings (which block movement and line of sight visibility) are marked by black rectangles. In the image you see the trajectory of the student as he is looking for lunch. How many food trucks are in the image?"
                    elif exp == 2:
                        instruction = "Q: Consider the following scenario: a hungry student leaves his office looking for lunch from one of three food trucks: Korean, Lebanese or Mexican. The university provides only two parking spots, so at most two trucks can be on campus on any given day. The student is marked by a triangle, the three trucks are marked by letters (K, Korean; L, Lebanese; M, Mexican), parking spaces are marked by yellow regions, and buildings (which block movement and line of sight visibility) are marked by black rectangles. In the image you see the trajectory of the student as he is looking for lunch. What is his favorite food truck, and why?"
                    
                    # Load image
                    prompts += (f'\nEXP_{exp+1}, {seq}: ')
                    image_pil = Image.open(seq)

                    # Pass input to model
                    model_inputs = processor(text=instruction, images=image_pil, return_tensors="pt").to("cuda:0")

                    # Model output
                    generation_output = model.generate(**model_inputs, max_new_tokens=7)
                    response = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
                    prompts += (f'{response}\n')
                    
        # For Jara-Ettinger Naive Utility Calculus (NUC)
        elif args.dataset == "EXP1A":

            # Add first frame of sequence to list
            directory = "images/naive_util_calc_images/Exp1a"
            images = glob.glob(f'{directory}/*.png')
            print(f"{len(images)} images.", flush=True)

            # Set basic prompt that explains context
            basic_prompt = "This task is about astronauts. The astronauts are exploring planets with alien terrains depicted with different colours and textures. Each astronaut has different skills, making each terrain more or less exhausting or easy for them to cross. All astronauts can ultimately cross all terrains, even if it's exhausting. The astronauts land far from the base and have to walk there. In each image, the black circle on the left indicates where the astronaut landed. The base is on the middle right part of the image. Sometimes care packages are dropped from above and the astronauts can pick them up. There are two kinds of care packages depicted with an orange cylinder and a white cube. Each astronaut has different preferences and likes each kind of care package in different amounts. The astronauts don't actually need the care packages. They can go straight to the base, or they can pick one up. You will see images of different astronauts with different skills and preferences travelling from their landing location to the home base. The astronauts always have a map. So they know all about the terrains and the care packages. "

            # Loop through images
            for i, seq in enumerate(images):

                # Load image
                url = os.path.join(seq)
                image_pil = Image.open(url).convert("RGB")

                # Set prompts for relevant experiment
                if i == 0 or i == 1:
                    instructions = ["1. What is the predominant color of the background of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                elif i == 2 or i == 3 or i == 4 or i == 5:
                    instructions = ["1. What is the predominant color of the background of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How much does the astronaut like the white care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!",
                                    "5. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                elif i == 6 or i == 7 or i == 8 or i == 9 or i == 10:
                    instructions = ["1. What is the predominant color of the background of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the yellow terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "5. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                elif i == 11 or i == 12 or i == 13 or i == 14 or i == 15:
                    instructions = ["1. What is the predominant color of the background of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the yellow terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "5. How much does the astronaut like the white care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!",
                                    "6. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]
                
                # Loop through questions
                for instruction in instructions:

                    # Combine both prompts
                    if instruction[0] == 1:
                        constraint = "Please answer the following question with a single color name only: "
                        comb_instruction = constraint + instruction[3:]
                    elif instruction[0] == 2:
                        constraint = "Please answer the following question with a number only: "
                        comb_instruction = constraint + instruction[3:]
                    else:
                        constraint = "Please answer the following question with a number only: "
                        comb_instruction = basic_prompt + constraint + instruction[3:]

                    # Load image
                    prompts += (f'\nSEQ_{i}, Q_{instruction[0]}: ')

                    # Pass input to model
                    model_inputs = processor(text=comb_instruction, images=[image_pil], device="cuda:0")
                    for k, v in model_inputs.items():
                        model_inputs[k] = v.to("cuda:0")

                    # Model output
                    num_tokens = 8
                    generation_output = model.generate(**model_inputs, max_new_tokens=num_tokens)
                    response = processor.batch_decode(generation_output[:, -num_tokens:], skip_special_tokens=True)
                    prompts += (f'{response}\n')

        # For Jara-Ettinger Naive Utility Calculus (NUC)
        elif args.dataset == "EXP1B":

            # Add first frame of sequence to list
            directory = "images/naive_util_calc_images/Exp1b"
            images = glob.glob(f'{directory}/*.png')
            print(f"{len(images)} images.", flush=True)

            # Set basic prompt that explains context
            basic_prompt = "This task is about astronauts. The astronauts are exploring planets with alien terrains depicted with different colours and textures. Each astronaut has different skills, making each terrain more or less exhausting or easy for them to cross. All astronauts can ultimately cross all terrains, even if it's exhausting. Sometimes, the astronauts land far from the base and have to walk there. In each image, the black circle indicates where the astronaut landed. The base is in the center of the image. Sometimes care packages are dropped from above and the astronauts can pick them up. There are two kinds of care packages depicted with an orange cylinder and a white cube. Sometimes both care packages are identical. The astronauts cannot pick both care packages. Each astronaut has different preferences and likes each kind of care package in different amounts. The astronauts don't actually need the care packages. They can go straight to the base, or they can pick one up. You will see images of different astronauts with different skills and preferences travelling from their landing location to the home base. The astronauts always have a map. So they know all about the terrains and the care packages."
            
            # Loop through images
            for i, seq in enumerate(images):

                # Load image
                url = os.path.join(seq)
                image_pil = Image.open(url).convert("RGB")
                i += 1  # add 1 to match file names

                # Set prompts for relevant experiment
                if i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7:
                    instructions = ["1. What is the predominant color of the background in the left half of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How easy is it for the astronaut to cross the yellow terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "5. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!",
                                    "6. How much does the astronaut like the white care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                elif i == 8 or i == 9 or i == 10:
                    instructions = ["1. What is the predominant color of the background in the left half of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How easy is it for the astronaut to cross the yellow terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "5. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                elif i == 11 or i == 12 or i == 13 or i == 14 or i == 15 or i == 16 or i == 17:
                    instructions = ["1. What is the predominant color of the background in the left half of the image? You are only allowed to answer with a single color name!",
                                    "2. How many orange or white containers are in the image? You are only allowed to answer with a number!",
                                    "3. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                    "4. How much does the astronaut like the orange care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!",
                                    "5. How much does the astronaut like the white care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                # Loop through questions
                for instruction in instructions:

                    # Combine both prompts
                    if instruction[0] == 1:
                        constraint = "Please answer the following question with a single color name only: "
                        comb_instruction = constraint + instruction[3:]
                    elif instruction[0] == 2:
                        constraint = "Please answer the following question with a number only: "
                        comb_instruction = constraint + instruction[3:]
                    else:
                        constraint = "Please answer the following question with a number only: "
                        comb_instruction = basic_prompt + constraint + instruction[3:]

                    # Load image
                    prompts += (f'\nSEQ_{i}, Q_{instruction[0]}: ')

                    # Pass input to model
                    model_inputs = processor(text=comb_instruction, images=[image_pil], device="cuda:0")
                    for k, v in model_inputs.items():
                        model_inputs[k] = v.to("cuda:0")

                    # Model output
                    num_tokens = 8
                    generation_output = model.generate(**model_inputs, max_new_tokens=num_tokens)
                    response = processor.batch_decode(generation_output[:, -num_tokens:], skip_special_tokens=True)
                    prompts += (f'{response}\n')

        # For Jara-Ettinger Naive Utility Calculus (NUC)
        elif args.dataset == "EXP1C":

            # Add first frame of sequence to list
            directory = "images/naive_util_calc_images/Exp1c"
            images = glob.glob(f'{directory}/*.png')
            print(f"{len(images)} images.", flush=True)

            # Set basic prompt that explains context
            basic_prompt = "This task is about astronauts. The astronauts are exploring planets with alien terrains depicted with different colours and textures. Each astronaut has different skills, making each terrain more or less exhausting or easy for them to cross. All astronauts can ultimately cross all terrains, even if it's exhausting. The astronauts land far from the base and have to walk there. In each image, the black circle on the left indicates where the astronaut landed. The base is on the right part of the image. The path astronauts take from where they land to their base is indicated by a thick black line between the black circle on the left and the astronaut on the right. Sometimes care packages depicted by a blue cube on a black background are dropped from above and the astronauts can pick them up. Each astronaut has different preferences and likes each care package in different amounts. The astronauts don't actually need the care packages. They can go straight to the base, or they can pick one up. You will see images of different astronauts with different skills and preferences travelling from their landing location to the home base. Your task is to judge how easy/exhausting it is for the astronaut in each image to cross each terrain, and how much they like each care package. The astronauts always have a map. So they know all about the terrains and the care packages."
            
            # Loop through images
            for i, seq in enumerate(images):

                # Load image
                url = os.path.join(seq)
                image_pil = Image.open(url).convert("RGB")
                i += 1  # add 1 to match file names

                # Set prompts for relevant experiment
                instructions = ["1. What is the predominant color of the background in the top half of the image? You are only allowed to answer with a single color name!",
                                "2. How many blue containers are in the image? You are only allowed to answer with a number!",
                                "3. How easy is it for the astronaut to cross the yellow terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                "4. How easy is it for the astronaut to cross the pink terrain on a scale from 0 (extremely easy) to 10 (extremely exhausting)? You are only allowed to answer with a number!",
                                "5. How much does the astronaut like the blue care package on a scale from 0 (not at all) to 10 (a lot)? You are only allowed to answer with a number!"]

                # Loop through questions
                for instruction in instructions:

                    # Combine both prompts
                    if instruction[0] == 1:
                        constraint = "Please answer the following question with a single color name only: "
                        comb_instruction = constraint + instruction[3:]
                    elif instruction[0] == 2:
                        constraint = "Please answer the following question with a number only: "
                        comb_instruction = constraint + instruction[3:]
                    else:
                        constraint = "Please answer the following question with a number only: "
                        comb_instruction = basic_prompt + constraint + instruction[3:]

                    # Load image
                    prompts += (f'\nSEQ_{i}, Q_{instruction[0]}: ')

                    # Pass input to model
                    model_inputs = processor(text=comb_instruction, images=[image_pil], device="cuda:0")
                    for k, v in model_inputs.items():
                        model_inputs[k] = v.to("cuda:0")

                    # Model output
                    num_tokens = 8
                    generation_output = model.generate(**model_inputs, max_new_tokens=num_tokens)
                    response = processor.batch_decode(generation_output[:, -num_tokens:], skip_special_tokens=True)
                    prompts += (f'{response}\n')

# Write all to output file
print(prompts)
with jsonlines.open(f'{model_str}/prompts.jsonl', 'w') as writer:
    writer.write_all(prompts)
