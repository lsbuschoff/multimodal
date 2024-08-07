This folder features the images and python scripts used to evaluate the open soruce models. Python scripts to evaluate Claude-3 Opus and GPT-4V preview are not included as they are identical but include private API keys. To run the scripts, you first need to install the respective models:
- For Fuyu: https://huggingface.co/adept/fuyu-8b
- For LLaMA-Adapter V2: https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b
- For Otter: https://github.com/Luodian/Otter/tree/main

All models were evaluated on single Nvidia A100s with a Slurm batch system (batch scripts are available upon request). 
