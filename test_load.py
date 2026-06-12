import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
print("Testing SmolVLM-256M-Instruct load...")
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16)
print("Successfully loaded.")
