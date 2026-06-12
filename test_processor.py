import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
image = np.zeros((256, 256, 3), dtype=np.uint8)
inputs = processor(images=[image], size={"longest_edge": 512}, return_tensors="pt")
pixel_values = inputs.pixel_values.view(-1, 3, 512, 512)
vision_outputs = model.model.vision_model(pixel_values=pixel_values)
v_tokens = model.model.connector(vision_outputs.last_hidden_state)
print("Success. Token shape:", v_tokens.shape)