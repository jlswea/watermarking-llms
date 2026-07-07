import os
import torch
from transformers import pipeline

print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])

print(torch.cuda.memory_allocated(device=0))
print(torch.cuda.memory_reserved(device=0))


pipe = pipeline(model="facebook/opt-1.3b", device_map=0, torch_dtype=torch.bfloat16)#, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
print(output)

print("After loading model")
print(torch.cuda.memory_allocated(device=0))
print(torch.cuda.memory_reserved(device=0))
