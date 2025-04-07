# coding=utf-8
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

inputs = tokenizer("如何做蛋炒饭？", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))