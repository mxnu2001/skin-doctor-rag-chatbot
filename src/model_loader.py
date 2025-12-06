from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

print("[MODEL] Loading Mistral-7B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_8bit=True
)

def generate_answer(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
