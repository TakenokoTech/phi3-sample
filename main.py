# https://colab.research.google.com/drive/1abHQiRau5aWl2Yh-sXAVz-jj9jmBRuVH#scrollTo=klP2cOv7nzv0
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mlx_lm import load, generate

device = torch.device('cpu')
model_id = "microsoft/Phi-3-mini-4k-Instruct"
# onnx_model_id = "microsoft/Phi-3-mini-4k-instruct-onnx"

model, tokenizer = load("mlx-community/Phi-3-mini-128k-instruct-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

messages = [
    {"role": "system", "content": "You are smart AI."},
    {"role": "user", "content": "Hallo."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

print(input_ids)
print(tokenizer.decode(input_ids[0]))
# print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
# input_ids.to(model.device)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

with torch.no_grad():
    outputs = model.generate(
        input_ids.to(device),
        max_new_tokens=128,
        # return_full_text=False,
        # eos_token_id=terminators,
        # do_sample=True,
        # temperature=0.1,
        # top_p=0.9,
    )
    print(outputs)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
