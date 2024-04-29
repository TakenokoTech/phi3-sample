import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

token = "4k"
tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-mini-{token}-instruct", trust_remote_code=True)
ort_sess = ort.InferenceSession(f"tmp/phi3-mini-{token}-instruct-cpu-int4-rtn-block-32.onnx")

conversation = [
    {"role": "system", "content": "You are smart AI."},
    {"role": "user", "content": "Hallo."},
]
input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="np")[0]
past_key = [np.zeros((1, 32, 0, 96), dtype=np.float32) for _ in range(32)]
past_value = [np.zeros((1, 32, 0, 96), dtype=np.float32) for _ in range(32)]

for count in range(100):
    # fix_ids = np.append(input_ids, np.zeros((past_key[0].shape[2],), dtype=np.int64))
    # fix_ids = np.append(input_ids, np.zeros((0,), dtype=np.int64))
    position_ids = np.array(list(range(len(input_ids))))
    attention_mask = np.array([1 for _ in range(len(input_ids))])
    # print(past_key[0].shape, past_key[0].dtype, past_value[1].shape, past_value[1].dtype)
    # print(input_ids.shape, attention_mask.shape, position_ids.shape)
    # print(tokenizer.decode(input_ids, skip_special_tokens=True))
    data = {
        'input_ids': np.array([input_ids], dtype=np.int64),
        "position_ids": np.array([position_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
    }
    for i in range(32):
        data[f"past_key_values.{i}.key"] = past_key[i]
        data[f"past_key_values.{i}.value"] = past_value[i]
    outputs = ort_sess.run(None, data)
    output_ids = [np.argmax(o) for o in outputs[0][0]]
    output_num = [o[np.argmax(o)] for o in outputs[0][0]]
    # print([o[np.argmax(o)] for o in outputs[0][0]])
    # print(outputs[0].shape)
    # print(tokenizer.decode(output_ids, skip_special_tokens=False))
    if count > 0:
        print(tokenizer.decode(output_ids, skip_special_tokens=False), end=" ")
    input_ids = [output_ids[np.argmax(output_num)]]
    past_key = [np.array(outputs[(i*2)+1]) for i in range(32)]
    past_value = [np.array(outputs[(i+1)*2]) for i in range(32)]
