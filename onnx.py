import random

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from entity.config import Config
from utils import softmax, normalize, temperature_scaling, top_k_sampling, top_p_sampling

# token = "4k"
token = "128k"
tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-mini-{token}-instruct", trust_remote_code=True)
ort_sess = ort.InferenceSession(f"tmp/phi3-mini-{token}-instruct-cpu-int4-rtn-block-32.onnx")


def reset_past():
    past_key = [np.zeros((1, 32, 0, 96), dtype=np.float32) for _ in range(32)]
    past_value = [np.zeros((1, 32, 0, 96), dtype=np.float32) for _ in range(32)]
    return past_key, past_value


def get_input_ids(message):
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing great."
                                         # "I have a long experience as an engineer."
                                         # "My favorite languages are python and kotlin."
                                         "Please ask me anything."
                                         "How can I help you today?"},
        # {"role": "user", "content": "Who are you?"},
        # {"role": "user", "content": "I would like to hear the Japanese folk tale Momotaro."},
        # {"role": "assistant", "content": message},
        {"role": "user", "content": "one plus one equals?"},
    ]
    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, return_tensors="np")[0]
    print("input =", tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))
    return input_ids


answer = []
(past_key, past_value) = reset_past()
input_ids = get_input_ids("")
input_ids = np.append(input_ids, np.zeros((100,), dtype=np.int64))


for count in range(100):
    position_ids = np.array(list(range(len(input_ids))))
    attention_mask = np.array([1 for _ in range(len(input_ids))])
    data = {
        'input_ids': np.array([input_ids], dtype=np.int64),
        "position_ids": np.array([position_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
    }
    for i in range(32):
        data[f"past_key_values.{i}.key"] = past_key[i]
        data[f"past_key_values.{i}.value"] = past_value[i]
    outputs = ort_sess.run(None, data)
    next_id = top_p_sampling(logits=outputs[0][0][0]) if count > 0 else 32001
    answer += [next_id]
    print("[decode]", tokenizer.decode(answer, skip_special_tokens=False, clean_up_tokenization_spaces=True), end="\n")
    if next_id == 32007: break
    input_ids = np.array([next_id])
    past_key = [np.array(outputs[(i*2)+1]) for i in range(32)]
    past_value = [np.array(outputs[(i+1)*2]) for i in range(32)]
    print("")

def log():
    pass
    # print("[top_k_sampling]", tokenizer.decode(top_k_sampling(outputs[0][0][0])))
    # print("[top_p_sampling]", tokenizer.decode(top_p_sampling(outputs[0][0][0])))
    # print(input_ids.shape, attention_mask.shape, position_ids.shape)
    # fix_ids = np.append(input_ids, np.zeros((past_key[0].shape[2],), dtype=np.int64))
    # fix_ids = np.append(input_ids, np.zeros((0,), dtype=np.int64))
    # print(past_key[0].shape, past_key[0].dtype, past_value[1].shape, past_value[1].dtype)
    # next_id = top10[0]
    # print(output_ids, output_ids[np.argmax(output_num)])
    # print([o[np.argmax(o)] for o in outputs[0][0]])
    # print(outputs[0].shape)
    # print(tokenizer.decode(output_ids, skip_special_tokens=False))
    # if np.max(output_num) < 0.5:
    #     next_id = input_ids[0]
    #     input_ids = get_input_ids(tokenizer.decode(answer, skip_special_tokens=True))
    #     (past_key, past_value) = reset_past()
    # output_ids = [np.argmax(o) for o in output_norm]
    # output_num = [o[np.argmax(o)] for o in output_norm]
    # print("==>", output_ids[0], output_num[0])
    # top = sorted([(n, i) for (i, n) in enumerate(output_norm[0])], reverse=True)[:5]
    # candidate = [i for (n, i) in top if n > 0.1]
    # candidate = candidate if len(candidate) > 0 else [top[0][1]]
    # next_id = random.choice(candidate) if count > 0 else 32001
    # print("[candidate]", tokenizer.decode(candidate))
    # print(f"</s> {output_norm[0][2]:.3%}, </endoftext> {output_norm[0][32000]:.3%}, <|end|> {output_norm[0][32007]:.3%}")
    # print("[top]", top)