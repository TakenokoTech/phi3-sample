import numpy as np

from entity.model import Model
from utils import utils

answer = []
(past_key, past_value) = Model.reset_past()
input_ids = Model.get_input_ids()


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
    outputs = Model.ort_sess.run(None, data)
    logits = utils.apply_repetition_penalty(outputs[0][0][0], answer)
    next_id = utils.top_k_sampling(logits=logits) if count > 0 else 32001
    answer += [next_id]
    print("[decode]", Model.decode(answer))
    if next_id == 32007: break
    input_ids = np.array([next_id])
    past_key = [np.array(outputs[(i*2)+1]) for i in range(32)]
    past_value = [np.array(outputs[(i+1)*2]) for i in range(32)]
    print("")
