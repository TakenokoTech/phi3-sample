import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from transformers import AutoTokenizer

from entity.config import Config


@dataclass
class Model:
    tokenizer = AutoTokenizer.from_pretrained(
        f"microsoft/Phi-3-mini-{Config.token}-instruct",
        trust_remote_code=True
    )
    ort_sess = ort.InferenceSession(
        f"tmp/phi3-mini-{Config.token}-instruct-cpu-int4-rtn-block-32.onnx"
    )

    @staticmethod
    def reset_past():
        past_key = [np.zeros((1, 32, 0, 96), dtype=np.float32) for _ in range(32)]
        past_value = [np.zeros((1, 32, 0, 96), dtype=np.float32) for _ in range(32)]
        return past_key, past_value

    @staticmethod
    def get_input_ids_template():
        conversation = [
            {"role": "user", "content": Config.user_prompt1},
            {"role": "assistant", "content": Config.assistant_prompt1},
            {"role": "user", "content": Config.user_prompt2},
        ]
        input_ids = Model.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=False, return_tensors="np")[0]
        print("[input]", Model.decode(input_ids), "\n")
        return input_ids

    @staticmethod
    def get_input_ids():
        input_ids = Model.tokenizer.encode(
            f"<|system|>\n{''.join(Config.system_prompt)}<|end|>\n"
            f"<|user|>\n{''.join(Config.user_prompt1)}<|end|>\n"
            f"<|assistant|>\n{''.join(Config.assistant_prompt1)}<|end|>\n"
            f"<|user|>\n{''.join(Config.user_prompt2)}<|end|>\n",
            return_tensors="np"
        )[0]
        print("[input]", Model.decode(input_ids), "\n")
        return input_ids

    @staticmethod
    def decode(token):
        return Model.tokenizer.decode(
            token,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )