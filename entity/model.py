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
    def decode(token):
        return Model.tokenizer.decode(
            token,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )