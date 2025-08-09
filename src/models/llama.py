from transformers import LlamaConfig, LlamaForCausalLM
from .registry import register_model

@register_model("llama", LlamaConfig)
class LLaMAModel(LlamaForCausalLM):
    pass