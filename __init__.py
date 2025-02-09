"""
Hunyuan Multi-Lora Loader
This node provides two versions: one for compatibility with HunyuanVideo Model Loader and one for standard LoRA loading.
"""

from .multi_lora_loader import HunyuanMultiLoraLoaderWrapper, HunyuanMultiLoraLoader

NODE_CLASS_MAPPINGS = {
    "HunyuanMultiLoraLoaderWrapper": HunyuanMultiLoraLoaderWrapper,
    "HunyuanMultiLoraLoader": HunyuanMultiLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanMultiLoraLoaderWrapper": "Hunyuan Multi-Lora Loader (Wrapper)",
    "HunyuanMultiLoraLoader": "Hunyuan Multi-Lora Loader",
}

__version__ = "1.0.0"
