import os
import folder_paths
import torch
from typing import Dict, List, Optional, Tuple
from comfy.model_patcher import ModelPatcher
from comfy.utils import load_torch_file
from comfy.sd import load_lora_for_models

class HunyuanMultiLoraLoaderWrapper:
    """
    Hunyuan Multi-Lora Loader (Wrapper)
    This node outputs LoRA information in HYVIDLORA format for compatibility with HunyuanVideo Model Loader.
    It does not have a model input or output.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_01": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_02": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_03": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_04": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_04": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
            },
        }

    RETURN_TYPES = ("HYVIDLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "get_loras"
    CATEGORY = "loaders/hunyuan"
    DESCRIPTION = "Output LoRA information in HYVIDLORA format for compatibility with HunyuanVideo Model Loader."

    def get_loras(self, **kwargs):
        """Generate LoRA information in HYVIDLORA format."""
        loras_list = []

        for i in range(1, 5):
            lora_name = kwargs.get(f"lora_0{i}")
            strength = kwargs.get(f"strength_0{i}")
            blocks_type = kwargs.get(f"blocks_type_0{i}")

            if lora_name != "None" and strength != 0:
                # Prepare the LoRA dictionary for HYVIDLORA output
                lora_dict = {
                    "path": folder_paths.get_full_path("loras", lora_name),
                    "strength": strength,
                    "name": lora_name.split(".")[0],
                    "blocks_type": blocks_type,
                    "blocks": None,  # You can add block-specific logic here if needed
                }
                loras_list.append(lora_dict)

        # Ensure the output is always a list, even if empty
        return (loras_list,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return f"{kwargs.get('lora_01')}_{kwargs.get('strength_01')}_{kwargs.get('blocks_type_01')}_" \
               f"{kwargs.get('lora_02')}_{kwargs.get('strength_02')}_{kwargs.get('blocks_type_02')}_" \
               f"{kwargs.get('lora_03')}_{kwargs.get('strength_03')}_{kwargs.get('blocks_type_03')}_" \
               f"{kwargs.get('lora_04')}_{kwargs.get('strength_04')}_{kwargs.get('blocks_type_04')}"


class HunyuanMultiLoraLoader:
    """
    Hunyuan Multi-Lora Loader
    This node works like the original lora_loader.py, with a required model input and output.
    It does not output LoRA information in HYVIDLORA format.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_01": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_02": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_03": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
                "lora_04": (['None'] + folder_paths.get_filename_list("loras"),),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "blocks_type_04": (["all", "single_blocks", "double_blocks"], {"default": "all"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_multiple_loras"
    CATEGORY = "loaders/hunyuan"
    DESCRIPTION = "Load and apply multiple LoRA models with different strengths and block types. Model input is required."

    def convert_key_format(self, key: str) -> str:
        """Standardize LoRA key format by removing prefixes."""
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_type: str) -> Dict[str, torch.Tensor]:
        """Filter LoRA weights based on block type."""
        if blocks_type == "all":
            return lora
        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            if blocks_type in base_key:
                filtered_lora[key] = value
        return filtered_lora

    def load_lora(self, lora_name: str, strength: float, blocks_type: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and filter a single LoRA model."""
        if not lora_name or strength == 0:
            return {}, {}

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA file not found: {lora_path}")

        # Load the LoRA weights
        lora_weights = load_torch_file(lora_path)

        # Filter the LoRA weights based on the block type
        filtered_lora = self.filter_lora_keys(lora_weights, blocks_type)

        return lora_weights, filtered_lora

    def load_multiple_loras(self, model, **kwargs):
        """Load and apply multiple LoRA models."""
        for i in range(1, 5):
            lora_name = kwargs.get(f"lora_0{i}")
            strength = kwargs.get(f"strength_0{i}")
            blocks_type = kwargs.get(f"blocks_type_0{i}")

            if lora_name != "None" and strength != 0:
                # Load and filter the LoRA weights
                lora_weights, filtered_lora = self.load_lora(lora_name, strength, blocks_type)

                # Apply the LoRA weights to the model
                if filtered_lora:
                    model, _ = load_lora_for_models(model, None, filtered_lora, strength, 0)

        return (model,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return f"{kwargs.get('lora_01')}_{kwargs.get('strength_01')}_{kwargs.get('blocks_type_01')}_" \
               f"{kwargs.get('lora_02')}_{kwargs.get('strength_02')}_{kwargs.get('blocks_type_02')}_" \
               f"{kwargs.get('lora_03')}_{kwargs.get('strength_03')}_{kwargs.get('blocks_type_03')}_" \
               f"{kwargs.get('lora_04')}_{kwargs.get('strength_04')}_{kwargs.get('blocks_type_04')}"
