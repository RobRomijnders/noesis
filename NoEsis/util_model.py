"""Utility functions for loading the MoE model."""
import copy
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import safetensors
import torch
import transformers

from NoEsis import logger
from NoEsis import moe as moe_noesis


expert_param_source = [
  'decoder.block.0.layer.2.DenseReluDense.wi.weight',
  'decoder.block.0.layer.2.DenseReluDense.wo.weight',
  'decoder.block.0.layer.2.layer_norm.weight',
  'decoder.block.1.layer.2.DenseReluDense.wi.weight',
  'decoder.block.1.layer.2.DenseReluDense.wo.weight',
  'decoder.block.1.layer.2.layer_norm.weight',
  'decoder.block.2.layer.2.DenseReluDense.wi.weight',
  'decoder.block.2.layer.2.DenseReluDense.wo.weight',
  'decoder.block.2.layer.2.layer_norm.weight',
  'decoder.block.3.layer.2.DenseReluDense.wi.weight',
  'decoder.block.3.layer.2.DenseReluDense.wo.weight',
  'decoder.block.3.layer.2.layer_norm.weight',
  'decoder.block.4.layer.2.DenseReluDense.wi.weight',
  'decoder.block.4.layer.2.DenseReluDense.wo.weight',
  'decoder.block.4.layer.2.layer_norm.weight',
  'decoder.block.5.layer.2.DenseReluDense.wi.weight',
  'decoder.block.5.layer.2.DenseReluDense.wo.weight',
  'decoder.block.5.layer.2.layer_norm.weight',
  'decoder.block.6.layer.2.DenseReluDense.wi.weight',
  'decoder.block.6.layer.2.DenseReluDense.wo.weight',
  'decoder.block.6.layer.2.layer_norm.weight',
  'decoder.block.7.layer.2.DenseReluDense.wi.weight',
  'decoder.block.7.layer.2.DenseReluDense.wo.weight',
  'decoder.block.7.layer.2.layer_norm.weight',
  'decoder.block.8.layer.2.DenseReluDense.wi.weight',
  'decoder.block.8.layer.2.DenseReluDense.wo.weight',
  'decoder.block.8.layer.2.layer_norm.weight',
  'decoder.block.9.layer.2.DenseReluDense.wi.weight',
  'decoder.block.9.layer.2.DenseReluDense.wo.weight',
  'decoder.block.9.layer.2.layer_norm.weight',
  'decoder.block.10.layer.2.DenseReluDense.wi.weight',
  'decoder.block.10.layer.2.DenseReluDense.wo.weight',
  'decoder.block.10.layer.2.layer_norm.weight',
  'decoder.block.11.layer.2.DenseReluDense.wi.weight',
  'decoder.block.11.layer.2.DenseReluDense.wo.weight',
  'decoder.block.11.layer.2.layer_norm.weight',
  ]


def extract_param_copies() -> List[Tuple[str, str]]:
    """Extracts a list pf parameters that need to be copied to the MoE model."""
    logger.info(f"Number of parameters to be copied out {len(expert_param_source)}")

    list_param_mapping = []

    for num_param, param_source in enumerate(expert_param_source):
        list_param_mapping.append(
            (copy.deepcopy(param_source).replace('layer.2.', 'layer.1.'), param_source))

        if num_param < 3:  # Print examples from the first three hits for information
            logger.info(f"Example: {list_param_mapping[-1]}")
    return list_param_mapping


def read_config(ckpt_dir: str) -> Dict[str, Any]:
    """Read the config file from the checkpoint directory."""
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir {ckpt_dir} not found")

    with open(os.path.join(ckpt_dir, 'config.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_noesis_model(ckpt_dir: str) -> moe_noesis.OurT5DecoderOnly:
    """Load a DecoderOnly model that was pretrained by NoEsis."""
    try:
        config_raw = read_config(ckpt_dir)
    except FileNotFoundError:
        # Try one parent directory up
        logger.info(f"Config in checkpoint {ckpt_dir} not found. Trying parent directory")
        config_raw = read_config(os.path.dirname(ckpt_dir.rstrip('/')))

    config_t5 = transformers.T5Config.from_dict(config_raw)
    config_t5.num_experts = config_raw['num_experts']
    config_t5.num_prompt_tokens = config_raw.get('num_prompt_tokens', -1)
    config_t5.num_prefix_tokens = config_raw.get('num_prefix_tokens', -1)
    config_t5.expert_layer_start = config_raw.get('expert_layer_start', 0)
    config_t5.freeze_backbone = config_raw.get('freeze_backbone', False)
    config_t5.freeze_common = config_raw.get('freeze_common', False)
    config_t5.freeze_domain = config_raw.get('freeze_domain', False)
    config_t5.rank_common = config_raw.get('rank_common', -1)
    config_t5.rank_domain = config_raw.get('rank_domain', -1)
    model = moe_noesis.OurT5DecoderOnly(config_t5)

    # Load weights from checkpoint
    try:
        state_dict = safetensors.torch.load_file(os.path.join(ckpt_dir, 'model.safetensors'))

        state_dict['decoder.embed_tokens.weight'] = state_dict['shared.weight']
        state_dict['lm_head.weight'] = state_dict['shared.weight']
    except FileNotFoundError:
        print(f"Model in {ckpt_dir}/model.safetensors not found; trying pytorch_model.bin.")
        state_dict = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'))

        for name in copy.deepcopy(list(state_dict.keys())):
            state_dict[name.replace('_module.', '')] = state_dict.pop(name)

        # TODO: Remove this once the model is saved correctly
        np.testing.assert_array_almost_equal(
            state_dict['shared.weight'].cpu().detach().numpy(),
            state_dict['lm_head.weight'].cpu().detach().numpy())
        np.testing.assert_array_almost_equal(
            state_dict['shared.weight'].cpu().detach().numpy(),
            state_dict['decoder.embed_tokens.weight'].cpu().detach().numpy())

    print(f"Number of parameters to be copied over {len(list(state_dict.keys()))}")

    model.load_state_dict(state_dict=state_dict)
    return model


def load_noesis_for_finetuning(args) -> moe_noesis.OurT5DecoderOnly:
    """Load the NoEsis model for fine-tuning.

    This assumes a model that was already trained with the NoEsis library.
    The shared parameters have been trained with DP, and now expert parameters will be trained.
    """
    config_raw = read_config(args.ckpt_dir)

    config_t5, unused_kwargs = transformers.T5Config.from_dict(
        config_raw, return_unused_kwargs=True)
    print(f"Unused kwargs: {unused_kwargs}")

    config_t5.freeze_backbone = config_raw['freeze_backbone']
    config_t5.rank_common = config_raw['rank_common']
    config_t5.num_prompt_tokens = config_raw['num_prompt_tokens']
    config_t5.num_prefix_tokens = config_raw['num_prefix_tokens']
    config_t5.expert_layer_start = config_raw['expert_layer_start']
    config_t5.lora_alpha = config_raw['lora_alpha']

    print(f"Load for fine-tuning with: num prompt tokens {config_t5.num_prompt_tokens}, "
          f"rank common {config_t5.rank_common}")

    assert args.freeze_backbone
    assert args.expert_layer_start < 0, (
        f"Expert layer start will be taken from pre-training, not {args.expert_layer_start}")
    assert args.finetune_second > 0
    config_t5.num_experts = args.num_experts
    config_t5.rank_domain = args.rank_domain
    config_t5.freeze_common = args.freeze_common
    config_t5.freeze_domain = args.freeze_domain
    model = moe_noesis.OurT5DecoderOnly(config_t5)

    # Load state dict from checkpoint
    state_dict = safetensors.torch.load_file(os.path.join(args.ckpt_dir, 'model.safetensors'))
    state_dict['decoder.embed_tokens.weight'] = state_dict['shared.weight']
    state_dict['lm_head.weight'] = state_dict['shared.weight']

    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)

    assert len(unexpected_keys) == 0, f"Unexpected keys {unexpected_keys}"  # pylint: disable=len-as-condition
    for key in missing_keys:
        # .experts_wi, .experts_wo, .common_wi, .common_wo for LoRA are initialized randomly
        if args.finetune_second > 0 and ('.experts_w' in key):
            continue
        if config_t5.rank_common > 0 and ('.common_w' in key):
            continue
        raise ValueError(f"Missing key {key}")
    return model


def load_upstream_model(args) -> moe_noesis.OurT5DecoderOnly:
    """Loads the upstream model.

    Loads the upstream model either from CodeT5+ or from a checkpoint in args.ckpt_dir.
    """
    num_experts = args.num_experts
    expert_layer_start = args.expert_layer_start

    if args.finetune_second > 0:
        print("Finetuning second", flush=True)
        return load_noesis_for_finetuning(args)

    if not os.path.exists(args.ckpt_dir):
        logger.error(
            f"Checkpoint {args.ckpt_dir} not found. A checkpoint to copy parameters for all"
            "experts must be provided. please contact rromijnders@brave.com for "
            "the base checkpoint")
        raise FileNotFoundError(f"Checkpoint {args.ckpt_dir} not found")

    with open(os.path.join(args.ckpt_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config_raw = json.load(f)

    config_t5 = transformers.T5Config.from_dict(config_raw)
    config_t5.num_experts = num_experts
    config_t5.num_prompt_tokens = args.num_prompt_tokens
    config_t5.num_prefix_tokens = args.num_prefix_tokens
    config_t5.expert_layer_start = expert_layer_start
    config_t5.rank_common = args.rank_common
    config_t5.rank_domain = args.rank_domain
    config_t5.lora_alpha = args.lora_alpha

    assert isinstance(args.freeze_backbone, bool)
    print(f"Freeze backbone: {args.freeze_backbone}")
    config_t5.freeze_backbone = args.freeze_backbone
    config_t5.freeze_common = args.freeze_common
    config_t5.freeze_domain = args.freeze_domain
    model = moe_noesis.OurT5DecoderOnly(config_t5)

    # Load weights from checkpoint
    state_dict = torch.load(os.path.join(args.ckpt_dir, 'pytorch_model.bin'), map_location='cpu')

    # Delete parameters related to cross attention
    for param_name in copy.deepcopy(list(state_dict.keys())):  # pylint: disable=consider-iterating-dictionary
        if ('EncDecAttention' in param_name) or ('layer.1.layer_norm.weight' in param_name):
            del state_dict[param_name]
    del state_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]

    state_dict['decoder.embed_tokens.weight'] = state_dict['shared.weight']
    state_dict['lm_head.weight'] = state_dict['shared.weight']

    list_param_mapping = extract_param_copies()
    logger.info(f"Number of parameters to be copied over {len(list(list_param_mapping))}")

    for param_target, param_source in list_param_mapping:
        # Map FFN parameters from layer.2 to layer.1 because cross-attention was removed
        state_dict[param_target] = state_dict.pop(param_source)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)

    assert len(unexpected_keys) == 0, f"Unexpected keys {unexpected_keys}"  # pylint: disable=len-as-condition
    for key in missing_keys:
        # .experts_wi, .experts_wo, .common_wi, .common_wo for LoRA are initialized randomly
        if not ('.experts_w' in key or '.common_w' in key or
                'prefix_param' in key or 'prompt_param' in key):
            raise ValueError(f"Missing key {key}")
    return model
