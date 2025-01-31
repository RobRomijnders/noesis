# pylint: disable-all
import copy
import logging
from typing import Optional, Tuple, Union
import warnings
import torch
import transformers
from torch import nn
from torchtune.modules import peft
from transformers import modeling_outputs
from transformers.models.t5 import modeling_t5

logger = logging.getLogger('transformers')


class OurT5Config(transformers.T5Config):

    def __init__(self, num_experts=-1, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts


# Cannot inherit from modeling_t5.T5DenseActDense as that tries some weird initialization
class OurLoRAT5DenseActDense(nn.Module):
    """Mimics the T5DenseActDense layer with LoRA."""

    def __init__(self, config: OurT5Config, num_block: int):
        super().__init__()

        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)  # Commone expert, initialized from upstream
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)  # Commone expert, initialized from upstream

        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = modeling_t5.ACT2FN[config.dense_act_fn]

        if config.freeze_backbone:
            self.wi.weight.requires_grad = False  # Common expert is never trainable
            self.wo.weight.requires_grad = False  # Common expert is never trainable

        self.rank_common: int = config.rank_common
        self.rank_domain: int = config.rank_domain
        self.lora_alpha: float = config.lora_alpha

        self.common_wi_a, self.common_wi_b, self.common_wo_a, self.common_wo_b = None, None, None, None
        if config.rank_common > 0:
            # LoRA on the common expert will receive gradients with DP noise
            self.common_wi_a = nn.Linear(config.d_model, config.rank_common, bias=False)
            self.common_wi_b = nn.Linear(config.rank_common, config.d_ff, bias=False)
            self.common_wo_a = nn.Linear(config.d_ff, config.rank_common, bias=False)
            self.common_wo_b = nn.Linear(config.rank_common, config.d_model, bias=False)
            if config.freeze_common:
                print("INFO: Freezing common LoRA")
                self.common_wi_a.weight.requires_grad = False
                self.common_wi_b.weight.requires_grad = False
                self.common_wo_a.weight.requires_grad = False
                self.common_wo_b.weight.requires_grad = False

        if (num_block >= config.expert_layer_start) and config.num_experts > 0:
            self.num_experts = config.num_experts
        else:
            self.num_experts = 0

        self.experts_wi_a, self.experts_wi_b, self.experts_wo_a, self.experts_wo_b = None, None, None, None
        if self.num_experts > 0:
            assert self.rank_domain > 0, "When experts are set, set rank"
            self.experts_wi_a = nn.ModuleList()
            self.experts_wi_b = nn.ModuleList()
            self.experts_wo_a = nn.ModuleList()
            self.experts_wo_b = nn.ModuleList()
            # Add one layer for each domain
            for _ in range(self.num_experts):
                self.experts_wi_a.append(nn.Linear(config.d_model, config.rank_domain, bias=False))
                self.experts_wi_b.append(nn.Linear(config.rank_domain, config.d_ff, bias=False))
                self.experts_wo_a.append(nn.Linear(config.d_ff, config.rank_domain, bias=False))
                self.experts_wo_b.append(nn.Linear(config.rank_domain, config.d_model, bias=False))

                self.experts_wi_a[-1].weight.is_expert = True  # Tag for escape in DP optimizer
                self.experts_wi_b[-1].weight.is_expert = True  # Tag for escape in DP optimizer
                self.experts_wo_a[-1].weight.is_expert = True  # Tag for escape in DP optimizer
                self.experts_wo_b[-1].weight.is_expert = True  # Tag for escape in DP optimizer

                self.experts_wi_a[-1].is_expert = True  # Tag for escape in DP optimizer
                self.experts_wi_b[-1].is_expert = True  # Tag for escape in DP optimizer
                self.experts_wo_a[-1].is_expert = True  # Tag for escape in DP optimizer
                self.experts_wo_b[-1].is_expert = True  # Tag for escape in DP optimizer

                if config.freeze_domain:
                    print("INFO: Freezing domain experts")
                    self.experts_wi_a[-1].weight.requires_grad = False
                    self.experts_wi_b[-1].weight.requires_grad = False
                    self.experts_wo_a[-1].weight.requires_grad = False
                    self.experts_wo_b[-1].weight.requires_grad = False

    def forward(self, hidden_states, expert_ids: Optional[torch.LongTensor]):
        hidden_i = self.wi(hidden_states)
        if self.rank_common > 0:
            hidden_i += self.lora_alpha * self.common_wi_b(self.common_wi_a(hidden_states))

        if self.num_experts > 0:
            for num_expert in range(self.num_experts):
                # Counting for num_expert is 0-based
                idx = torch.where(expert_ids.squeeze() == num_expert)[0]  # pytype: disable=attribute-error
                if len(idx) == 0:
                    continue
                correction = self.lora_alpha * self.experts_wi_b[num_expert](
                    self.experts_wi_a[num_expert](hidden_states[idx]))
                hidden_i.index_add_(0, idx, correction)

        hidden_i = self.act(hidden_i)
        hidden_i = self.dropout(hidden_i)
        hidden_o = self.wo(hidden_i)
        if self.rank_common > 0:
            hidden_o += self.lora_alpha * self.common_wo_b(self.common_wo_a(hidden_i))

        if self.num_experts > 0:
            for num_expert in range(self.num_experts):
                # Counting for num_expert is 0-based
                idx = torch.where(expert_ids.squeeze() == num_expert)[0]  # pytype: disable=attribute-error
                if len(idx) == 0:
                    continue
                correction = self.lora_alpha * self.experts_wo_b[num_expert](
                    self.experts_wo_a[num_expert](hidden_i[idx]))
                hidden_o.index_add_(0, idx, correction)

        return hidden_o

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        # pylint: disable=protected-access
        if self.rank_common > 0:
            peft.lora._lora_a_init_params(self.common_wi_a)
            peft.lora._lora_b_init_params(self.common_wi_b)
            peft.lora._lora_a_init_params(self.common_wo_a)
            peft.lora._lora_b_init_params(self.common_wo_b)
        if self.num_experts > 0:
            for i in range(self.num_experts):
                peft.lora._lora_a_init_params(self.experts_wi_a[i])
                peft.lora._lora_b_init_params(self.experts_wi_b[i])
                peft.lora._lora_a_init_params(self.experts_wo_a[i])
                peft.lora._lora_b_init_params(self.experts_wo_b[i])


class OurLoRAFF(modeling_t5.T5LayerFF):
    """Subclass of T5LayerFF with LoRA."""

    def __init__(self, config: OurT5Config, num_block: int = -1):
        super().__init__(config)
        assert not config.is_gated_act, "Gated experts not implemented yet"
        self.DenseReluDense = OurLoRAT5DenseActDense(config, num_block)
        if config.freeze_backbone:
            self.layer_norm.weight.requires_grad = False  # Common expert is never trainable

    def forward(self, hidden_states, expert_ids: Optional[torch.LongTensor]):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, expert_ids)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class OurT5Block(nn.Module):
    def __init__(self, config: transformers.T5Config, has_relative_attention_bias=False,
                 num_block: int = -1):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(modeling_t5.T5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias))

        if config.freeze_backbone:
            print("INFO: Freezing backbone")
            for param in self.layer[0].parameters():
                # Layer 0 is the self-attention layer, which gets frozen when freeze_backbone
                param.requires_grad = False

        self.layer.append(OurLoRAFF(config, num_block=num_block))

        # Learnable prefix tokens
        self.num_prefix_tokens = config.num_prefix_tokens
        if self.num_prefix_tokens > 0:
            self.fixed_input = 0.02 * torch.eye(n=self.num_prefix_tokens)
            self.prefix_param_embed = nn.Linear(self.num_prefix_tokens, config.d_model, bias=False)

            if config.freeze_common:
                print("INFO: Freezing prefix tokens")
                self.prefix_param_embed.weight.requires_grad = False

    def forward(
        self,
        expert_ids: Optional[torch.LongTensor],
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        del encoder_hidden_states, encoder_attention_mask
        self_attn_past_key_value, cross_attn_past_key_value = None, None

        # Append learnable token for prefix tuning
        fixed_input_exp = None
        batch_size = hidden_states.shape[0]
        if self.num_prefix_tokens > 0:
            fixed_input_exp = self.fixed_input.to(hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)

            prefix_embeds = self.prefix_param_embed(fixed_input_exp)
            hidden_states = torch.cat([prefix_embeds, hidden_states], dim=1)

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,  # None
            position_bias=None,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.num_prefix_tokens > 0:
            hidden_states = hidden_states[:, self.num_prefix_tokens:, :]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        # hidden_states of size [batch_size, seq_len, d_model]
        # d_model = 768, which can be found in checkpoint/config.json
        # layer[1] is the OurLoRAFF
        hidden_states = self.layer[1](hidden_states, expert_ids)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        del fixed_input_exp  # Free up memory
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


# class OurT5Stack(modeling_t5.T5PreTrainedModel):
class OurT5Stack(modeling_t5.T5Stack):
    def __init__(self, config: transformers.T5Config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [OurT5Block(config, has_relative_attention_bias=False, num_block=i) for i in range(config.num_layers)]
        )
        self.final_layer_norm = modeling_t5.T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        if config.freeze_backbone:  # Final layer norm is part of the backbone
            self.final_layer_norm.weight.requires_grad = False
        self.dropout = nn.Dropout(config.dropout_rate)

        self.num_prompt_tokens = config.num_prompt_tokens
        self.num_prefix_tokens = config.num_prefix_tokens
        if self.num_prompt_tokens > 0:
            self.fixed_input = 0.02 * torch.eye(n=self.num_prompt_tokens)
            self.prompt_param_embed = nn.Linear(self.num_prompt_tokens, config.d_model, bias=False)

            if config.freeze_common:
                print("INFO: Freezing prompt tokens")
                self.prompt_param_embed.weight.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
        self,
        expert_ids: Optional[torch.LongTensor],
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # Append learnable token for prompt tuning
        fixed_input_exp = None
        if self.num_prompt_tokens > 0:
            fixed_input_exp = self.fixed_input.to(inputs_embeds.device).unsqueeze(0).expand(batch_size, -1, -1)

            prefix_embeds = self.prompt_param_embed(fixed_input_exp)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        if (self.num_prompt_tokens > 0) or (self.num_prefix_tokens > 0):
            extra_tokens = max((0, self.num_prompt_tokens)) + max((0, self.num_prefix_tokens))
            attention_mask = torch.cat([
                torch.ones(attention_mask.shape[0], extra_tokens, device=attention_mask.device),
                attention_mask], dim=1)

            seq_length += extra_tokens
            input_shape = (batch_size, seq_length)

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    expert_ids,
                    hidden_states,
                    extended_attention_mask,
                    None,  # Position bias
                    None,  # encoder_hidden_states
                    None,  # encoder_extended_attention_mask
                    None,  # encoder_decoder_position_bias
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    None,  # output_attentions
                )
            else:
                layer_outputs = layer_module(
                    expert_ids,
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    layer_head_mask=None,
                    cross_attn_layer_head_mask=None,
                    past_key_value=None,
                    use_cache=use_cache,
                    output_attentions=None,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.num_prompt_tokens > 0:
            hidden_states = hidden_states[:, self.num_prompt_tokens:, :]
            attention_mask = attention_mask[:, self.num_prompt_tokens:]

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                ]
                if v is not None
            )
        del fixed_input_exp
        return modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            attentions=None,
            cross_attentions=None,
        )


# class T5ForConditionalGeneration(T5PreTrainedModel):
class OurT5DecoderOnly(transformers.T5ForConditionalGeneration):

    def __init__(self, config: modeling_t5.T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = None  # Decoder only model

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # Overwrite the original decoder with our decoder
        self.decoder = OurT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        expert_ids: Optional[torch.LongTensor],
        input_ids: Optional[torch.LongTensor] = None,  # Unused argument
        attention_mask: Optional[torch.FloatTensor] = None,  # Unused argument
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # Unused argument
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], modeling_outputs.Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        assert input_ids is None, "input_ids is not used in T5DecoderOnly"
        assert output_attentions is None, "output_attentions is not used in T5DecoderOnly"
        del encoder_outputs, input_ids  # Unused argument
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(modeling_t5.__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            assert False

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            # local change, add expert_ids
            expert_ids=expert_ids,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            self.lm_head = self.lm_head.to(self.decoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        return lm_logits


    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # Tie weights of decoder only
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
