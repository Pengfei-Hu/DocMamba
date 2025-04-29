import torch
import math
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging
from transformers.models.layoutlmv3 import LayoutLMv3Model
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mamba import MambaModel, MambaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from .configuration_docmamba import DocMambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer, MambaBlock, MambaCache, MambaOutput

from layoutlmft.modules.bimamba import mamba_inner_fn_no_out_proj

logger = logging.get_logger(__name__)


class DocMambaModel(LayoutLMv3Model):
    """
    The bare DocMamba Model outputting raw hidden-states without any specific head on top.
    """
    config_class = DocMambaConfig

    def __init__(self, config, cfg):
        super().__init__(config)
        self.config = config # hugging face's config
        self.cfg = cfg # libs/config/default.py
        self.mamba_config = None

        self.embeddings = DocMambaTextLayoutEmbeddings(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.cal_mlm_loss = nn.CrossEntropyLoss(reduction='none')
        if not self.config.image_use_position_embeddings:
            self.pos_embed = None
            self.pos_drop = None
            self.norm = None
        if self.config.layout_point_num == 8:
            x1, y1, x3, y3 = self.visual_bbox[:, 0], self.visual_bbox[:, 1], self.visual_bbox[:, 2], self.visual_bbox[:, 3]
            x1y1x2y2x3y3x4y4 = torch.stack([x1, y1, x3, y1, x3, y3, x1, y3], dim=1)
            self.visual_bbox = x1y1x2y2x3y3x4y4
        if self.config.encoder_type == 'mamba': # default to 'transformer' (original encoder of layoutlmv3)
            mamba_config = MambaConfig.from_pretrained(config.mamba_pretrained_path)
            self.encoder = MambaModel(mamba_config)
            self.encoder.embeddings = None
            self.mamba_config = mamba_config
        elif self.config.encoder_type == 'bimamba':
            mamba_config = MambaConfig.from_pretrained(config.mamba_pretrained_path)
            self.encoder = BiMambaModel(mamba_config)
            self.encoder.embeddings = None
            self.mamba_config = mamba_config
        
        if not self.cfg.use_image_token:
            self.forward_image = None
        self.init_weights()


    def forward(
        self,
        input_ids,
        bbox,
        attention_mask,
        pixel_values,
        unmask_input_ids,
        mlm_mask,
        data_idx,
        return_dict=True
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # text embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
        )

        if self.cfg.use_image_token:
            # image embeddings
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones((batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device)
            # cat text and image
            embedding_output = torch.cat([visual_embeddings, embedding_output], dim=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
            attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)


        # mamba encoder
        if self.config.encoder_type == 'mamba':
            encoder_outputs = self.encoder(inputs_embeds=embedding_output, return_dict=return_dict)
        else:
            encoder_outputs = self.encoder(inputs_embeds=embedding_output, return_dict=return_dict, attention_mask=attention_mask)
        
        # mlm loss
        if self.training:
            hidden_states = encoder_outputs.last_hidden_state
            if self.cfg.use_image_token:
                text_hidden_states = hidden_states[:, visual_embeddings.shape[1]:visual_embeddings.shape[1] + unmask_input_ids.shape[-1]]
            else:
                text_hidden_states = hidden_states[:, :unmask_input_ids.shape[-1]]
            mlm_logits = self.mlm_head(text_hidden_states)
            mlm_logits = mlm_logits.float()
            mlm_logits = mlm_logits.permute(0, 2, 1) # (b, l, vocab_size) -> (b, vocab_size, l)
            ce_loss = self.cal_mlm_loss(mlm_logits, unmask_input_ids)
            mlm_loss = (ce_loss * mlm_mask).sum() / (mlm_mask.sum() + 1e-5)
        else:
            mlm_loss = 0.0

        loss = mlm_loss

        if self.training and torch.isnan(loss).any():
            logger.warning("loss nan. Set loss to 0.")
            loss = torch.nan_to_num(loss)

        return TokenClassifierOutput(
            loss=loss,
        )

class DocMambaModelForTokenClassification(DocMambaModel):
    def __init__(self, config, cfg):
        super().__init__(config, cfg)
        self.num_labels = cfg.num_labels

        # self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_dropout = nn.Dropout(0.5)
        self.classifier_linear = nn.Linear(config.hidden_size, self.num_labels)


        self.cal_classifier_loss = nn.CrossEntropyLoss()

        self.mlm_head = None
        self.cal_mlm_loss = None


    def forward(
        self,
        input_ids,
        bbox,
        attention_mask,
        pixel_values,
        labels,
        return_dict=True
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # text embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
        )

        if self.cfg.use_image_token:
            # image embeddings
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones((batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device)
            # cat text and image
            embedding_output = torch.cat([visual_embeddings, embedding_output], dim=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
            attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)


        # mamba encoder
        encoder_outputs = self.encoder(inputs_embeds=embedding_output, return_dict=return_dict, attention_mask=attention_mask)
        
        # classification loss
        hidden_states = encoder_outputs.last_hidden_state
        if self.cfg.use_image_token:
            text_hidden_states = hidden_states[:, visual_embeddings.shape[1]:visual_embeddings.shape[1] + input_ids.shape[-1]]
        else:
            text_hidden_states = hidden_states[:, :input_ids.shape[-1]]
        
        if hasattr(self, 'classifier_dropout'):
            text_hidden_states = self.classifier_dropout(text_hidden_states)
        classifier_logits = self.classifier_linear(text_hidden_states)
        classifier_logits = classifier_logits.float()
        classifier_logits = classifier_logits.permute(0, 2, 1) # (b, l, c) -> (b, c, l)
        classification_loss = self.cal_classifier_loss(classifier_logits, labels)

        return TokenClassifierOutput(
            loss=classification_loss,
            logits=classifier_logits.permute(0, 2, 1) # (b, c, l) -> (b, l, c)
        )


class DocMambaTextLayoutEmbeddings(PreTrainedModel):
    """
    DocMamba text embedding and layout embedding.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.text_use_position_embeddings = config.text_use_position_embeddings
        self.layout_point_num = config.layout_point_num
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id
        self.xy_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.x1_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.y1_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.x2_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.y2_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.x3_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.y3_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.x4_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        self.y4_pos_embeddings = nn.Parameter(torch.ones(1, 1, config.coordinate_size))
        if self.text_use_position_embeddings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None and module.weight.shape[0] == self.config.vocab_size:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)          
    
    def forward(self, input_ids, bbox, token_type_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.text_use_position_embeddings:
            position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)
        embeddings = embeddings + spatial_position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def calculate_spatial_position_embeddings(self, bbox):
        x1 = self.xy_position_embeddings(bbox[:, :, 0]) + self.x1_pos_embeddings
        y1 = self.xy_position_embeddings(bbox[:, :, 1]) + self.y1_pos_embeddings
        x2 = self.xy_position_embeddings(bbox[:, :, 2]) + self.x2_pos_embeddings
        y2 = self.xy_position_embeddings(bbox[:, :, 3]) + self.y2_pos_embeddings
        x3 = self.xy_position_embeddings(bbox[:, :, 4]) + self.x3_pos_embeddings
        y3 = self.xy_position_embeddings(bbox[:, :, 5]) + self.y3_pos_embeddings
        x4 = self.xy_position_embeddings(bbox[:, :, 6]) + self.x4_pos_embeddings
        y4 = self.xy_position_embeddings(bbox[:, :, 7]) + self.y4_pos_embeddings
        spatial_position_embeddings = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)

        return spatial_position_embeddings
    
    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx


class BiMambaModel(MambaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([BiMambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, BiMambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True
            module.A_log_back._no_weight_decay = True
            module.D_back._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
                nn.init.constant_(module.dt_proj_back.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)
                nn.init.uniform_(module.dt_proj_back.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
                module.dt_proj_back.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True
            module.dt_proj_back.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/竏哢 where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        cache_params=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        attention_mask=None, # copy from MambaModel forward. Add `attention_mask`
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params, attention_mask=attention_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )
    
    

class BiMambaBlock(MambaBlock):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mixer = BiMambaMixer(config, layer_idx=layer_idx)
    
    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        return hidden_states


class BiMambaMixer(MambaMixer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # backward path
        A_back = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A_back = A_back.expand(self.intermediate_size, -1).contiguous()
        self.A_log_back = nn.Parameter(torch.log(A_back))

        self.conv1d_back = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.x_proj_back = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)

        self.dt_proj_back = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        self.D_back = nn.Parameter(torch.ones(self.intermediate_size))
        

    def forward(self, hidden_states, cache_params, attention_mask):
        if "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
        raise ValueError("Tensors must be on the GPU for BiMamba.")


    def cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        A_back = -torch.exp(self.A_log_back.float())
        out = mamba_inner_fn_no_out_proj(
            projected_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        projected_states_back = projected_states.flip([-1])

        out_back = mamba_inner_fn_no_out_proj(
            projected_states_back,
            self.conv1d_back.weight,
            self.conv1d_back.bias,
            self.x_proj_back.weight,
            self.dt_proj_back.weight,
            A_back,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D_back.float(),
            delta_bias=self.dt_proj_back.bias.float(),
            delta_softplus=True,
        )


        contextualized_states = F.linear(rearrange(out + out_back.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)
        
        return contextualized_states
            
        
