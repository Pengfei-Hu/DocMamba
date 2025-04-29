import sys
sys.path.append('./')
sys.path.append('../')
import torch

from layoutlmft.models.docmamba.modeling_docmamba import BiMambaModel
from transformers.models.mamba import MambaModel, MambaConfig

batch_size = 8
length = 128
num_channel = 768
mamba_config_path = './config.json'

input_embeddings = torch.randn(batch_size, length, num_channel).cuda()
mamba_config = MambaConfig.from_pretrained(mamba_config_path)

# left-to-right Mamba
encoder = MambaModel(mamba_config).cuda()
encoder.embeddings = None
encoder_outputs = encoder(inputs_embeds=input_embeddings, return_dict=True)
output = encoder_outputs.last_hidden_state

# bidirectional Mamba
biencoder = BiMambaModel(mamba_config).cuda()
biencoder.embeddings = None
biencoder_outputs = biencoder(inputs_embeds=input_embeddings, return_dict=True)
bioutput = biencoder_outputs.last_hidden_state
