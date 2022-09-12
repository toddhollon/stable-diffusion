import sys
from ldm.models.autoencoder import VQModel, AutoencoderKL
import yaml
from pathlib import Path
from pytorch_model_summary import summary
import torch



# example of VQ model
config = yaml.safe_load(Path('configs/latent-diffusion/srh_config_vq_v1.yaml').read_text())
print(config)

model = VQModel(
	ddconfig=config['model']['params']['first_stage_config']['params']['ddconfig'],
	lossconfig=config['model']['params']['first_stage_config']['params']['lossconfig'],
	embed_dim=2,
	n_embed=8192, 

)


# example of VQ model
config = yaml.safe_load(Path(''))
print(model)
# print(summary(model, torch.zeros((1, 2, 256, 256)), show_input=True))
