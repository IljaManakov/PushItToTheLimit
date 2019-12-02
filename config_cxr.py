"""
config file for training a model, used in conjunction with Trainer.from_config
"""

from autoencoder import ConvAE2d
from dataset import OCTDataset
from torch.optim import Adam
# from apex.fp16_utils import FP16_Optimizer
from torch.nn import MSELoss, ZeroPad2d
from torch import float16, float32


dtype = float32
cuda = True
seed = 1278654

MODEL = ConvAE2d
LOSS = MSELoss
DATASET = OCTDataset
OPTIMIZER = Adam
# APEX = FP16_Optimizer
LOGDIR = ''  # enter folder for logs and checkpoints

model = {
    'kernel_size': (3, 3),
    'n_residual': (2, 2),
    'affine': True,
    'channels': [4, 8, 16, 32, 64, 128],
    'padding': ZeroPad2d
}

dataset = {
    'folder': '',  # enter root folder containing the dataset
    'fraction': 0.8
}

dataloader = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 8
}

loss = {
    # 'window_size': 5
}

optimizer = {
    'lr': 0.0005
}

# apex = {
#     'dynamic_loss_scale': True,
#     'dynamic_loss_args': {'init_scale': 2**16},
#     'verbose': False
# }

trainer = {
    'loss_decay': 0.0,
    'split_sample': lambda x: (x[0], x[0].float())
}
