"""
script for training the model
uses pt-trainer package for automating training and monitoring
"""

from trainer import Trainer, events, Config
from trainer.handlers import EventSave
from torch.utils.data import DataLoader


config_file = 'config.py'
config = Config.from_file(config_file)

# init trainer
trainer = Trainer.from_config(config)

# setup for validation
validation_loader = DataLoader(config.DATASET(config.dataset['hdf5'], config.dataset['fraction']-1), **config.dataloader)
sample = next(iter(validation_loader))[0:8] + next(iter(trainer.dataloader))[0:8]

# setup monitoring
trainer.register_event_handler(events.EACH_STEP, trainer, name='sample', sample=sample, interval=100)  # sample evolution
trainer.register_event_handler(events.EACH_EPOCH, trainer.validate, dataloader=validation_loader)      # validation
trainer.register_event_handler(events.EACH_EPOCH, EventSave(), interval=100)                           # checkpointing

# train classifier
trainer.train(n_epochs=300, resume=True)
