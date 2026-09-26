"""Independent train/validation worker settings, including synchronous loading."""
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


def build_training_loaders(train_data, val_data, config):
    train_params = OmegaConf.to_container(config.params, resolve=True)
    val_params = dict(train_params)
    val_params.update(OmegaConf.to_container(config.get('val_params', {}), resolve=True)
                      if OmegaConf.is_config(config.get('val_params', {}))
                      else dict(config.get('val_params', {})))
    for params in (train_params, val_params):
        if params.get('num_workers', 0) == 0:
            params.pop('prefetch_factor', None)
            params.pop('persistent_workers', None)
            params.pop('multiprocessing_context', None)
    return (DataLoader(train_data, **train_params, shuffle=True),
            DataLoader(val_data, **val_params, shuffle=False))
