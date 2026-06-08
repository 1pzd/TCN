import yaml
import os


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return cfg
