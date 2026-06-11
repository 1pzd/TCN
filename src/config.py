import yaml
import os


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg['data']['bj_path'] = os.path.join(cfg['data']['data_dir'], 'dataset_BJ.csv')
    cfg['data']['zj_path'] = os.path.join(cfg['data']['data_dir'], 'dataset_ZJ.csv')

    return cfg
