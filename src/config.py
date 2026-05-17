import yaml
import os


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if 'data_dir' in cfg.get('data', {}):
        base_dir = cfg['data']['data_dir']
        if cfg['data'].get('bj_path') and not os.path.isabs(cfg['data']['bj_path']):
            cfg['data']['bj_path'] = os.path.join(base_dir, os.path.basename(cfg['data']['bj_path']))
        if cfg['data'].get('zj_path') and not os.path.isabs(cfg['data']['zj_path']):
            cfg['data']['zj_path'] = os.path.join(base_dir, os.path.basename(cfg['data']['zj_path']))

    return cfg
