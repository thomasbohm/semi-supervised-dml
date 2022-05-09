import torch
import warnings
import argparse
import yaml

from trainer import Trainer

warnings.filterwarnings("ignore")


def main(config_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    trainer = Trainer(config, device)
    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DML Baseline')
    parser.add_argument('--config_path', type=str, default='config/cars.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config_path)
