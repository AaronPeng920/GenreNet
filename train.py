from models import GenreNetModule, VGG19Module, ResNet50, AlexNetModule
from model import GenreNet, VGG, ResNet, AlexNet
from dataset import MelSpectrogramDataModule
import yaml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='genrenet',
        choices=['genrenet', 'alexnet', 'resnet', 'vgg'],
        help='choose model to train'
    )
    
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='configs/config_genrenet.yaml',
        help='choose config file to control training progress'
    )
    
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    opt = get_opt()
    
    # load config
    with open(opt.config, 'r') as confread:
        try:
            config = yaml.safe_load(confread)
        except yaml.YAMLError as e:
            print(e)
    train_config = config['train']
    data_config = config['data']

    # init model
    if opt.model == 'genrenet':
        genre_model = GenreNetModule(**config['model'])
        model = GenreFormer(genre_model, **train_config)
    elif opt.model == 'vgg':
        vgg19_model = VGG19Module(**config['model'])
        model = VGG(vgg19_model, **train_config)
    elif opt.model == 'resnet':
        resnet50_model = ResNet50(**config['model'])
        model = ResNet(resnet50_model, **train_config)
    elif opt.model == 'alexnet':
        alexnet_model = AlexNetModule(**config['model'])
        model = AlexNet(alexnet_model, **train_config)

    # init logger
    logger = TensorBoardLogger(**train_config['logging'])

    # seed
    seed_everything(train_config['manual_seed'], True)

    # prepare dataloader
    data = MelSpectrogramDataModule(data_config)
    data.setup()
    print(data.train_dataset.class_to_idx)

    # checkpoint_callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="accuracy",
        mode="max",
        filename="save_ckpt-{epoch:02d}-{val_loss:.2f}"
    )

    # init trainer
    trainer = Trainer(
        logger=logger, 
        num_sanity_val_steps=0, 
        **train_config['trainer'],
        callbacks=[checkpoint_callback]
    )

    # train 
    trainer.fit(model, datamodule=data)

