import argparse

import torch
from torch import nn

from utils.train import train
from model.UNet import Unet
from data.dataset import COCOStuff10kDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Unet trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--epoch", default=100, type=int, help="number epochs")
parser.add_argument("--block", default='resnet', type=str, help="block type")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--state_path", default="model states/", type=str, help="Path to model states")
parser.add_argument("path", metavar="PATH", type=str, help="Path to dataset")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.path

    train_dataset = COCOStuff10kDataset(dataset_path, split='train')
    test_dataset = COCOStuff10kDataset(dataset_path, split='test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch)

    segment_model = Unet(block=args.block).to(device)
    optimizer = torch.optim.Adam(segment_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    state_dict_path = args.state_path + f'model states/mobileUnetResnet1.pt'
    savefig_dir = args.state_path + 'result images/'

    train(
        segment_model,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
        state_dict_path=state_dict_path,
        device=device,
        n_epochs=args.epoch,
        show_interval=1
    )
