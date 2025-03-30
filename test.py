import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance


def main(data_path, weight_path, CFG, show):
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=1, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show)
    test.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/zjl/code/DFMS-Net-master/datasets/DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", default="/home/zjl/code/DFMS-Net-master/saved/DFMS/241003105721/checkpoint-epoch50.pth", type=str,
                        help='the path of weight.pt')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()
    with open("config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    main(args.dataset_path, args.wetght_path, CFG, args.show)
