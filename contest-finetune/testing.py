from argparse import ArgumentParser
import torch
from torch.optim import SGD
import Scheduler
from torch.nn import CrossEntropyLoss, TripletMarginLoss
import torch.nn as nn
import numpy as np



import MDL as MDL
import ContestDataset as cd
import Transforms
from Trainer import Trainer

seed = 0xdc51ab
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='./resnet101_ibn_a.pth')
    parser.add_argument('--dataset', '-d', type=str, default='../train')
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--embedding_dim', type=int, default=2048)


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_transform = Transforms.get_test_transform()
    queries = np.load('./query_indices.npy')
    val_set = cd.get_contest_test(args.dataset, test_transform)

    # net = MDL.make_ibn_model(args.backbone, num_classes=3421, embedding_dim=args.embedding_dim)

    net = MDL.make_finetuned_model(backbone='swin', weights=None, num_classes=3421)
    net = net.to(device)

    net.load_state_dict(torch.load(args.model))
    net = net.to(device)
    net.eval()



    trainer = Trainer(
        net=net,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        ce_loss_fn=None,
        triplet_loss_fn=None,
        center_loss_fn=None,
        center_optimizer=None,
        optimizer=None
    )

    trainer.test(
        val_set=val_set,
    )

    # print(net)

