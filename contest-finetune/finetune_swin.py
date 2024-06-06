from argparse import ArgumentParser
import torch
from torch.optim import AdamW, SGD
from torch.nn import CrossEntropyLoss, TripletMarginLoss
from CenterLoss import CenterLoss 
import numpy as np
import MDL


import ContestDataset as cd
import Transforms
from Trainer import Trainer

seed = 0xdc51ab
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='../train')
    parser.add_argument('--disable_join', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--margin', '-m', type=float, default=0.6)
    parser.add_argument('--early_stopping', type=int, default=6)
    parser.add_argument('--weight_decay', '--wd', default=1e-6)


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = Transforms.get_training_transform()
    train_loader, num_classes = cd.get_contest_final(args.dataset, args.workers, args.batch_size, train_transform, args.disable_join)

    test_transform = Transforms.get_test_transform()
    val_set = cd.get_contest_val(args.dataset, test_transform)


    net = MDL.make_finetuned_model(backbone='swin', weights='swin_reid', num_classes=num_classes)
    net = net.to(device)



    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=2048, use_gpu=True)
    center_optim = SGD(center_loss_fn.parameters(), lr=0.5)


    trainer = Trainer(
        net=net,
        ce_loss_fn=CrossEntropyLoss(label_smoothing=args.smoothing),
        triplet_loss_fn=TripletMarginLoss(margin=args.margin),
        center_loss_fn=center_loss_fn,
        center_optimizer=center_optim,
        optimizer=optim,
        device=device,
    )

    trainer.fit(
        train_loader=train_loader,
        val_set=val_set,
        epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping=args.early_stopping,
    )