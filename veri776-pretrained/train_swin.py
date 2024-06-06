from argparse import ArgumentParser
import veri776
import Transforms
from model import make_model
from Trainer import ReIDTrainer
from torch.optim import AdamW
import os
from torch.nn import CrossEntropyLoss, TripletMarginLoss
from CenterLoss import CenterLoss
import torch
import numpy as np




if __name__ == '__main__':
    seed = 0xdc51ab
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../veri776')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', '-b', type=int, default=4) # -b=20 is the limit on my machine (12GB GPU memory)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '--wd', default=1e-5)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--margin', '-m', type=float, default=0.6)
    parser.add_argument('--save_dir', '-s', type=str, required=True)
    parser.add_argument('--check_init', action='store_true')
    parser.add_argument('--early_stopping', type=int, default=6)
    parser.add_argument('--embedding_dim', type=int, default=2048)

    args = parser.parse_args()


    # datasets
    train_loader = veri776.get_veri776_train(
        veri776_path=args.dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        transform=Transforms.get_training_transform(),
        drop_last=True,
        shuffle=True
    )

    test_set = veri776.get_veri776_test(
        veri_776_path=args.dataset,
        transform=Transforms.get_test_transform(),
    )

    net = make_model(backbone='swin', num_classes=576)


    # Trainer
    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = ReIDTrainer(
        net=net,
        ce_loss_fn=CrossEntropyLoss(label_smoothing=args.smoothing),
        triplet_loss_fn=TripletMarginLoss(margin=args.margin),
        center_loss_fn=CenterLoss(num_classes=576, feat_dim=args.embedding_dim, use_gpu=True), 
        optimizer=optim,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    trainer.fit(
        train_loader=train_loader,
        test_set=test_set,
        gt_index_path=os.path.join(args.dataset, 'gt_index.txt'),
        name_query_path=os.path.join(args.dataset, 'name_query.txt'),
        jk_index_path=os.path.join(args.dataset, 'jk_index.txt'),
        epochs=args.epochs,
        save_dir=args.save_dir,
        check_init=args.check_init,
        early_stopping=args.early_stopping
    )
