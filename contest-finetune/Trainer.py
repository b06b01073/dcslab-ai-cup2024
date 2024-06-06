import os
import torch
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from contextlib import nullcontext

class Trainer:
    def __init__(self, net, ce_loss_fn, triplet_loss_fn, center_loss_fn, center_optimizer, optimizer, lr_scheduler=None, device='cuda'):
        self.device = device
        self.net = net.to(self.device)

        print(f'Training on {self.device}')

        self.ce_loss_fn = ce_loss_fn
        self.triplet_loss_fn = triplet_loss_fn

        self.center_loss_fn = center_loss_fn
        self.center_optimizer = center_optimizer

        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        

    def fit(self, train_loader, val_set, epochs, save_dir, early_stopping, check_init=True):
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if check_init:
            self.test(val_set)


        best_eu_val = 0
        best_cos_val = 0

        patience = 0

        for epoch in range(epochs):
            print(f'start epoch {epoch}')

            self.train(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()


            eu_rank1, cos_rank1 = self.test(val_set)

            if eu_rank1 > best_eu_val:
                best_eu_val = eu_rank1
                torch.save(self.net.state_dict(), os.path.join(save_dir, 'best_eu.pth'))
                patience = 0

            if cos_rank1 > best_cos_val:
                best_cos_val = cos_rank1
                torch.save(self.net.state_dict(), os.path.join(save_dir, 'best_cos.pth'))
                patience = 0


            torch.save(self.net.state_dict(), os.path.join(save_dir, 'last.pth'))
            
            patience += 1

            print(f'patience: {patience}')

            if patience >= early_stopping:
                print('early stopped')
                break

    def train(self, train_loader):
        self.net.train()

        total_ce_loss = 0
        total_triplet_loss = 0
        total_cent_loss = 0
        for images, labels in tqdm(train_loader, dynamic_ncols=True, desc='train'):
            images, labels = images.to(self.device), labels.to(self.device)
            
            anchors = images[:, 0, :].squeeze()
            positvies = images[:, 1, :].squeeze()
            negatives = images[:, 2, :].squeeze()

            anchor_embeddings, _, anchor_out = self.net(anchors)
            positive_embeddings, _, positive_out = self.net(positvies)
            negative_embeddings, _, negative_out = self.net(negatives)

            triplet_loss = self.triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

            
            preds = rearrange([anchor_out, positive_out, negative_out], 't b e -> (b t) e')
            labels_batch = torch.flatten(labels)

            ce_loss = self.ce_loss_fn(preds, labels_batch)


            cent_preds = rearrange([anchor_embeddings, positive_embeddings, negative_embeddings], 't b e -> (b t) e')

            cent_loss = self.center_loss_fn(cent_preds, labels_batch)


            self.optimizer.zero_grad()
            self.center_optimizer.zero_grad()

            loss = triplet_loss + ce_loss + self.center_loss_fn.alpha * cent_loss
            loss.backward()
            self.optimizer.step()

            for param in self.center_loss_fn.parameters():
                param.grad.data *= (1. / self.center_loss_fn.alpha)
            self.center_optimizer.step()

            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_cent_loss += cent_loss.item()

        print(f'triplet_loss: {total_triplet_loss:.4f}, ce_loss: {total_ce_loss:.4f}, cent_loss: {total_cent_loss:.4f}')


    @torch.no_grad()
    def test(self, val_set):
        self.net.eval()
        gt_ids = []
        test_euclidean_embeddings = []
        test_cosine_embeddings = []

        for file_name, id in tqdm(val_set.gt, dynamic_ncols=True, desc='test gt'):
            img = Image.open(file_name)

            if val_set.transform is not None:
                img = val_set.transform(img)

            img = img.to(self.device)
            euclidean_embedding, cosine_embedding, _ = self.net(img.unsqueeze(dim=0))

            test_euclidean_embeddings.append(euclidean_embedding.to('cpu'))
            test_cosine_embeddings.append(cosine_embedding.to('cpu'))

            gt_ids.append(id)


        gt_ids = torch.IntTensor(gt_ids)
        test_euclidean_embeddings = torch.stack(test_euclidean_embeddings)
        test_cosine_embeddings = torch.stack(test_cosine_embeddings)


        euclidean_hits = 0
        cosine_hits = 0


        for file_name, id in tqdm(val_set.queries, dynamic_ncols=True, desc='matching'):
            img = Image.open(file_name)
            
            if val_set.transform is not None:
                img = val_set.transform(img)

            img = img.to(self.device)

            euclidean_query, cosine_query, _ = self.net(img.unsqueeze(dim=0))
            
            euclidean_dist_mat = self.get_euclidean_dist(euclidean_query.to('cpu'), test_euclidean_embeddings)
            cosine_dist_mat = self.get_cosine_dist(cosine_query.to('cpu'), test_cosine_embeddings)


            # print(id, gt_ids[torch.argmin(dist_mat)], id == gt_ids[torch.argmin(dist_mat)].item())

            if id == gt_ids[torch.argmin(euclidean_dist_mat)].item():
                euclidean_hits += 1

            if id == gt_ids[torch.argmin(cosine_dist_mat)].item():
                cosine_hits += 1

        print(f'R@1: Euclidean {euclidean_hits / len(val_set.queries)}, Cosine {cosine_hits / len(val_set.queries)}')
        return euclidean_hits / len(val_set.queries), cosine_hits / len(val_set.queries)


    def get_euclidean_dist(self, query_embedding, test_embeddings):

        query_embedding = rearrange(query_embedding, '(b n) f -> b n f', b=1, n=1)
        test_embeddings = rearrange(test_embeddings, 'b n f -> n b f')

        dist_mat = torch.cdist(query_embedding, test_embeddings).squeeze()

        return dist_mat


    def get_cosine_dist(self, query_embedding, test_embeddings):
        test_embeddings = rearrange(test_embeddings, 'b n f -> (b n) f')
        return 1 - pairwise_cosine_similarity(query_embedding, test_embeddings)