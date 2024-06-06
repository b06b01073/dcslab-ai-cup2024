from tqdm import tqdm
import torch
import os
from einops import rearrange

class ReIDTrainer:
    '''
        A wrapper class that launches the training process 
    '''
    def __init__(self, net, ce_loss_fn=None, triplet_loss_fn=None, center_loss_fn=None, optimizer=None, lr_scheduler=None, device='cpu'):
        '''
            Args: 
                net (nn.Module): the network to be trained 
                ce_loss_fn (CrossEntropyLoss): cross entropy loss function from Pytorch
                triplet_loss_fn (TripletMarginLoss): triplet loss function from Pytorch
                optimizer (torch.optim): optimizer for `net`
                lr_scheduler (torch.optim.lr_scheduler): scheduler for `optimizer`
                device (str, 'cuda' or 'cpu'): the device to train the model 
        '''
        self.device = device
        self.net = net.to(self.device)

        print(f'Training on {self.device}')

        self.ce_loss_fn = ce_loss_fn
        self.triplet_loss_fn = triplet_loss_fn
        self.center_loss_fn = center_loss_fn

        self.optimizer = optimizer
        self.scheduler = lr_scheduler


    def fit(self, train_loader, test_set, epochs, gt_index_path, name_query_path, jk_index_path, save_dir, early_stopping, check_init=False):
        '''
            Train the model for `epochs` epochs, where each epoch is composed of a training step and a testing step 

            Args:
                train_loader (DataLoader): a dataloader that wraps the training dataset (a Veri776Train instance)
                epochs (int): epochs
                gt_index_path (str): the path to gt_index.txt under veri776 root folder
                name_query_path (str): the path to name_query.txt under veri776 root folder
                save_dir (str): path to save the model
                check_init (boolean): if true, then test the model with initial weight
                early_stopping (int): if the performance on validation set stop improving for a continuous `early_stopping` epochs, the `fit` method returns control
        '''

        # if the save if provided and the path hasn't existed yet
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)


        name_queries = self.get_queries(name_query_path)
        gt_indices = self.get_label_indices(gt_index_path)
        jk_indices = self.get_label_indices(jk_index_path)


        if check_init:
            print('check init')
            self.test(
                test_set=test_set, 
                gt_indices=gt_indices, 
                jk_indices=jk_indices, 
                name_queries=name_queries
            )

        best_val = 0
        patience = 0

        for epoch in range(epochs):
            self.train(train_loader)

            complete_hits, gt_hits = self.test(
                test_set=test_set, 
                gt_indices=gt_indices, 
                jk_indices=jk_indices, 
                name_queries=name_queries
            )

            if complete_hits + gt_hits > best_val:
                print('model saved')
                best_val = complete_hits + gt_hits
                patience = 0
                torch.save(self.net.state_dict(), os.path.join(save_dir, f'best.pth'))

            if self.scheduler is not None:
                self.scheduler.step()

            patience += 1
            if patience >= early_stopping:
                return


    def train(self, train_loader):
        '''
            Args:
                train_loader (DataLoader): a dataloader that wraps the training dataset (a ReIDDataset instance)
        '''
        self.net.train()

        total_ce_loss = 0
        total_triplet_loss = 0
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
            loss = triplet_loss + ce_loss + 3.5e-4 * cent_loss
            loss.backward()
            self.optimizer.step()

            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()

        print(f'triplet_loss: {total_triplet_loss:.4f}, ce_loss: {total_ce_loss:.4f}')


    @staticmethod
    def get_queries(name_queries_path):
        '''
            Args:
                name_query_path (str): the path to name_query.txt under veri776 root folder
            Returns:
                return the file names in name_query.txt
               
        '''
        with open(name_queries_path) as f:
            return [line.rstrip('\n') for line in f.readlines()]


    @staticmethod
    def get_label_indices(index_path):
        '''
            Args:
                gt_index_path (str): the path to gt_index.txt under veri776 root folder
            Returns:
                a list composed of ground truths
        '''
        with open(index_path) as f:
            lines = f.readlines()
            ls = []
            for line in lines:
                ls.append([int(x) for x in line.rstrip(' \n').split(' ')])

            return ls


    @torch.no_grad()
    def test(self, test_set, gt_indices, jk_indices, name_queries):
        '''
            Args:
                test_set (Veri776Test): a Veri776Test instance
                gt_indices (list of list): a list of ground truths (see gt_index.txt)
                name_queries (list of str): a list of query file name (see name_query.txt)
        '''

        self.net.eval()

        test_dict = self.build_feature_mapping(test_set)
        query_indices_map = self.build_query_indices(test_set.img_file_names, name_queries)
        
        # img_names = test_set.img_file_names
        test_feats = torch.stack(list(test_dict.values()))

        gt_hits = 0
        complete_hits = 0

        for jk, gt, query in tqdm(zip(jk_indices, gt_indices, name_queries), total=len(gt_indices), dynamic_ncols=True, desc='querying gt'):
            query_feat = test_dict[query]
            dist = self.get_euclidean_dist(query_feat, test_feats)
            sorted_arg = torch.argsort(dist) + 1 # the indices are 0-indexed, however, the gt and jk are 1-indexed
            gt_hits += self.query_gt(sorted_arg, gt, jk, query_indices_map[query])
            complete_hits += self.query_complete(sorted_arg, gt + jk, query_indices_map[query])


        print(f'R@1 gt_hits: {gt_hits / len(name_queries)}, R@1 complete_hits: {complete_hits / len(name_queries)}')

        return complete_hits, gt_hits


    def query_complete(self, sorted_args, complete, query_idx):
        for idx in sorted_args:
            if idx == query_idx:
                continue

            return 1 if idx in complete else 0


    def query_gt(self, sorted_arg, gt, jk, query_idx):
        '''
            Return 1 if the closest image which (is not in `jk` and is not itself) is in gt, otherwise return  0
        '''

        for idx in sorted_arg:
            if (idx in jk) or (idx == query_idx):
                continue

            return 1 if idx in gt else 0



    def build_query_indices(self, img_file_names, name_queries):
        img_ptr = 0
        query_ptr = 0

        indices = dict()

        while query_ptr < len(name_queries) and img_ptr < len(img_file_names):
            if name_queries[query_ptr] == img_file_names[img_ptr]:
                indices[name_queries[query_ptr]] = img_ptr + 1 # the query, ground truth, ..., should be 1-indexed
                query_ptr += 1
                img_ptr += 1
            else:
                img_ptr += 1

        return indices


    def get_euclidean_dist(self, query_feats, test_feats):
        ''' calculate the Euclidean distance between the query and the entire test set 

            Args: 
                query_feats (torch.tensor): the embedding feature representation of the query img
                test_feats (torch.tensor): the embedding feature representations of the entire test set

            Returns:
                returns a 1d torch.tensor where entry `i` represents the distance between the query_feats and the `i`th test_feats
        '''
        query_feats = rearrange(query_feats, '(b n f) -> b n f', b=1, n=1)
        test_feats = rearrange(test_feats, '(b n) f -> b n f', b=1)

        dist_mat = torch.cdist(query_feats, test_feats).squeeze()

        return dist_mat

    
    def build_feature_mapping(self, test_set):
        '''
            Args:
                test_set (Veri776Test): a Veri776Test instance

            Returns:
                a dictionary which maps the file name in the test set to its embedding feature
        '''
        test_dict = dict()
        
        for img_name, img in tqdm(test_set, dynamic_ncols=True, desc='Trainer.build_feature_mapping'):
            img = img.to(self.device)
            feat, _, _ = self.net(img.unsqueeze(dim=0)) # use f_t for now

            test_dict[img_name] = feat.squeeze()


        return test_dict