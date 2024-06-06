WARMUP_EPOCH = 10
DECAY_FACTOR = 0.1
from torch.optim.lr_scheduler import LambdaLR

def warmup_schedule(epoch):
    if epoch < WARMUP_EPOCH:
        return epoch / WARMUP_EPOCH
    else:
        return 1

def decay_schedule(epoch):
    if epoch < WARMUP_EPOCH: 
        return 1
    if WARMUP_EPOCH <= epoch <= 25:
        return  DECAY_FACTOR 

    return DECAY_FACTOR ** 2 

def combined_schedule(epoch):
    return warmup_schedule(epoch) * decay_schedule(epoch)


def get_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=combined_schedule)
