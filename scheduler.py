import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def get_slanted_triangular_scheduler(optimizer, num_epochs, cut_frac=0.2, ratio=32):
    def slanted_triangular_func(epoch):
        cut = int(np.floor(num_epochs * cut_frac))
        # p = epoch / cut if epoch < cut else 1 - (epoch - cut) / (cut * (ratio - 1))
        p = epoch / cut if epoch < cut else (num_epochs - epoch) / (num_epochs - cut)
        return (1 + p * (ratio - 1)) / ratio
    
    scheduler = LambdaLR(optimizer, slanted_triangular_func, verbose=True)
    return scheduler

def get_linear_scheduler(optimizer, num_training_epochs, initial_lr, final_lr=1e-5):
    assert initial_lr > final_lr, "The initial learning rate must be larger than the final learning rate"

    def lr_lambda(epoch):
        return (1 - epoch/num_training_epochs) + (epoch/num_training_epochs) * (final_lr/initial_lr)
    return LambdaLR(optimizer, lr_lambda, verbose=True)