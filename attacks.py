from advertorch.attacks import LinfPGDAttack
import torch.nn as nn


def get_pgd_adversary(model, eps, num_iter, lr):
    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=num_iter, eps_iter=lr, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
    return adversary