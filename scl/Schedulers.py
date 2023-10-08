import math


def linear_warmup_cosine_anneal(warmup_steps, total_steps, min_lr=6e-5):
    # https://github.com/Lightning-Universe/lightning-bolts/blob/ba6b4c679ed72901923089ae01f5d4565aa7d12e/src/pl_bolts/optimizers/lr_scheduler.py#L128
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    def foo(step):
        if step < warmup_steps:
            eta = step / warmup_steps
        else:
            T_i = step - warmup_steps
            T_max = total_steps - warmup_steps
            eta = max(0.5 * (1 + math.cos(T_i / T_max * math.pi)), min_lr)
        return eta
    return foo
