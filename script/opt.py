
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Any, Iterable, Dict, Union, TypeAlias


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class Lion(Optimizer):
    def __init__(self, params, lr: Union[float, Tensor] = 1e-3, betas=(0.9, 0.99), weight_decay: float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr = lr, betas = betas, weight_decay = weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign_(), alpha = -group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

        return loss


class Tiger(Optimizer):

    def __init__(
        self, 
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3, 
        beta: float = 0.965, 
        weight_decay: float = 1e-2):

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {lr}')
        if not 0.0 <= beta < 1.0:
            raise ValueError('Invalid beta parameter: {beta}')
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta = group['beta']
                update = beta * exp_avg + (1 - beta) * grad
        
                p.add_(update.sign_(), alpha = -group['lr'])

        return loss