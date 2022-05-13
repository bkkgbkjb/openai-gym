from torch import nn


class NeuralNetworks(nn.Module):
    def __init__(self):
        super(NeuralNetworks, self).__init__()

    def hard_update_to(self, target: nn.Module):
        # assert self.parameters() == len(target.parameters())
        for c, t in zip(self.parameters(), target.parameters()):
            c.data.copy_(t.data)
        return self

    def soft_update_to(self, target: nn.Module, tau: float):
        for c, t in zip(self.parameters(), target.parameters()):
            c.data.copy_(c.data * (1 - tau) + t.data * tau)
        return self

    def no_grad(self):
        for p in self.parameters():
            p.requires_grad = False
        return self
