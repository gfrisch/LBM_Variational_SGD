import torch
from lbm_abstract import LbmAbstract
from torch_utils import sample_categorial
from optimizer import train_model


class LbmBernoulli(LbmAbstract):
    def __init__(self, temperature=0.6, device=None):
        super().__init__(temperature, device)

    @property
    def prob_func(self):
        return "bernoulli"

    @property
    def pi(self):
        return torch.sigmoid(self.params["r_pi"]).double()

    def fit(self, X, nq, nl, *, init_parameters=None, **kwargs):
        X, _ = self.check_params(X, nq, nl, init_parameters=init_parameters)
        train_model(self, X, **kwargs)
        return self

    def decode(self, rows_sampled, columns_sampled, *args):
        Y1_sampled = sample_categorial(
            self.temperature, self.tau_1[rows_sampled,]
        )
        Y2_sampled = sample_categorial(
            self.temperature, self.tau_2[columns_sampled,]
        )
        prob_X = Y1_sampled @ self.pi @ Y2_sampled.T
        probas = torch.cat(((1 - prob_X), prob_X))
        return probas.view(2, -1).T
