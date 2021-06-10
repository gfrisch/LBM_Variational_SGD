import abc

import torch
from torch_utils import (
    random_init_lbm,
    batch_entropy_categorical,
    batch_categorial_expectation_loglike,
    sample_categorial,
    gaussian_nll_loss,
)

from torch_utils import (
    random_init_lbm_bernoulli,
    random_init_lbm_bernoulli_multiplex,
    random_init_lbm_poisson,
    random_init_lbm_categorical,
    random_init_lbm_ordinal,
    random_init_lbm_gaussian,
)


class LbmAbstract(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self, temperature=0.6, device=None):
        super(LbmAbstract, self).__init__()
        self.temperature = temperature
        self.params = torch.nn.ParameterDict({})
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cpu")

    @abc.abstractmethod
    def fit(self, n1, n2, nq, nl):
        pass

    @property
    @abc.abstractmethod
    def prob_func(self):
        pass

    @property
    @abc.abstractmethod
    def pi(self):
        pass

    @abc.abstractmethod
    def decode(self, rows_sampled, columns_sampled):
        pass

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    @property
    def alpha_1(self):
        return (
            torch.nn.functional.softmax(self.params["r_alpha_1"], 1)
            .double()
            .reshape(self.nq, 1)
        )

    @property
    def alpha_2(self):
        return (
            torch.nn.functional.softmax(self.params["r_alpha_2"], 1)
            .double()
            .reshape(self.nl, 1)
        )

    @property
    def tau_1(self):
        return torch.nn.functional.softmax(self.params["r_tau_1"], 1).double()

    @property
    def tau_2(self):
        return torch.nn.functional.softmax(self.params["r_tau_2"], 1).double()

    def check_params(
        self,
        X,
        nq,
        nl,
        *,
        covariates=None,
        init_parameters=None,
        missing_data=False,
    ):
        x_shape = X.shape
        self.n1, self.n2 = X.shape[-2:]
        self.nq, self.nl = nq, nl
        nk = x_shape[0] if len(x_shape) == 3 else 0
        if nk:
            self.nk = nk
        if self.prob_func in ["categorical", "ordinal"]:
            self.nb_categories = int(X.max().item() + 1)
            if missing_data:
                self.nb_categories -= 1  # Zeros are for missing values

        if init_parameters:
            self.params.update(init_parameters)
        else:
            nb_covariates = (
                covariates.shape[-1] if covariates is not None else 0
            )
            if self.prob_func == "bernoulli":
                if nk:
                    self.params.update(
                        random_init_lbm_bernoulli_multiplex(
                            self.n1,
                            self.n2,
                            self.nq,
                            self.nl,
                            self.nk,
                            device=self.device,
                        )
                    )
                else:
                    self.params.update(
                        random_init_lbm_bernoulli(
                            self.n1,
                            self.n2,
                            self.nq,
                            self.nl,
                            nb_covariates=nb_covariates,
                            device=self.device,
                        )
                    )
            elif self.prob_func == "poisson":
                self.params.update(
                    random_init_lbm_poisson(
                        self.n1,
                        self.n2,
                        self.nq,
                        self.nl,
                        nb_covariates=nb_covariates,
                        device=self.device,
                    )
                )
            elif self.prob_func == "categorical":
                self.params.update(
                    random_init_lbm_categorical(
                        self.n1,
                        self.n2,
                        self.nq,
                        self.nl,
                        self.nb_categories,
                        device=self.device,
                    )
                )
            elif self.prob_func == "ordinal":
                self.params.update(
                    random_init_lbm_ordinal(
                        self.n1,
                        self.n2,
                        self.nq,
                        self.nl,
                        nb_covariates=nb_covariates,
                        device=self.device,
                    )
                )
                # Fixed levels for ordinal regression.
                self._theta = (
                    torch.arange(0, self.nb_categories - 1, device=self.device)
                    + 0.5
                )
                self.register_buffer("theta", self._theta)
            elif self.prob_func == "gaussian":
                self.params.update(
                    random_init_lbm_gaussian(
                        self.n1,
                        self.n2,
                        self.nq,
                        self.nl,
                        nb_covariates=nb_covariates,
                        nk=nk,
                        homoscedastic=(
                            self.homoscedastic
                            if hasattr(self, "homoscedastic")
                            else False
                        ),
                        device=self.device,
                    )
                )
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device)
        else:
            X = X.to(self.device)
        if covariates is not None:
            if not isinstance(covariates, torch.Tensor):
                covariates = torch.tensor(covariates, device=self.device)
            else:
                covariates = covariates.to(self.device)

        return X, covariates

    def batch_entropy(self, rows_sampled, columns_sampled):
        batch_ratio_rows = columns_sampled.size()[0] / self.n2
        batch_ratio_cols = rows_sampled.size()[0] / self.n1
        return batch_entropy_categorical(
            batch_ratio_rows, self.tau_1[rows_sampled]
        ) + batch_entropy_categorical(
            batch_ratio_cols, self.tau_2[columns_sampled]
        )

    def batch_expectation_loglike_latent(self, rows_sampled, columns_sampled):
        batch_ratio_rows = columns_sampled.size()[0] / self.n2
        batch_ratio_cols = rows_sampled.size()[0] / self.n1
        return batch_categorial_expectation_loglike(
            batch_ratio_rows, self.tau_1[rows_sampled], self.alpha_1
        ) + batch_categorial_expectation_loglike(
            batch_ratio_cols, self.tau_2[columns_sampled], self.alpha_2
        )

    def forward(
        self, x, rows_sampled, columns_sampled, cov_batch=None, **kwargs
    ):
        reconstructed_x = self.decode(rows_sampled, columns_sampled, cov_batch)
        loss = -self.batch_entropy(
            rows_sampled, columns_sampled
        ) - self.batch_expectation_loglike_latent(
            rows_sampled, columns_sampled
        )
        if self.prob_func in ["bernoulli", "categorical", "ordinal"]:
            loss += torch.nn.functional.nll_loss(
                torch.log(reconstructed_x), x.flatten(), reduction="sum"
            )
        elif self.prob_func == "poisson":
            loss += torch.nn.functional.poisson_nll_loss(
                reconstructed_x, x.flatten(), log_input=False, reduction="sum"
            )
        elif self.prob_func == "gaussian":
            nk = 1 if len(x.shape) == 2 else x.shape[0]
            var = (
                self.sigma_sq.repeat(nk)
                if self.sigma_sq.flatten().size(0) == 1
                else self.sigma_sq
            )
            loss += gaussian_nll_loss(
                reconstructed_x,
                x.view(nk, -1),
                var,
                full=True,
                reduction="sum",
            )
        else:
            raise Exception("prob_func does not exist")
        return loss
