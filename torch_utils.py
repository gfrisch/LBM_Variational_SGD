import torch
import math
import numpy as np

from torch.distributions.relaxed_categorical import (
    RelaxedOneHotCategorical as RelaxedOneHotCategorical,
)
from torch.distributions.relaxed_bernoulli import (
    RelaxedBernoulli as RelaxedBernoulli,
)

inv_softplus = lambda x: x + torch.log(-torch.expm1(-x))


shrink_simplex_internal = (
    lambda p: 1
    - p[:, :-1]
    / torch.flip(torch.cumsum(torch.flip(p, dims=(0, 1)), dim=1), dims=(0, 1))[
        :, :-1
    ]
)
shrinkpow = lambda s, d: torch.exp(
    (torch.arange(s.shape[1], 0, -1, device=d).reshape((1, -1))) * torch.log(s)
)
shrink_simplex = lambda p, d: shrinkpow(shrink_simplex_internal(p), d)


def random_init_mar(n1, n2, sparsity=0.5, device=torch.device("cpu")):
    r_mu = torch.logit(torch.tensor(sparsity, device=device))
    r_nu_row = torch.zeros((n1, 1), device=device)
    r_nu_col = torch.zeros((1, n2), device=device)
    r_rho_row = inv_softplus(torch.ones((n1, 1), device=device).double())
    r_rho_col = inv_softplus(torch.ones((1, n2), device=device).double())
    r_sigma_sq_row = inv_softplus(torch.ones(1, device=device).double())
    r_sigma_sq_col = inv_softplus(torch.ones(1, device=device).double())

    return {
        "r_mu": torch.nn.Parameter(r_mu),
        "r_nu_row": torch.nn.Parameter(r_nu_row),
        "r_nu_col": torch.nn.Parameter(r_nu_col),
        "r_rho_row": torch.nn.Parameter(r_rho_row),
        "r_rho_col": torch.nn.Parameter(r_rho_col),
        "r_sigma_sq_row": torch.nn.Parameter(r_sigma_sq_row),
        "r_sigma_sq_col": torch.nn.Parameter(r_sigma_sq_col),
    }


def random_init_lbm(n1, n2, nq, nl, device=torch.device("cpu")):
    r_alpha_1 = torch.ones((1, nq), device=device)
    r_alpha_2 = torch.ones((1, nl), device=device)

    r_tau_1 = (
        torch.distributions.Multinomial(
            probs=torch.ones(nq, device=device) / nq
        ).sample((n1,))
        * 4
    )

    r_tau_2 = (
        torch.distributions.Multinomial(
            probs=torch.ones(nl, device=device) / nl
        ).sample((n2,))
        * 4
    )

    return {
        "r_tau_1": torch.nn.Parameter(r_tau_1),
        "r_tau_2": torch.nn.Parameter(r_tau_2),
        "r_alpha_1": torch.nn.Parameter(r_alpha_1),
        "r_alpha_2": torch.nn.Parameter(r_alpha_2),
    }


def random_init_lbm_gaussian(
    n1,
    n2,
    nq,
    nl,
    nb_covariates=0,
    nk=0,
    homoscedastic=False,
    device=torch.device("cpu"),
):
    res = random_init_lbm(n1, n2, nq, nl, device=device)

    if nk:
        r_pi = torch.rand((nk, nq, nl), device=device)
        if homoscedastic:
            r_sigma_sq = inv_softplus(torch.ones(1, device=device))
        else:
            r_sigma_sq = inv_softplus(torch.ones(nk, device=device))
    else:
        r_pi = torch.rand((nq, nl), device=device)
        r_sigma_sq = torch.ones(1, device=device)
    res.update(
        {
            "r_pi": torch.nn.Parameter(r_pi),
            "r_sigma_sq": torch.nn.Parameter(r_sigma_sq),
        }
    )

    if nb_covariates:
        r_beta = torch.rand(nb_covariates, device=device)
        res.update({"r_beta": torch.nn.Parameter(r_beta)})
    return res


def random_init_lbm_bernoulli(
    n1, n2, nq, nl, nb_covariates=0, device=torch.device("cpu")
):
    res = random_init_lbm(n1, n2, nq, nl, device=device)
    if nb_covariates:
        r_pi = torch.rand((nq, nl), device=device)
        r_beta = torch.rand(nb_covariates, device=device)
        res.update({"r_beta": torch.nn.Parameter(r_beta)})
    else:
        r_pi = torch.logit(torch.rand((nq, nl), device=device))
    res.update({"r_pi": torch.nn.Parameter(r_pi)})
    return res


def random_init_lbm_ordinal(
    n1, n2, nq, nl, nb_covariates=0, device=torch.device("cpu")
):
    res = random_init_lbm(n1, n2, nq, nl, device=device)
    r_pi = torch.rand((nq, nl), device=device)
    if nb_covariates:
        r_beta = torch.rand(nb_covariates, device=device)
        res.update({"r_beta": torch.nn.Parameter(r_beta)})
    r_sigma = inv_softplus(0.8 * torch.ones(1, device=device).double())
    res.update(
        {
            "r_pi": torch.nn.Parameter(r_pi),
            "r_sigma": torch.nn.Parameter(r_sigma),
        }
    )

    return res


def random_init_lbm_bernoulli_multiplex(
    n1, n2, nq, nl, nk, device=torch.device("cpu")
):
    res = random_init_lbm(n1, n2, nq, nl, device=device)
    r_pi = torch.rand((nk, nq, nl), device=device)
    r_pi /= r_pi.sum(0).view(1, nq, nl)
    r_pi = torch.logit(
        shrink_simplex(r_pi.view(nk, -1).T, device).T.view(nk - 1, nq, nl)
    )
    res.update({"r_pi": torch.nn.Parameter(r_pi)})

    return res


def random_init_lbm_categorical(
    n1, n2, nq, nl, nb_categories, nb_covariates=0, device=torch.device("cpu")
):
    res = random_init_lbm(n1, n2, nq, nl, device=device)
    if nb_covariates:
        r_pi = torch.rand((nb_categories, nq, nl), device=device)
        r_beta = torch.rand(nb_categories, nb_covariates, device=device)
        res.update({"r_beta": torch.nn.Parameter(r_beta)})
    else:
        r_pi = torch.rand((nb_categories, nq, nl), device=device)
        r_pi /= r_pi.sum(0).view(1, nq, nl)
        r_pi = torch.logit(
            shrink_simplex(r_pi.view(nb_categories, -1).T, device).T.view(
                nb_categories - 1, nq, nl
            )
        )
    res.update({"r_pi": torch.nn.Parameter(r_pi)})

    return res


def random_init_lbm_poisson(
    n1, n2, nq, nl, nb_covariates=None, device=torch.device("cpu")
):
    res = random_init_lbm(n1, n2, nq, nl, device=device)
    r_pi = torch.logit(torch.rand((nq, nl), device=device) / (n1 * n2))
    res.update({"r_pi": torch.nn.Parameter(r_pi)})
    if nb_covariates:
        r_beta = torch.rand(nb_covariates, device=device)
        res.update({"r_beta": torch.nn.Parameter(r_beta)})
    return res


def batch_entropy_categorical(batch_ratio, tau):
    return batch_ratio * (-torch.sum(tau * torch.log(tau)))


def batch_entropy_bernoulli(batch_ratio, tau):
    return batch_ratio * (
        (-tau * torch.log(tau) - (1 - tau) * torch.log(1 - tau)).sum()
    )


def batch_bernoulli_expectation_loglike(batch_ratio, tau, alpha):
    return batch_ratio * (
        tau.sum(0) @ torch.log(alpha) + (1 - tau).sum(0) @ torch.log(1 - alpha)
    )


def batch_categorial_expectation_loglike(batch_ratio, tau, alpha):
    return batch_ratio * (tau.sum(0) @ torch.log(alpha))


def sample_categorial(temperature, prob):
    return RelaxedOneHotCategorical(temperature, probs=prob).rsample().double()


def sample_bernoulli(temperature, prob):
    return RelaxedBernoulli(temperature, probs=prob).rsample().double()


def sample_gaussian(mu, var):
    std = torch.sqrt(var)
    eps = torch.randn_like(std)
    return mu + eps * std


def batch_entropy_gaussian(batch_ratio, nb_samples, rho):
    device = rho.device
    return (
        0.5
        * batch_ratio
        * (
            nb_samples
            * (
                torch.log(
                    torch.tensor(2 * np.pi, dtype=torch.double, device=device)
                )
                + 1
            )
            + torch.sum(torch.log(rho))
        )
    )


def batch_gaussian_expectation_loglike(
    batch_ratio, nb_samples, sigma_sq, nu, rho
):
    device = nu.device
    return batch_ratio * (
        -nb_samples
        / 2
        * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.double, device=device)
            )
            + torch.log(sigma_sq)
        )
        - 1 / (2 * sigma_sq) * torch.sum(rho + nu ** 2)
    )


def gaussian_nll_loss(
    input, target, var, *, full=False, eps=1e-6, reduction="mean"
):
    r"""Gaussian negative log likelihood loss.
    See :class:`~torch.nn.GaussianNLLLoss` for details.
    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    var = var.view(input.size(0), -1)
    if var.size(1) != input.size(1) and var.size(1) != 1:
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var).view(
        input.size(0), -1
    ).sum(dim=1)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


from typing import Union

from scipy.sparse import coo_matrix
from scipy.special import comb


def CARI(
    labels_true_part_1: Union[np.ndarray, list],
    labels_true_part_2: Union[np.ndarray, list],
    labels_pred_part_1: Union[np.ndarray, list],
    labels_pred_part_2: Union[np.ndarray, list],
) -> float:
    """Coclustering Adjusted Rand Index for two sets of biclusters.

    The Coclustering Adjuster Rand Index (CARI) computes a similarity measure
    between two coclusterings and is an adaptation of the
    Adjusted Rand Index (ARI) developed by Hubert and Arabie (1985) from a
    coclustering point of view.
    Like the ARI, this index is symmetric and takes the value 1 when the
    couples of partitions agree perfectly up to a permutation.

    Parameters
    ----------
    labels_true_part_1 : int array, shape = (n_samples_1,)
        Ground truth class labels of the first partition used as reference

    labels_true_part_2 : int array, shape = (n_samples_2,)
        Ground truth class labels of the second partition used as reference

    labels_pred_part_1 : int array, shape = (n_samples_1,)
        Cluster labels of the fist partition to evaluate

    labels_pred_part_2 : int array, shape = (n_samples_2,)
        Cluster labels of the second partition to evaluate

    Returns
    -------
    cari : float
       Similarity score between -1.0 and 1.0. Random labelings have a CARI
       close to 0.0. 1.0 stands for perfect match.

    Examples
    --------
      >>> CARI(
            [0, 0, 1, 1],
            [0, 0, 1, 2, 2, 2],
            [0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2]
        )
      0.649746192893401

    References
    ----------
    .. [Robert2019] Valérie Robert, Yann Vasseur, Vincent Brault.
      Comparing high dimensional partitions with the Coclustering Adjusted Rand
      Index. 2019. https://hal.inria.fr/hal-01524832v4

    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    """

    labels_true_part_1 = np.array(labels_true_part_1).flatten()
    labels_true_part_2 = np.array(labels_true_part_2).flatten()
    labels_pred_part_1 = np.array(labels_pred_part_1).flatten()
    labels_pred_part_2 = np.array(labels_pred_part_2).flatten()

    assert labels_true_part_1.size == labels_pred_part_1.size
    assert labels_true_part_2.size == labels_pred_part_2.size

    n_samples_part_1 = labels_true_part_1.size
    n_samples_part_2 = labels_true_part_2.size

    n_classes_part_1 = len(set(labels_true_part_1))
    n_clusters_part_1 = len(set(labels_pred_part_1))
    n_classes_part_2 = len(set(labels_true_part_2))
    n_clusters_part_2 = len(set(labels_pred_part_2))

    if (
        (
            n_classes_part_1
            == n_clusters_part_1
            == n_classes_part_2
            == n_clusters_part_2
            == 1
        )
        or n_classes_part_1
        == n_clusters_part_1
        == n_classes_part_2
        == n_clusters_part_2
        == 0
        or (
            n_classes_part_1 == n_clusters_part_1 == n_samples_part_1
            and n_classes_part_2 == n_clusters_part_2 == n_samples_part_2
        )
    ):
        return 1.0

    # Compute the contingency data tables
    _, true_class_idx_part_1 = np.unique(
        labels_true_part_1, return_inverse=True
    )
    _, pred_class_idx_part_1 = np.unique(
        labels_pred_part_1, return_inverse=True
    )
    contingency_part_1 = np.zeros((n_classes_part_1, n_clusters_part_1))
    np.add.at(
        contingency_part_1, (true_class_idx_part_1, pred_class_idx_part_1), 1
    )
    _, true_class_idx_part_2 = np.unique(
        labels_true_part_2, return_inverse=True
    )
    _, pred_class_idx_part_2 = np.unique(
        labels_pred_part_2, return_inverse=True
    )
    contingency_part_2 = np.zeros((n_classes_part_2, n_clusters_part_2))
    np.add.at(
        contingency_part_2, (true_class_idx_part_2, pred_class_idx_part_2), 1
    )

    # Theorem 3.3 of Robert2019 (https://hal.inria.fr/hal-01524832v4) defines
    # the final contingency matrix by the Kronecker product between the two
    # contingency matrices of patition 1 and 2.
    contingency_table = np.kron(contingency_part_1, contingency_part_2)
    sum_tt_comb = comb(contingency_table, 2).sum()
    sum_a_comb = comb(contingency_table.sum(axis=1), 2).sum()
    sum_b_comb = comb(contingency_table.sum(axis=0), 2).sum()
    comb_n = comb(n_samples_part_1 * n_samples_part_2, 2).sum()

    ari = ((sum_tt_comb) - (sum_a_comb * sum_b_comb / comb_n)) / (
        0.5 * (sum_a_comb + sum_b_comb) - (sum_a_comb * sum_b_comb) / comb_n
    )
    return ari
