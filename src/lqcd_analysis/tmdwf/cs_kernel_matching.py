from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.special as sp

CF = 4.0 / 3.0
NF = 3.0
TF = 1.0 / 2.0
CA = 3.0
FM_GEV = 5.0676896
GAMMA_E = 0.5772156649

BETA = {
    0: 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * NF,
    1: 34.0 / 3.0 * CA**2 - (20.0 / 3.0 * CA + 4.0 * CF) * TF * NF,
    2: 2857.0 / 54.0 * CA**3
    + (2.0 * CF**2 - 205.0 / 9.0 * CF * CA - 1415.0 / 27.0 * CA**2) * TF * NF
    + (44.0 / 9.0 * CF + 158.0 / 27.0 * CA) * TF**2 * NF**2,
}

CUSP = {
    0: 2.0 * CF,
    1: 2.0 * CF * ((67.0 / 9.0 - np.pi**2 / 3.0) * CA - 20.0 / 9.0 * TF * NF),
    2: 2.0
    * CF
    * (
        CA**2 * (245.0 / 6.0 - 134.0 * np.pi**2 / 27.0 + 11.0 * np.pi**4 / 45.0 + 22.0 / 3.0 * sp.zeta(3))
        + CA * TF * NF * (-418.0 / 27.0 + 40.0 * np.pi**2 / 27.0 - 56.0 / 3.0 * sp.zeta(3))
        + CF * TF * NF * (-55.0 / 3.0 + 16.0 * sp.zeta(3))
        - 16.0 / 27.0 * TF**2 * NF**2
    ),
}

GI_NONCUSP_MU = {
    0: -2.0 * CF,
    1: CF
    * (
        CF * (-4.0 + 14.0 / 3.0 * np.pi**2 - 24.0 * sp.zeta(3))
        + CA * (-554.0 / 27.0 - 11.0 * np.pi**2 / 6.0 + 22.0 * sp.zeta(3))
        + NF * (80.0 / 27.0 + np.pi**2 / 3.0)
    ),
}

GI_NONCUSP_C = {
    0: 2.0 * CF - complex(0.0, np.pi * CUSP[0]),
    1: CF
    * (
        CF * (4.0 - 14.0 / 3.0 * np.pi**2 + 24.0 * sp.zeta(3))
        + CA * (950.0 / 27.0 + 11.0 * np.pi**2 / 9.0 - 22.0 * sp.zeta(3))
        + NF * (-152.0 / 27.0 - 2.0 * np.pi**2 / 9.0)
    )
    - complex(0.0, np.pi * (CUSP[1] + BETA[0] * (2.0 * 2.0 * CF - CUSP[0]))),
}

CG_NONCUSP_MU = {
    0: -6.0 * CF,
    1: 0.0,
}

CG_NONCUSP_C = {
    0: 6.0 * CF,
    1: 0.0,
}

ORDER_LABEL_TO_LEVEL = {
    "LO": 0,
    "NLO": 1,
    "NLL": 1,
    "NNLO": 2,
    "NNLL": 2,
}


def normalize_cs_scheme(label: str) -> str:
    scheme = str(label).strip().upper()
    if scheme != "CG":
        raise ValueError(f"unsupported TMDWF CS-kernel scheme '{label}'; currently supported schemes: CG")
    return scheme


def perturbative_order_from_label(label: str) -> int:
    normalized = str(label).strip().upper()
    if normalized not in ORDER_LABEL_TO_LEVEL:
        allowed = ", ".join(sorted(ORDER_LABEL_TO_LEVEL))
        raise ValueError(f"unsupported CS-kernel label '{label}'; expected one of: {allowed}")
    return ORDER_LABEL_TO_LEVEL[normalized]


def uses_rg_improved_matching(label: str) -> bool:
    return str(label).strip().upper() in {"NLL", "NNLL"}


def alphas_nloop(mu: float, order: int = 0, n_flavors: int = 3) -> float:
    del n_flavors  # Legacy implementation fixes Nf=3 in the beta coefficients above.
    a_s = 0.293 / (4.0 * np.pi)
    x = 1.0 + a_s * BETA[0] * np.log((mu / 2.0) ** 2)
    if order < 1:
        return float(a_s * 4.0 * np.pi / x)
    if order == 1:
        return float(a_s * 4.0 * np.pi / (x + a_s * BETA[1] / BETA[0] * np.log(x)))
    if order == 2:
        correction = a_s**2 * (
            BETA[2] / BETA[0] * (1.0 - 1.0 / x)
            + BETA[1] ** 2 / BETA[0] ** 2 * (np.log(x) / x + 1.0 / x - 1.0)
        )
        return float(a_s * 4.0 * np.pi / (x + a_s * BETA[1] / BETA[0] * np.log(x) + correction))
    raise ValueError(f"unsupported running-coupling order {order}; maximum supported order is 2")


def kernel_cg(mu0: float, mu: float, order: int) -> complex:
    lz = np.log(mu0**2 / mu**2)
    a_s = alphas_nloop(mu, order - 1) / (4.0 * np.pi)
    kernel = 1.0 + 0.0j
    if order > 0:
        kernel += a_s * CF * (-0.5 * lz**2 + 3.0 * lz - 12.0 + 7.0 * np.pi**2 / 12.0)
    if order > 1:
        raise ValueError("NNLO CG matching is not available in the legacy CS-kernel implementation")
    return complex(kernel)


def kernel_cg_k1(mu0: float, mu: float, order: int) -> complex:
    r = alphas_nloop(mu, order) / alphas_nloop(mu0, order)
    a_s = alphas_nloop(mu0, order) / (4.0 * np.pi)
    kernel = 1.0 / a_s * (1.0 - 1.0 / r - np.log(r))
    if order > 0:
        kernel += (CUSP[1] / CUSP[0] - BETA[1] / BETA[0]) * (1.0 - r + np.log(r))
        kernel += BETA[1] / (2.0 * BETA[0]) * np.log(r) ** 2
    if order > 1:
        kernel += a_s * (
            ((BETA[1] / BETA[0]) ** 2 - BETA[2] / BETA[0]) * ((1.0 - r**2) / 2.0 + np.log(r))
            + (BETA[1] / BETA[0] * CUSP[1] / CUSP[0] - (BETA[1] / BETA[0]) ** 2) * (1.0 - r + r * np.log(r))
            - (CUSP[2] / CUSP[0] - BETA[1] / BETA[0] * CUSP[1] / CUSP[0]) * (1.0 - r) ** 2 / 2.0
        )
    if order > 2:
        raise ValueError("NNNLL CG matching is not available in the legacy CS-kernel implementation")
    return complex(-CUSP[0] / (4.0 * BETA[0] ** 2) * kernel)


def kernel_cg_k3(mu0: float, mu: float, order: int) -> complex:
    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    kernel = np.log(r) if order > 0 else 0.0
    if order > 2:
        raise ValueError("NNNLL CG matching is not available in the legacy CS-kernel implementation")
    return complex(-CG_NONCUSP_MU[0] / (2.0 * BETA[0]) * kernel)


def kernel_cg_eta(mu0: float, mu: float, order: int) -> complex:
    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    a_s = alphas_nloop(mu0, order - 1) / (4.0 * np.pi)
    kernel = np.log(r) if order > 0 else 0.0
    if order > 1:
        kernel += a_s * (CUSP[1] / CUSP[0] - BETA[1] / BETA[0]) * (r - 1.0)
    if order > 2:
        raise ValueError("NNNLL CG matching is not available in the legacy CS-kernel implementation")
    del kernel
    return 0.0j


def kernel_gi(mu0: float, mu: float, order: int) -> complex:
    lz = np.log(mu0**2 / mu**2) + complex(0.0, np.pi)
    a_s = alphas_nloop(mu, order - 1) / (4.0 * np.pi)
    kernel = 1.0 + 0.0j
    if order > 0:
        kernel += a_s * CF * (-0.5 * lz**2 + lz - 2.0 - 5.0 * np.pi**2 / 12.0)
    if order > 1:
        lz4 = CF / 4.0 * lz**4
        lz3 = -(CF - 11.0 / 9.0 * CA + 2.0 / 9.0 * NF) * lz**3
        lz2 = ((3.0 + 5.0 * np.pi**2 / 12.0) * CF + (np.pi**2 / 3.0 - 100.0 / 9.0) * CA + 16.0 / 9.0 * NF) * lz**2
        lz1 = -(
            (11.0 * np.pi**2 / 2.0 - 24.0 * sp.zeta(3)) * CF
            + (22.0 * sp.zeta(3) - 44.0 * np.pi**2 / 9.0 - 950.0 / 27.0) * CA
            + (152.0 / 27.0 + 8.0 * np.pi**2 / 9.0) * NF
        ) * lz
        lz0 = (
            (-30.0 * sp.zeta(3) + 65.0 * np.pi**2 / 3.0 - 167.0 * np.pi**4 / 144.0 - 16.0) * CF
            + (241.0 * sp.zeta(3) / 9.0 + 53.0 * np.pi**4 / 60.0 - 1759.0 * np.pi**2 / 108.0 - 3884.0 / 81.0) * CA
            + (2.0 * sp.zeta(3) / 9.0 + 113.0 * np.pi**2 / 54.0 + 656.0 / 81.0) * NF
        )
        kernel += a_s**2 * CF / 2.0 * (lz4 + lz3 + lz2 + lz1 + lz0)
    if order > 2:
        raise ValueError("NNNLO GI matching is not available in the legacy CS-kernel implementation")
    return complex(kernel)


def kernel_gi_k1(mu0: float, mu: float, order: int) -> complex:
    r = alphas_nloop(mu, order) / alphas_nloop(mu0, order)
    a_s = alphas_nloop(mu0, order) / (4.0 * np.pi)
    kernel = 1.0 / a_s * (1.0 - 1.0 / r - np.log(r))
    if order > 0:
        kernel += (CUSP[1] / CUSP[0] - BETA[1] / BETA[0]) * (1.0 - r + np.log(r))
        kernel += BETA[1] / (2.0 * BETA[0]) * np.log(r) ** 2
    if order > 1:
        kernel += a_s * (
            ((BETA[1] / BETA[0]) ** 2 - BETA[2] / BETA[0]) * ((1.0 - r**2) / 2.0 + np.log(r))
            + (BETA[1] / BETA[0] * CUSP[1] / CUSP[0] - (BETA[1] / BETA[0]) ** 2) * (1.0 - r + r * np.log(r))
            - (CUSP[2] / CUSP[0] - BETA[1] / BETA[0] * CUSP[1] / CUSP[0]) * (1.0 - r) ** 2 / 2.0
        )
    if order > 2:
        raise ValueError("NNNLL GI matching is not available in the legacy CS-kernel implementation")
    return complex(-CUSP[0] / (4.0 * BETA[0] ** 2) * kernel)


def kernel_gi_k3(mu0: float, mu: float, order: int) -> complex:
    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    a_s = alphas_nloop(mu0, order - 1) / (4.0 * np.pi)
    kernel = np.log(r) if order > 0 else 0.0
    if order > 1:
        kernel += a_s * (GI_NONCUSP_MU[1] / GI_NONCUSP_MU[0] - BETA[1] / BETA[0]) * (r - 1.0)
    if order > 2:
        raise ValueError("NNNLL GI matching is not available in the legacy CS-kernel implementation")
    return complex(-GI_NONCUSP_MU[0] / (2.0 * BETA[0]) * kernel)


def kernel_gi_eta(mu0: float, mu: float, order: int) -> complex:
    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    a_s = alphas_nloop(mu0, order - 1) / (4.0 * np.pi)
    kernel = np.log(r) if order > 0 else 0.0
    if order > 1:
        kernel += a_s * (CUSP[1] / CUSP[0] - BETA[1] / BETA[0]) * (r - 1.0)
    if order > 2:
        raise ValueError("NNNLL GI matching is not available in the legacy CS-kernel implementation")
    return complex(-complex(0.0, np.pi) * CUSP[0] / (2.0 * BETA[0]) * kernel)


@dataclass(frozen=True)
class CSDgamma:
    """Legacy TMDWF Collins-Soper matching corrections.

    The formulas are refactored from the legacy CS-kernel scripts but kept
    numerically compatible with the original implementation.
    """

    mu: float
    order: int

    def alphas(self, mu: float, n_flavors: int = 3) -> float:
        return alphas_nloop(mu, self.order, n_flavors)

    def dgamma_cg(self, p1: float, p2: float, x: float, component: str = "real") -> float:
        if self.order >= 2:
            raise ValueError("NNLO CG matching is not available in the legacy type-2 workflow")
        sum_pqcd = kernel_cg(2.0 * x * p1, self.mu, self.order) - kernel_cg(2.0 * x * p2, self.mu, self.order)
        sum_pqcd += kernel_cg(2.0 * (1.0 - x) * p1, self.mu, self.order) - kernel_cg(2.0 * (1.0 - x) * p2, self.mu, self.order)
        if component == "real":
            return float(-1.0 / np.log(p1 / p2) * sum_pqcd.real)
        if component == "imag":
            return 0.0
        raise ValueError("component must be 'real' or 'imag'")

    def dgamma_gi(self, p1: float, p2: float, x: float, component: str = "real") -> float:
        if self.order < 2:
            sum_pqcd = kernel_gi(2.0 * x * p1, self.mu, self.order) - kernel_gi(2.0 * x * p2, self.mu, self.order)
            sum_pqcd += kernel_gi(2.0 * (1.0 - x) * p1, self.mu, self.order) - kernel_gi(2.0 * (1.0 - x) * p2, self.mu, self.order)
        elif self.order == 2:
            sum_pqcd = kernel_gi(2.0 * x * p1, self.mu, self.order) - kernel_gi(2.0 * x * p2, self.mu, self.order)
            sum_pqcd += kernel_gi(2.0 * (1.0 - x) * p1, self.mu, self.order) - kernel_gi(2.0 * (1.0 - x) * p2, self.mu, self.order)
            sum_pqcd += -0.5 * ((kernel_gi(2.0 * x * p1, self.mu, 1) - 1.0) ** 2 - (kernel_gi(2.0 * x * p2, self.mu, 1) - 1.0) ** 2)
            sum_pqcd += -0.5 * (
                (kernel_gi(2.0 * (1.0 - x) * p1, self.mu, 1) - 1.0) ** 2
                - (kernel_gi(2.0 * (1.0 - x) * p2, self.mu, 1) - 1.0) ** 2
            )
        else:
            raise ValueError("NNNLO GI matching is not available in the legacy type-2 workflow")
        if component == "real":
            return float(-1.0 / np.log(p1 / p2) * sum_pqcd.real)
        if component == "imag":
            return float(-1.0 / np.log(p1 / p2) * sum_pqcd.imag)
        raise ValueError("component must be 'real' or 'imag'")

    def dgamma_cg_rg(self, p1: float, p2: float, x: float, component: str = "real") -> float:
        if self.order < 2:
            sum_p1_x = -2.0 * kernel_cg_k1(2.0 * x * p1, self.mu, self.order) + kernel_cg_k3(2.0 * x * p1, self.mu, self.order) + kernel_cg_eta(2.0 * x * p1, self.mu, self.order)
            sum_p2_x = -2.0 * kernel_cg_k1(2.0 * x * p2, self.mu, self.order) + kernel_cg_k3(2.0 * x * p2, self.mu, self.order) + kernel_cg_eta(2.0 * x * p2, self.mu, self.order)
            sum_p1_xbar = -2.0 * kernel_cg_k1(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_cg_k3(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_cg_eta(2.0 * (1.0 - x) * p1, self.mu, self.order)
            sum_p2_xbar = -2.0 * kernel_cg_k1(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_cg_k3(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_cg_eta(2.0 * (1.0 - x) * p2, self.mu, self.order)
        elif self.order == 2:
            sum_p1_x = -2.0 * kernel_cg_k1(2.0 * x * p1, self.mu, self.order) + kernel_cg_k3(2.0 * x * p1, self.mu, self.order) + kernel_cg_eta(2.0 * x * p1, self.mu, self.order) + kernel_cg(2.0 * x * p1, 2.0 * x * p1, 1)
            sum_p2_x = -2.0 * kernel_cg_k1(2.0 * x * p2, self.mu, self.order) + kernel_cg_k3(2.0 * x * p2, self.mu, self.order) + kernel_cg_eta(2.0 * x * p2, self.mu, self.order) + kernel_cg(2.0 * x * p2, 2.0 * x * p2, 1)
            sum_p1_xbar = -2.0 * kernel_cg_k1(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_cg_k3(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_cg_eta(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_cg(2.0 * (1.0 - x) * p1, 2.0 * (1.0 - x) * p1, 1)
            sum_p2_xbar = -2.0 * kernel_cg_k1(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_cg_k3(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_cg_eta(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_cg(2.0 * (1.0 - x) * p2, 2.0 * (1.0 - x) * p2, 1)
        else:
            raise ValueError("NNNLL CG matching is not available in the legacy type-2 workflow")
        total = sum_p1_x - sum_p2_x + sum_p1_xbar - sum_p2_xbar
        if component == "real":
            return float(-1.0 / np.log(p1 / p2) * total.real)
        if component == "imag":
            return float(-1.0 / np.log(p1 / p2) * total.imag)
        raise ValueError("component must be 'real' or 'imag'")

    def dgamma_gi_rg(self, p1: float, p2: float, x: float, component: str = "real") -> float:
        if self.order < 2:
            sum_p1_x = -2.0 * kernel_gi_k1(2.0 * x * p1, self.mu, self.order) + kernel_gi_k3(2.0 * x * p1, self.mu, self.order) + kernel_gi_eta(2.0 * x * p1, self.mu, self.order)
            sum_p2_x = -2.0 * kernel_gi_k1(2.0 * x * p2, self.mu, self.order) + kernel_gi_k3(2.0 * x * p2, self.mu, self.order) + kernel_gi_eta(2.0 * x * p2, self.mu, self.order)
            sum_p1_xbar = -2.0 * kernel_gi_k1(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_gi_k3(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_gi_eta(2.0 * (1.0 - x) * p1, self.mu, self.order)
            sum_p2_xbar = -2.0 * kernel_gi_k1(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_gi_k3(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_gi_eta(2.0 * (1.0 - x) * p2, self.mu, self.order)
        elif self.order == 2:
            sum_p1_x = -2.0 * kernel_gi_k1(2.0 * x * p1, self.mu, self.order) + kernel_gi_k3(2.0 * x * p1, self.mu, self.order) + kernel_gi_eta(2.0 * x * p1, self.mu, self.order) + kernel_gi(2.0 * x * p1, 2.0 * x * p1, 1)
            sum_p2_x = -2.0 * kernel_gi_k1(2.0 * x * p2, self.mu, self.order) + kernel_gi_k3(2.0 * x * p2, self.mu, self.order) + kernel_gi_eta(2.0 * x * p2, self.mu, self.order) + kernel_gi(2.0 * x * p2, 2.0 * x * p2, 1)
            sum_p1_xbar = -2.0 * kernel_gi_k1(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_gi_k3(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_gi_eta(2.0 * (1.0 - x) * p1, self.mu, self.order) + kernel_gi(2.0 * (1.0 - x) * p1, 2.0 * (1.0 - x) * p1, 1)
            sum_p2_xbar = -2.0 * kernel_gi_k1(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_gi_k3(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_gi_eta(2.0 * (1.0 - x) * p2, self.mu, self.order) + kernel_gi(2.0 * (1.0 - x) * p2, 2.0 * (1.0 - x) * p2, 1)
        else:
            raise ValueError("NNNLL GI matching is not available in the legacy type-2 workflow")
        total = sum_p1_x - sum_p2_x + sum_p1_xbar - sum_p2_xbar
        if component == "real":
            return float(-1.0 / np.log(p1 / p2) * total.real)
        if component == "imag":
            return float(-1.0 / np.log(p1 / p2) * total.imag)
        raise ValueError("component must be 'real' or 'imag'")


def build_cs_dgamma(mu: float, kernel_label: str) -> CSDgamma:
    return CSDgamma(mu=float(mu), order=perturbative_order_from_label(kernel_label))


def evaluate_type2_matching_correction(
    *,
    scheme: str,
    kernel_label: str,
    mu: float,
    p1: float,
    p2: float,
    x: float,
    component: str = "real",
) -> float:
    normalized_scheme = normalize_cs_scheme(scheme)
    correction = build_cs_dgamma(mu, kernel_label)
    if normalized_scheme != "CG":
        raise ValueError(f"unsupported type-2 TMDWF CS-kernel scheme '{scheme}'")
    if uses_rg_improved_matching(kernel_label):
        return correction.dgamma_cg_rg(p1, p2, x, component=component)
    return correction.dgamma_cg(p1, p2, x, component=component)