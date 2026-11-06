# -*- coding: utf-8 -*-
"""
Módulo de Simulação Física do Campo Solar e Receptor Termossolar CSP (Calha Parabólica).
Implementação exata da física SAM (C++ csp_solver_trough_collector_receiver.cpp).

Inclui:
- Receptor evacuado Forristall 1D com perdas térmicas por convecção, radiação e condução nos suportes.
- Propriedades termofísicas O(1) do fluido térmico Therminol VP-1.
- Dimensionamento automatizado de headers, runners, vasos de expansão, SGS e inércias térmicas.
- Balanço térmico transiente com subpassos de 600s para partidas, operação nominal, congelamento e defocus.
"""

import math
from math import pi
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Constantes físicas fundamentais
SIGMA = 5.670374419e-8  # Stefan-Boltzmann [W/(m2-K4)]
GRAV = 9.80665          # Gravidade [m/s2]
C1 = 2.331e-20          # Constante de condução em gás rarefeito [mmHg*cm3/K]
CSI = 2.4e-8            # Diâmetro molecular efetivo de H2 [cm]
P_A = 0.0001            # Pressão de vácuo no HCE [torr]

# Geometria e propriedades do coletor e receptor (SAM Trough Physical Model)
A_APERTURE = 656.0      # Área de abertura refletora por SCA [m2]
W_APERTURE = 6.0        # Largura de abertura refletora do coletor [m] (SAM W_aperture)
L_FOCAL = 2.15          # Distância focal [m] (SAM Ave_Focal_Length)
L_SCA = 115.0           # Comprimento refletor do SCA [m] (SAM L_SCA)
D_SCA = 1.0             # Distância entre SCAs em série [m] (SAM Distance_SCA)

D2 = 0.076              # Diâmetro interno do tubo absorvedor de aço inoxidável [m]
D3 = 0.080              # Diâmetro externo do tubo absorvedor [m]
D4 = 0.115              # Diâmetro interno do tubo de vidro borosilicato [m]
D5 = 0.120              # Diâmetro externo do tubo de vidro [m]

EPS_4 = 0.86            # Emissividade interna do vidro
EPS_5 = 0.86            # Emissividade externa do vidro
ALPHA_ENV_0 = 0.02      # Absortância do envelope de vidro
TAU_ENV_0 = 0.964       # Transmitância do envelope de vidro
ALPHA_ABS_0 = 0.963     # Absortância do tubo absorvedor
DIRT_HCE_0 = 0.98       # Fator de sujidade do HCE
SHADOW_0 = 0.935        # Fator de sombreamento das hastes e folgas
DIRT_SHADOW_0 = DIRT_HCE_0 * SHADOW_0

R_45COND = math.log(D5 / D4) / (2.0 * pi * 1.04)
LOG_D3_D2 = math.log(D3 / D2)
A_CS_TUBE = pi * (D2 ** 2) / 4.0

# Diâmetros nominais de tubulação comercial ASME B36.10M
PIPE_SCHED_M = [
    0.01855, 0.02173, 0.03115, 0.0374, 0.04375, 0.0499, 0.0626,
    0.06880860, 0.08468360, 0.1082040, 0.16146780, 0.2063750, 0.260350,
    0.311150, 0.33975040, 0.39055040, 0.438150, 0.488950, 0.53340,
    0.58420, 0.6350, 0.679450, 0.730250, 0.781050, 0.82864960, 0.87630,
    1.02870, 1.16840, 1.32080, 1.47320, 1.62560, 1.7780, 1.8796, 1.9812,
    2.1844, 2.286
]


def pipe_sched(de: float, select_larger: bool = True) -> float:
    """Retorna o diâmetro padronizado mais próximo da tabela ASME."""
    if select_larger:
        for d_val in PIPE_SCHED_M:
            if d_val >= de:
                return d_val
    else:
        for d_val in reversed(PIPE_SCHED_M):
            if d_val <= de:
                return d_val
    return de


# Propriedades termofísicas O(1) do Therminol VP-1 (SAM htf_props.cpp)
def vp1_cp_k(t_k: float) -> float:
    """Calor específico do Therminol VP-1 [kJ/kg-K] para T em Kelvin."""
    t_c = t_k - 273.15
    return 1.509 + 0.002496 * t_c + 0.0000007888 * t_c * t_c


def vp1_cp_ave_k(t_cold_k: float, t_hot_k: float) -> float:
    """Média de 5 pontos do Cp do Therminol VP-1 [kJ/kg-K]."""
    dt = (t_hot_k - t_cold_k) / 5.0
    t_i = t_cold_k + 0.5 * dt
    s = 0.0
    for _ in range(5):
        t_c = t_i - 273.15
        s += 1.509 + 0.002496 * t_c + 0.0000007888 * t_c * t_c
        t_i += dt
    return s * 0.2


def vp1_dens_k(t_k: float) -> float:
    """Massa específica do Therminol VP-1 [kg/m3] para T em Kelvin."""
    t_c = t_k - 273.15
    d = 1074.0 - 0.6367 * t_c - 0.0007762 * t_c * t_c
    return d if d > 400.0 else 400.0


def vp1_visc_k(t_k: float) -> float:
    """Viscosidade dinâmica do Therminol VP-1 [Pa-s] para T em Kelvin."""
    t_c = max(t_k - 273.15, 20.0)
    return 0.001 * (10.0 ** 0.8703) * (t_c ** (0.2877 - 0.3638 * math.log10(t_c)))


def vp1_cond_k(t_k: float) -> float:
    """Condutividade térmica do Therminol VP-1 [W/m-K] para T em Kelvin."""
    t_c = t_k - 273.15
    k = 0.1381 - 0.00008708 * t_c - 0.0000001729 * t_c * t_c
    return k if k > 0.001 else 0.001


def air_visc_k(t_k: float) -> float:
    v = 0.0000010765 + 7.15173e-8 * t_k - 5.03525e-11 * t_k * t_k + 2.02799e-14 * (t_k ** 3)
    return v if v > 1e-6 else 1e-6


def air_dens_k(t_k: float, p_pa: float) -> float:
    return p_pa / (287.058 * t_k)


def air_cp_k(t_k: float) -> float:
    return (1.03749 - 0.000305497 * t_k + 7.49335e-7 * t_k * t_k - 3.39363e-10 * (t_k ** 3)) * 1000.0


def air_cond_k(t_k: float) -> float:
    k = 0.00145453 + 0.0000872152 * t_k - 2.20614e-8 * t_k * t_k
    return k if k > 1e-4 else 1e-4


def h2_cp_k(t_k: float) -> float:
    val = (
        -45.4022 + 0.690156 * t_k - 0.00327354 * (t_k ** 2)
        + 0.00000817326 * (t_k ** 3) - 1.13234e-8 * (t_k ** 4)
        + 8.24995e-12 * (t_k ** 5) - 2.46804e-15 * (t_k ** 6)
    )
    return min(max(val, 11.3), 14.7) * 1000.0


def h2_cv_k(t_k: float) -> float:
    val = (
        -49.5264 + 0.690156 * t_k - 0.00327354 * (t_k ** 2)
        + 0.00000817326 * (t_k ** 3) - 1.13234e-8 * (t_k ** 4)
        + 8.24995e-12 * (t_k ** 5) - 2.46804e-15 * (t_k ** 6)
    )
    return min(max(val, 7.20), 10.60) * 1000.0


def h2_cond_k(t_k: float) -> float:
    val = 0.0302888 + 0.00053634 * t_k - 1.59604e-7 * (t_k ** 2)
    return val if val > 0.01 else 0.01


EPS_T_ARR = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]
EPS_V_ARR = [0.061, 0.062, 0.064, 0.0665, 0.07, 0.0745, 0.08, 0.0865, 0.094, 0.1025, 0.112]


def interp_eps3_fast(t_c: float) -> float:
    """Interpolação linear da emissividade do revestimento seletivo (Luz Cermet)."""
    if t_c <= 0.0:
        return 0.061
    if t_c >= 500.0:
        return 0.112
    idx = int(t_c // 50.0)
    f = (t_c - EPS_T_ARR[idx]) * 0.02
    return EPS_V_ARR[idx] + f * (EPS_V_ARR[idx + 1] - EPS_V_ARR[idx])


def fq_34conv_sam(t_3: float, t_4: float) -> float:
    t_34 = 0.5 * (t_3 + t_4)
    lam = C1 * t_34 / (P_A * CSI * CSI)
    cp_34 = h2_cp_k(t_34)
    cv_34 = h2_cv_k(t_34)
    gamma_gas = cp_34 / cv_34
    b_val = (9.0 * gamma_gas - 5.0) / (2.0 * gamma_gas + 2.0)
    k_34 = h2_cond_k(t_34)
    h_conv = k_34 / (0.5 * D3 * math.log(D4 / D3) + b_val * (lam * 0.01) * (D3 / D4 + 1.0))
    return pi * D3 * h_conv * (t_3 - t_4)


def fq_34rad_sam(t_3: float, t_4: float) -> float:
    eps_3 = interp_eps3_fast(t_3 - 273.15)
    h_rad = SIGMA * (t_3 * t_3 + t_4 * t_4) * (t_3 + t_4) / (1.0 / eps_3 + (D3 / D4) * (1.0 / EPS_4 - 1.0))
    return pi * D3 * h_rad * (t_3 - t_4)


def fq_56conv_sam(t_5: float, t_6: float, p_6_pa: float, v_6: float) -> float:
    t_56 = 0.5 * (t_5 + t_6)
    if v_6 <= 0.1:
        mu_56 = air_visc_k(t_56)
        rho_56 = air_dens_k(t_56, p_6_pa)
        cp_56 = air_cp_k(t_56)
        k_56 = air_cond_k(t_56)
        nu_56 = mu_56 / rho_56
        alpha_56 = k_56 / (cp_56 * rho_56)
        beta_56 = 1.0 / t_56
        ra_d5 = GRAV * beta_56 * abs(t_5 - t_6) * (D5 ** 3) / (alpha_56 * nu_56)
        pr_56 = nu_56 / alpha_56
        nu_bar = (0.60 + (0.387 * (ra_d5 ** 0.1667)) / ((1.0 + (0.559 / pr_56) ** 0.5625) ** 0.2963)) ** 2
        h_6 = nu_bar * k_56 / D5
        return h_6 * pi * D5 * (t_5 - t_6)
    else:
        mu_5 = air_visc_k(t_5)
        mu_6 = air_visc_k(t_6)
        rho_5 = air_dens_k(t_5, p_6_pa)
        rho_6 = air_dens_k(t_6, p_6_pa)
        k_5 = air_cond_k(t_5)
        k_6 = air_cond_k(t_6)
        cp_5 = air_cp_k(t_5)
        cp_6 = air_cp_k(t_6)
        alpha_5 = k_5 / (cp_5 * rho_5)
        alpha_6 = k_6 / (cp_6 * rho_6)
        nu_5 = mu_5 / rho_5
        nu_6 = mu_6 / rho_6
        pr_5 = nu_5 / alpha_5
        pr_6 = nu_6 / alpha_6
        re_d5 = v_6 * D5 * rho_6 / mu_6
        n_val = 0.37 if pr_6 <= 10.0 else 0.36
        if re_d5 < 40.0:
            c_val, m_val = 0.75, 0.4
        elif re_d5 < 1e3:
            c_val, m_val = 0.51, 0.5
        elif re_d5 < 2e5:
            c_val, m_val = 0.26, 0.6
        else:
            c_val, m_val = 0.076, 0.7
        nus_6 = c_val * (re_d5 ** m_val) * (pr_6 ** n_val) * ((pr_6 / pr_5) ** 0.25)
        h_6 = nus_6 * k_6 / D5
        return h_6 * pi * D5 * (t_5 - t_6)


def fq_cond_bracket_sam(t_3: float, t_6: float, p_6_pa: float, v_6: float) -> float:
    p_brac, d_brac, a_cs_brac, k_brac_steel = 0.2032, 0.0508, 0.00016129, 48.0
    t_base = t_3 - 10.0
    t_brac = 0.5 * (t_base + t_6)
    t_brac6 = 0.5 * (t_brac + t_6)
    if v_6 <= 0.1:
        mu_brac6 = air_visc_k(t_brac6)
        rho_brac6 = air_dens_k(t_brac6, p_6_pa)
        cp_brac6 = air_cp_k(t_brac6)
        k_brac6 = air_cond_k(t_brac6)
        nu_brac6 = mu_brac6 / rho_brac6
        alpha_brac6 = k_brac6 / (cp_brac6 * rho_brac6)
        beta_brac6 = 1.0 / t_brac6
        ra_dbrac = GRAV * beta_brac6 * abs(t_brac - t_6) * (d_brac ** 3) / (alpha_brac6 * nu_brac6)
        pr_brac6 = nu_brac6 / alpha_brac6
        nu_bar = (0.60 + (0.387 * (ra_dbrac ** 0.1667)) / ((1.0 + (0.559 / pr_brac6) ** 0.5625) ** 0.2963)) ** 2
        h_brac6 = nu_bar * k_brac6 / d_brac
    else:
        mu_6 = air_visc_k(t_6)
        rho_6 = air_dens_k(t_6, p_6_pa)
        rho_brac = air_dens_k(t_brac, p_6_pa)
        k_6 = air_cond_k(t_6)
        k_brac6 = air_cond_k(t_brac6)
        cp_6 = air_cp_k(t_6)
        nu_6 = mu_6 / rho_6
        alpha_6 = k_6 / (cp_6 * rho_6)
        re_dbrac = v_6 * d_brac / nu_6
        pr_brac = (air_visc_k(t_brac) / rho_brac) / (air_cond_k(t_brac) / (air_cp_k(t_brac) * rho_brac))
        pr_6 = nu_6 / alpha_6
        n_val = 0.37 if pr_6 <= 10.0 else 0.36
        if re_dbrac < 40.0:
            c_val, m_val = 0.75, 0.4
        elif re_dbrac < 1e3:
            c_val, m_val = 0.51, 0.5
        elif re_dbrac < 2e5:
            c_val, m_val = 0.26, 0.6
        else:
            c_val, m_val = 0.076, 0.7
        nu_bar = c_val * (re_dbrac ** m_val) * (pr_6 ** n_val) * ((pr_6 / pr_brac) ** 0.25)
        h_brac6 = nu_bar * k_brac6 / d_brac
    return math.sqrt(h_brac6 * p_brac * k_brac_steel * a_cs_brac) * (t_base - t_6) / 4.06


def ft_2_sam(q_12conv: float, t_1: float, t_2g: float, v_1: float) -> float:
    if v_1 <= 0.1:
        return t_1
    t_2g = max(t_2g, 285.15)
    mu_1 = vp1_visc_k(t_1)
    mu_2 = vp1_visc_k(t_2g)
    cp_1 = vp1_cp_k(t_1) * 1000.0
    cp_2 = vp1_cp_k(t_2g) * 1000.0
    k_1 = vp1_cond_k(t_1)
    k_2 = vp1_cond_k(t_2g)
    rho_1 = vp1_dens_k(t_1)
    pr_1 = (cp_1 * mu_1) / k_1
    pr_2 = (cp_2 * mu_2) / k_2
    re_d2 = (rho_1 * D2 * v_1) / mu_1
    if re_d2 <= 2300.0:
        nu_d2 = 4.36
    else:
        f = (1.82 * math.log10(re_d2) - 1.64) ** (-2)
        nu_d2 = (f / 8.0) * (re_d2 - 1000.0) * pr_1 / (1.0 + 12.7 * math.sqrt(f / 8.0) * ((pr_1 ** 0.6667) - 1.0)) * ((pr_1 / pr_2) ** 0.11)
    h_1 = nu_d2 * k_1 / D2
    return t_1 + q_12conv / (h_1 * D2 * pi)


def evac_receiver_sam(
    t_1_in: float,
    m_dot: float,
    t_amb: float,
    t_sky: float,
    v_6: float,
    p_6_pa: float,
    q_i: float,
    col_opt_eff_sca: float,
) -> tuple:
    """
    Solucionador exato 1D para o receptor Forristall EvacReceiver (SAM).
    Retorna (q_heatloss [W/m], q_12conv [W/m], q_34tot [W/m], c_1ave [kJ/kg-K], rho_1ave [kg/m3], q_3reflect [W/m]).
    """
    colopteff_tot = col_opt_eff_sca * DIRT_SHADOW_0
    q_3solabs = q_i * colopteff_tot * TAU_ENV_0 * ALPHA_ABS_0
    q_5solabs = q_i * colopteff_tot * ALPHA_ENV_0
    q_3reflect = q_i * colopteff_tot * TAU_ENV_0 * (1.0 - ALPHA_ABS_0)

    cp_1 = vp1_cp_k(t_1_in) * 1000.0
    t_3 = t_1_in + 5.0
    t_4 = t_amb + 0.1 * (t_3 - t_amb)

    for _ in range(2):
        for _ in range(2):
            q_34tot = fq_34conv_sam(t_3, t_4) + fq_34rad_sam(t_3, t_4)
            q_45cond = q_34tot + q_5solabs
            t_5 = t_4 - q_45cond * R_45COND
            q_56 = fq_56conv_sam(t_5, t_amb, p_6_pa, v_6)
            q_5out = q_56 + EPS_5 * SIGMA * ((t_5 ** 4) - (t_sky ** 4))
            res = q_5out - q_45cond
            if abs(res) < 0.05:
                break
            dq34_dt4 = -q_34tot / (t_3 - t_4) if abs(t_3 - t_4) > 1e-4 else -5.0
            h56_pi_d5 = q_56 / (t_5 - t_amb) if abs(t_5 - t_amb) > 1e-4 else 5.0
            dq5_dt5 = h56_pi_d5 + 4.0 * EPS_5 * SIGMA * (t_5 ** 3)
            dres = dq5_dt5 * (1.0 - R_45COND * dq34_dt4) - dq34_dt4
            if abs(dres) > 1e-6:
                t_4 = max(t_sky, min(max(t_3, t_amb), t_4 - res / dres))

        q_34tot = fq_34conv_sam(t_3, t_4) + fq_34rad_sam(t_3, t_4)
        q_cond_bracket = fq_cond_bracket_sam(t_3, t_amb, p_6_pa, v_6)
        q_12conv = q_3solabs - (q_34tot + q_cond_bracket)
        q_in_w = q_12conv * L_SCA

        t_1_out = max(t_sky, q_in_w / (m_dot * cp_1) + t_1_in)
        t_1_ave = 0.5 * (t_1_in + t_1_out)
        cp_1 = vp1_cp_k(t_1_ave) * 1000.0
        t_1_out = max(t_sky, q_in_w / (m_dot * cp_1) + t_1_in)
        t_1_ave = 0.5 * (t_1_in + t_1_out)

        rho_1ave = vp1_dens_k(t_1_ave)
        v_1 = m_dot / (rho_1ave * A_CS_TUBE)
        t_2 = max(10.0, ft_2_sam(q_12conv, t_1_ave, t_1_ave, v_1))
        k_23 = 0.013 * (0.5 * (t_2 + t_3) - 273.15) + 15.2
        t_3 = t_2 + q_12conv * LOG_D3_D2 / (2.0 * pi * k_23)

    q_heatloss = q_34tot + q_cond_bracket
    return q_heatloss, q_12conv, q_34tot, cp_1 * 1e-3, rho_1ave, q_3reflect


def m_dot_runner_sam(m_dot_field: float, nfieldsec: int, irnr: int) -> float:
    nrnrsec = int(math.floor(nfieldsec / 4.0)) + 1
    irnr_onedir = (2 * nrnrsec - irnr - 1) if (irnr > nrnrsec - 1) else irnr
    m_dot_rnr_0 = m_dot_field / 2.0
    m_dot_rnr_1 = m_dot_rnr_0 * (1.0 - float(nfieldsec % 4) / float(nfieldsec))
    if irnr_onedir == 0:
        m_dot_rnr = m_dot_rnr_0
    elif irnr_onedir == 1:
        m_dot_rnr = m_dot_rnr_1
    else:
        m_dot_rnr = m_dot_rnr_1 - (irnr_onedir - 1) * m_dot_field / float(nfieldsec) * 2.0
    return max(m_dot_rnr, 0.0)


def m_dot_header_sam(m_dot_field: float, nfieldsec: int, n_loops_field: int, ihdr: int) -> float:
    nhdrsec = int(math.ceil(float(n_loops_field) / float(nfieldsec * 2)))
    ihdr_onedir = (2 * nhdrsec - ihdr - 1) if (ihdr > nhdrsec - 1) else ihdr
    m_dot_oneloop = m_dot_field / float(n_loops_field)
    return m_dot_field / float(nfieldsec) - ihdr_onedir * 2.0 * m_dot_oneloop


def dimensionar_campo_tubulacoes_sam(
    n_loops_field: int = 91,
    p_ref_mw: float = 55.5,
    eta_ref: float = 0.356,
    i_bn_des: float = 950.0,
    nfsec: int = 2,
    v_hdr_cold_max: float = 3.0,
    v_hdr_cold_min: float = 2.0,
    v_hdr_hot_max: float = 3.0,
    v_hdr_hot_min: float = 2.0,
    n_max_hdr_diams: int = 10,
    row_distance: float = 15.0,
    l_rnr_pb: float = 25.0,
    l_rnr_per_xpan: float = 70.0,
    l_xpan_hdr: float = 20.0,
    l_xpan_rnr: float = 20.0,
    min_rnr_xpans: int = 1,
    northsouth_field_sep: float = 20.0,
    n_hdr_per_xpan: int = 2,
    offset_xpan_hdr: int = 1,
    mc_bal_hot_per_mw: float = 0.2,
    mc_bal_cold_per_mw: float = 0.2,
    mc_bal_sca_wht_km: float = 4.5,
    pipe_hl_coef: float = 0.45,
) -> dict:
    """Replicador fiel do dimensionamento de tubulações e solar multiple (SAM init_fieldgeom)."""
    n_sca = 8
    single_loop_ap = n_sca * A_APERTURE  # 5248.0 m2
    ap_tot = n_loops_field * single_loop_ap

    t_in_des_k = 293.0 + 273.15
    t_out_des_k = 391.0 + 273.15

    # Eficiência óptica nominal de projeto SAM (Shadowing * Tracking * Geom * Refl * Dirt_mirror * Dirt_HCE * Error * Alpha_abs * Tau_env)
    opteff_des = (
        0.935 * 0.988 * 0.952 * 0.93 * 0.97 * 0.98 * 1.0 * 0.963 * 0.964
    )  # 0.72166487
    loss_tot_loop = 190.0 * L_SCA * n_sca  # 174,800 W por loop
    q_loop_ideal = single_loop_ap * i_bn_des
    q_loop_actual = q_loop_ideal * opteff_des - loss_tot_loop
    tot_loop_conv_eff_des = q_loop_actual / q_loop_ideal

    q_pb_design_w = (p_ref_mw * 1e6) / eta_ref
    req_ap_sm1 = q_pb_design_w / (i_bn_des * tot_loop_conv_eff_des)
    solar_mult = ap_tot / req_ap_sm1

    c_htf_ave = vp1_cp_ave_k(t_in_des_k, t_out_des_k) * 1000.0
    m_dot_design = (ap_tot * i_bn_des * opteff_des - loss_tot_loop * float(n_loops_field)) / (
        c_htf_ave * (t_out_des_k - t_in_des_k)
    )
    m_dot_field_min = 1.0 * n_loops_field
    m_dot_field_max = 12.0 * n_loops_field
    m_dot_design = min(max(m_dot_design, m_dot_field_min), m_dot_field_max)
    m_dot_loop_des = m_dot_design / float(n_loops_field)
    q_design_actual_w = m_dot_design * c_htf_ave * (t_out_des_k - t_in_des_k)

    mc_bal_hot = mc_bal_hot_per_mw * 3.6 * q_design_actual_w
    mc_bal_cold = mc_bal_cold_per_mw * 3.6 * q_design_actual_w
    mc_bal_sca = mc_bal_sca_wht_km * 3.6e3

    rho_cold = vp1_dens_k(t_in_des_k)
    rho_hot = vp1_dens_k(t_out_des_k)
    rho_ave = vp1_dens_k(0.5 * (t_in_des_k + t_out_des_k))

    nhdrsec = int(math.ceil(float(n_loops_field) / float(nfsec * 2)))
    nrunsec = int(math.floor(float(nfsec) / 4.0)) + 1

    d_runner = [0.0] * (2 * nrunsec)
    m_dot_rnr = [0.0] * (2 * nrunsec)
    v_cold_avg = 0.5 * (v_hdr_cold_max + v_hdr_cold_min)
    v_hot_avg = 0.5 * (v_hdr_hot_max + v_hdr_hot_min)
    m_dot_hdrs = m_dot_design / float(nfsec)
    m_dot_2loops = m_dot_hdrs / float(nhdrsec)

    m_dot_rnr[0] = m_dot_design / 2.0
    m_dot_rnr[2 * nrunsec - 1] = m_dot_rnr[0]
    d_runner[0] = pipe_sched(math.sqrt(4.0 * m_dot_rnr[0] / (rho_cold * v_cold_avg * pi)), True)
    d_runner[2 * nrunsec - 1] = pipe_sched(math.sqrt(4.0 * m_dot_rnr[2 * nrunsec - 1] / (rho_hot * v_hot_avg * pi)), True)
    for i in range(1, nrunsec):
        if i == 1:
            m_dot_rnr[i] = m_dot_rnr[i - 1] * (1.0 - float(nfsec % 4) / float(nfsec))
        else:
            m_dot_rnr[i] = max(m_dot_rnr[i - 1] - m_dot_hdrs * 2.0, 0.0)
        m_dot_rnr[2 * nrunsec - i - 1] = m_dot_rnr[i]
        d_runner[i] = pipe_sched(math.sqrt(4.0 * m_dot_rnr[i] / (rho_cold * v_cold_avg * pi)), True)
        d_runner[2 * nrunsec - i - 1] = pipe_sched(math.sqrt(4.0 * m_dot_rnr[2 * nrunsec - i - 1] / (rho_hot * v_hot_avg * pi)), True)

    d_hdr = [0.0] * (2 * nhdrsec)
    nd = 0
    m_dot_enter = 0.0
    for i in range(nhdrsec):
        if i == 0:
            m_dot_enter = m_dot_hdrs
            d_hdr[i] = pipe_sched(math.sqrt(4.0 * m_dot_enter / (rho_cold * v_hdr_cold_max * pi)), True)
            v_enter = 4.0 * m_dot_enter / (rho_cold * pi * d_hdr[i] * d_hdr[i])
            if v_enter < v_hdr_cold_min:
                d_hdr_next = pipe_sched(math.sqrt(4.0 * m_dot_enter / (rho_cold * v_hdr_cold_max * pi)), False)
                v_enter_next = 4.0 * m_dot_enter / (rho_cold * pi * d_hdr_next * d_hdr_next)
                if v_enter_next - v_hdr_cold_max <= v_hdr_cold_min - v_enter:
                    d_hdr[i] = d_hdr_next
            nd += 1
        elif nd < n_max_hdr_diams:
            m_dot_enter -= m_dot_2loops
            v_enter = 4.0 * m_dot_enter / (rho_cold * pi * d_hdr[i - 1] * d_hdr[i - 1])
            if v_enter < v_hdr_cold_min:
                d_hdr_next = pipe_sched(math.sqrt(4.0 * m_dot_enter / (rho_cold * v_hdr_cold_max * pi)), True)
                v_enter_next = 4.0 * m_dot_enter / (rho_cold * pi * d_hdr_next * d_hdr_next)
                if v_enter_next < v_hdr_cold_min:
                    d_hdr_next2 = pipe_sched(math.sqrt(4.0 * m_dot_enter / (rho_cold * v_hdr_cold_max * pi)), False)
                    v_enter_next2 = 4.0 * m_dot_enter / (rho_cold * pi * d_hdr_next2 * d_hdr_next2)
                    if v_hdr_cold_min - v_enter_next < v_enter_next2 - v_hdr_cold_max:
                        d_hdr[i] = d_hdr_next
                    else:
                        d_hdr[i] = d_hdr_next2
                else:
                    d_hdr[i] = d_hdr_next
                if (d_hdr[i - 1] - d_hdr[i]) > 0.001:
                    nd += 1
            else:
                d_hdr[i] = d_hdr[i - 1]
        else:
            m_dot_enter -= m_dot_2loops
            d_hdr[i] = d_hdr[i - 1]

    nd = 0
    m_dot_leave = 0.0
    for i in range(nhdrsec, 2 * nhdrsec):
        if i == nhdrsec:
            m_dot_leave = m_dot_2loops
            d_hdr[i] = pipe_sched(math.sqrt(4.0 * m_dot_leave / (rho_hot * v_hdr_hot_min * pi)), False)
            v_leave = 4.0 * m_dot_leave / (rho_hot * pi * d_hdr[i] * d_hdr[i])
            if v_leave > v_hdr_hot_max:
                d_hdr_next = pipe_sched(math.sqrt(4.0 * m_dot_leave / (rho_hot * v_hdr_hot_min * pi)), True)
                v_leave_next = 4.0 * m_dot_leave / (rho_hot * pi * d_hdr_next * d_hdr_next)
                if v_hdr_hot_min - v_leave_next < v_leave - v_hdr_hot_max:
                    d_hdr[i] = d_hdr_next
            nd += 1
        elif nd < n_max_hdr_diams:
            m_dot_leave += m_dot_2loops
            v_leave = 4.0 * m_dot_leave / (rho_hot * pi * d_hdr[i - 1] * d_hdr[i - 1])
            if v_leave > v_hdr_hot_max:
                d_hdr_next = pipe_sched(math.sqrt(4.0 * m_dot_leave / (rho_hot * v_hdr_hot_min * pi)), False)
                v_leave_next = 4.0 * m_dot_leave / (rho_hot * pi * d_hdr_next * d_hdr_next)
                if v_leave_next > v_hdr_hot_max:
                    d_hdr_next2 = pipe_sched(math.sqrt(4.0 * m_dot_leave / (rho_hot * v_hdr_hot_min * pi)), True)
                    v_leave_next2 = 4.0 * m_dot_leave / (rho_hot * pi * d_hdr_next2 * d_hdr_next2)
                    if v_leave_next - v_hdr_hot_max <= v_hdr_hot_min - v_leave_next2:
                        d_hdr[i] = d_hdr_next
                    else:
                        d_hdr[i] = d_hdr_next2
                else:
                    d_hdr[i] = d_hdr_next
                if (d_hdr[i] - d_hdr[i - 1]) > 0.001:
                    nd += 1
            else:
                d_hdr[i] = d_hdr[i - 1]
        else:
            m_dot_leave += m_dot_2loops
            d_hdr[i] = d_hdr[i - 1]

    l_hdr = [2.0 * row_distance] * (2 * nhdrsec)
    for i in range(nhdrsec):
        if (i - offset_xpan_hdr) % n_hdr_per_xpan == 0:
            l_hdr[i] += l_xpan_hdr
            l_hdr[2 * nhdrsec - 1 - i] += l_xpan_hdr

    l_runner = [0.0] * (2 * nrunsec)
    x1 = 2.0 if ((nfsec // 2) % 2 == 1) else 1.0
    l_runner[0] = l_rnr_pb
    l_runner[2 * nrunsec - 1] = l_rnr_pb
    if nrunsec > 1:
        for i in range(1, nrunsec):
            l_rnr_lin = x1 * (northsouth_field_sep + (L_SCA + D_SCA) * float(n_sca) / 2.0)
            n_xpan = max(min_rnr_xpans, int(math.floor(l_rnr_lin / l_rnr_per_xpan + 0.5)))
            l_runner[i] = l_rnr_lin + l_xpan_rnr * n_xpan
            l_runner[2 * nrunsec - i - 1] = l_runner[i]
            x1 = 2.0

    v_hot = 0.0
    v_cold = 0.0
    for i in range(nhdrsec):
        v_cold += (d_hdr[i] ** 2) * 0.25 * pi * l_hdr[i] * float(nfsec)
    for i in range(nhdrsec, 2 * nhdrsec):
        v_hot += (d_hdr[i] ** 2) * 0.25 * pi * l_hdr[i] * float(nfsec)
    for i in range(nrunsec):
        v_cold += (d_runner[i] ** 2) * 0.25 * pi * l_runner[i] * 2.0
    for i in range(nrunsec, 2 * nrunsec):
        v_hot += (d_runner[i] ** 2) * 0.25 * pi * l_runner[i] * 2.0

    l_line_sgs = [0.0, 0.0, 90.0, 100.0, 120.0, 80.0, 120.0, 80.0]
    vel_max_sgs = 1.85
    v_dotsf = m_dot_design / rho_ave
    v_dotpb = (m_dot_design / solar_mult) / rho_ave
    v_dot_sgs = [v_dotsf, 0.5 * v_dotsf, 0.5 * v_dotsf, v_dotsf, v_dotsf, v_dotpb, v_dotpb, v_dotpb]
    v_sgs = 0.0
    for l_s, vd_s in zip(l_line_sgs, v_dot_sgs):
        d_s = pipe_sched(math.sqrt(4.0 * vd_s / (vel_max_sgs * pi)), True)
        v_sgs += (d_s ** 2) * 0.25 * pi * l_s

    v_hot += 0.5 * v_sgs
    v_cold += 0.5 * v_sgs

    rnr_flow_frac = [m_dot_runner_sam(1.0, nfsec, i) for i in range(2 * nrunsec)]
    hdr_flow_frac = [m_dot_header_sam(1.0, nfsec, n_loops_field, i) for i in range(2 * nhdrsec)]

    a_hdr = sum(d * pi * l for d, l in zip(d_hdr, l_hdr)) * float(nfsec)
    a_rnr = sum(d * pi * l for d, l in zip(d_runner, l_runner)) * 2.0
    a_piping_tot = a_hdr + a_rnr

    return {
        "n_loops": n_loops_field,
        "a_piping_tot": a_piping_tot,
        "n_sca": n_sca,
        "ap_tot": ap_tot,
        "opteff_des": opteff_des,
        "i_bn_des": i_bn_des,
        "m_dot_design": m_dot_design,
        "m_dot_loop_des": m_dot_loop_des,
        "q_design_actual_w": q_design_actual_w,
        "mc_bal_hot": mc_bal_hot,
        "mc_bal_cold": mc_bal_cold,
        "mc_bal_sca": mc_bal_sca,
        "v_hot": v_hot,
        "v_cold": v_cold,
        "nfsec": nfsec,
        "nhdrsec": nhdrsec,
        "nrunsec": nrunsec,
        "d_hdr": d_hdr,
        "l_hdr": l_hdr,
        "d_runner": d_runner,
        "l_runner": l_runner,
        "rnr_flow_frac": rnr_flow_frac,
        "hdr_flow_frac": hdr_flow_frac,
        "row_distance": row_distance,
        "pipe_hl_coef": pipe_hl_coef,
        "t_in_des_k": t_in_des_k,
        "t_out_des_k": t_out_des_k,
        "c_htf_ave": c_htf_ave,
        "t_startup_k": 325.34 + 273.15,
        "t_shutdown_k": 325.34 + 273.15,
        "t_fp_k": 150.0 + 273.15,
        "m_dot_htfmin": 1.0,
        "m_dot_htfmax": 12.0,
        "solar_mult": solar_mult,
        "t_cold_in_op_k": 295.5 + 273.15,
        "rec_su_delay_s": 0.2 * 3600.0,
        "rec_qf_delay": 0.25,
    }


UA_INTC_1M = 0.45 * pi * 0.0635 * (1.0 + 2.0 * 0.0194) * 1.0


def step_interconnect_temp(t_in_k: float, t_db_k: float, m_dot_loop: float, n_1m_segs: int) -> tuple:
    t_cur = t_in_k
    hl_tot = 0.0
    for _ in range(n_1m_segs):
        hl = UA_INTC_1M * (t_cur - t_db_k)
        cp_j = vp1_cp_k(t_cur) * 1000.0
        t_cur -= hl / (m_dot_loop * cp_j)
        hl_tot += hl
    return t_cur, hl_tot


def loop_energy_balance_t_int(
    cfg: dict,
    state_last: dict,
    t_htf_cold_in_k: float,
    m_dot_htf_loop: float,
    t_db_k: float,
    t_sky_k: float,
    v_wind: float,
    p_amb_pa: float,
    q_sca_arr: np.ndarray,
    col_opt_eff_arr: np.ndarray,
    dt_step: float,
) -> dict:
    """Balanço térmico de loop transiente com integração analítica de inércia."""
    n_loops_f = float(cfg["n_loops"])
    n_sca = cfg["n_sca"]
    m_dot_tot = m_dot_htf_loop * n_loops_f

    t_sys_c_last = state_last["t_sys_c_t_end_last"]
    t_sys_h_last = state_last["t_sys_h_t_end_last"]
    t_sca_last = state_last["t_htf_out_t_end_last"]

    rho_hdr_cold = vp1_dens_k(t_sys_c_last)
    rho_hdr_hot = vp1_dens_k(t_sys_h_last)
    c_hdr_cold_last = vp1_cp_k(t_sys_c_last) * 1000.0

    tau_c_inv = m_dot_tot / (cfg["v_cold"] * rho_hdr_cold + cfg["mc_bal_cold"] / c_hdr_cold_last)
    exp_c = math.exp(-tau_c_inv * dt_step)
    t_sys_c_t_end = (t_sys_c_last - t_htf_cold_in_k) * exp_c + t_htf_cold_in_k
    t_sys_c_t_int = t_htf_cold_in_k - ((t_sys_c_last - t_htf_cold_in_k) / (tau_c_inv * dt_step)) * (exp_c - 1.0)

    cp_sys_c_int = vp1_cp_k(t_sys_c_t_int) * 1000.0
    nrunsec = cfg["nrunsec"]
    nhdrsec = cfg["nhdrsec"]
    nfsec = cfg["nfsec"]
    d_runner = cfg["d_runner"]
    l_runner = cfg["l_runner"]
    d_hdr = cfg["d_hdr"]
    rnr_frac = cfg["rnr_flow_frac"]
    hdr_frac = cfg["hdr_flow_frac"]
    pipe_u = cfg["pipe_hl_coef"]
    row_dist = cfg["row_distance"]

    t_rnr_c = t_sys_c_t_int
    rnr_hl_cold = 0.0
    for i in range(nrunsec):
        if i != 0:
            t_rnr_c -= rnr_hl_cold / (m_dot_tot * rnr_frac[i - 1] * cp_sys_c_int)
        rnr_hl_cold = l_runner[i] * pi * d_runner[i] * pipe_u * (t_rnr_c - t_db_k)

    t_hdr_c = t_rnr_c - rnr_hl_cold / (m_dot_tot * rnr_frac[nrunsec - 1] * cp_sys_c_int)
    hdr_hl_cold = 0.0
    for i in range(nhdrsec):
        if i != 0:
            t_hdr_c -= hdr_hl_cold / (m_dot_tot * hdr_frac[i - 1] * cp_sys_c_int)
        hdr_hl_cold = row_dist * d_hdr[i] * pi * pipe_u * (t_hdr_c - t_db_k)

    q_dot_loss_hr_cold = hdr_hl_cold + rnr_hl_cold
    t_loop_in = t_hdr_c - hdr_hl_cold / (m_dot_tot * hdr_frac[nhdrsec - 1] * cp_sys_c_int)
    e_hr_cold_mj = (cfg["v_cold"] * rho_hdr_cold * cp_sys_c_int + cfg["mc_bal_cold"]) * (t_sys_c_t_end - t_sys_c_last) * 1e-6

    t_in_sca_0, _ = step_interconnect_temp(t_loop_in, t_db_k, m_dot_htf_loop, 3)

    t_htf_in_t_int = [0.0] * n_sca
    t_htf_out_t_end = [0.0] * n_sca
    t_htf_out_t_int = [0.0] * n_sca
    t_htf_in_t_int[0] = t_in_sca_0

    q_sca_loss_sum_w = 0.0
    q_sca_abs_sum_w = 0.0
    q_xover_loss_sum_w = 0.0
    e_sca_sum_mj = 0.0
    mc_bal_sca_l = cfg["mc_bal_sca"] * L_SCA

    for i in range(n_sca):
        q_loss_wm, q_abs_wm, _, c_htf_kj, rho_htf_i, _ = evac_receiver_sam(
            t_htf_in_t_int[i], m_dot_htf_loop, t_db_k, t_sky_k, v_wind, p_amb_pa,
            q_sca_arr[i], col_opt_eff_arr[i]
        )
        q_abs_sca_w = q_abs_wm * L_SCA
        q_loss_sca_w = q_loss_wm * L_SCA
        c_htf_i = c_htf_kj * 1000.0

        m_node = rho_htf_i * A_CS_TUBE * L_SCA
        mc_tot_sca = m_node * c_htf_i + mc_bal_sca_l
        tau_sca_inv = (m_dot_htf_loop * c_htf_i) / mc_tot_sca
        exp_sca = math.exp(-tau_sca_inv * dt_step)
        dt_ss = q_abs_sca_w / (m_dot_htf_loop * c_htf_i)
        diff_t_sca = t_sca_last[i] - t_htf_in_t_int[i] - dt_ss

        t_htf_out_t_end[i] = t_htf_in_t_int[i] + dt_ss + diff_t_sca * exp_sca
        t_htf_out_t_int[i] = t_htf_in_t_int[i] + dt_ss - (diff_t_sca / (tau_sca_inv * dt_step)) * (exp_sca - 1.0)

        e_sca_i_mj = mc_tot_sca * (t_htf_out_t_end[i] - t_sca_last[i]) * 1e-6
        q_sca_abs_sum_w += q_abs_sca_w
        q_sca_loss_sum_w += q_loss_sca_w
        e_sca_sum_mj += e_sca_i_mj

        if i < n_sca - 1:
            t_next, hl_xover = step_interconnect_temp(t_htf_out_t_int[i], t_db_k, m_dot_htf_loop, 4)
            t_htf_in_t_int[i + 1] = t_next
            q_xover_loss_sum_w += hl_xover

    t_loop_out_x, _ = step_interconnect_temp(t_htf_out_t_int[n_sca - 1], t_db_k, m_dot_htf_loop, 3)

    c_hdr_hot = vp1_cp_k(t_loop_out_x) * 1000.0
    t_hdr_h = t_loop_out_x
    hdr_hl_hot = 0.0
    hdr_hl_hot_tot = 0.0
    for i in range(nhdrsec, 2 * nhdrsec):
        if i != nhdrsec:
            t_hdr_h -= hdr_hl_hot / (m_dot_tot * hdr_frac[i - 1] * c_hdr_hot)
        hdr_hl_hot = row_dist * d_hdr[i] * pi * pipe_u * (t_hdr_h - t_db_k)
        hdr_hl_hot_tot += nfsec * hdr_hl_hot

    t_rnr_h = t_hdr_h - hdr_hl_hot / (m_dot_tot * hdr_frac[2 * nhdrsec - 1] * c_hdr_hot)
    rnr_hl_hot = 0.0
    rnr_hl_hot_tot = 0.0
    for i in range(nrunsec, 2 * nrunsec):
        if i != nrunsec:
            t_rnr_h -= rnr_hl_hot / (m_dot_tot * rnr_frac[i - 1] * c_hdr_hot)
        rnr_hl_hot = l_runner[i] * pi * d_runner[i] * pipe_u * (t_rnr_h - t_db_k)
        rnr_hl_hot_tot += 2.0 * rnr_hl_hot

    q_dot_loss_hr_hot = hdr_hl_hot_tot + rnr_hl_hot_tot
    t_sys_h_in = t_loop_out_x - q_dot_loss_hr_hot / (m_dot_tot * c_hdr_hot)

    tau_h_inv = m_dot_tot / (cfg["v_hot"] * rho_hdr_hot + cfg["mc_bal_hot"] / c_hdr_hot)
    exp_h = math.exp(-tau_h_inv * dt_step)
    t_sys_h_t_end = (t_sys_h_last - t_sys_h_in) * exp_h + t_sys_h_in
    t_sys_h_t_int = t_sys_h_in - ((t_sys_h_last - t_sys_h_in) / (tau_h_inv * dt_step)) * (exp_h - 1.0)
    e_hr_hot_mj = (cfg["v_hot"] * rho_hdr_hot * c_hdr_hot + cfg["mc_bal_hot"]) * (t_sys_h_t_end - t_sys_h_last) * 1e-6

    c_htf_ave_ts = vp1_cp_ave_k(t_htf_cold_in_k, t_sys_h_t_int) * 1000.0
    q_dot_htf_to_sink_subts = m_dot_tot * c_htf_ave_ts * (t_sys_h_t_int - t_htf_cold_in_k) * 1e-6

    return {
        "t_sys_c_t_end": t_sys_c_t_end,
        "t_sys_c_t_int": t_sys_c_t_int,
        "t_sys_h_t_end": t_sys_h_t_end,
        "t_sys_h_t_int": t_sys_h_t_int,
        "t_htf_in_t_int": t_htf_in_t_int,
        "t_htf_out_t_end": t_htf_out_t_end,
        "t_htf_out_t_int": t_htf_out_t_int,
        "q_dot_sca_loss_subts": q_sca_loss_sum_w * 1e-6 * n_loops_f,
        "q_dot_sca_abs_subts": q_sca_abs_sum_w * 1e-6 * n_loops_f,
        "q_dot_xover_loss_subts": q_xover_loss_sum_w * 1e-6 * n_loops_f,
        "q_dot_hr_cold_loss_subts": q_dot_loss_hr_cold * 1e-6,
        "q_dot_hr_hot_loss_subts": q_dot_loss_hr_hot * 1e-6,
        "e_dot_sca_subts": e_sca_sum_mj * (n_loops_f / dt_step),
        "e_dot_hr_cold_subts": e_hr_cold_mj / dt_step,
        "e_dot_hr_hot_subts": e_hr_hot_mj / dt_step,
        "q_dot_htf_to_sink_subts": q_dot_htf_to_sink_subts,
        "m_dot_loop": m_dot_htf_loop,
        "m_dot_tot": m_dot_tot,
    }


def simular_campo_solar_sam_8760(
    cfg: dict,
    cenario_clima: dict,
    latitude: float,
    longitude: float,
    tz: int,
) -> dict:
    """
    Simulação física completa de 8760 horas do Coletor-Receptor SAM.
    Gera as 9 séries horárias físicas fundamentais com precisão idêntica ao SAM.
    """
    dni_arr = np.asarray(cenario_clima["dni"], dtype=float)
    tdry_arr = np.asarray(cenario_clima["temp"], dtype=float)
    n_steps = len(dni_arr)

    if "tdew" in cenario_clima:
        tdew_arr = np.asarray(cenario_clima["tdew"], dtype=float)
    else:
        tdew_arr = tdry_arr - 4.5

    if "vento_vel" in cenario_clima:
        wspd_arr = np.asarray(cenario_clima["vento_vel"], dtype=float)
    elif "wind_speed" in cenario_clima:
        wspd_arr = np.asarray(cenario_clima["wind_speed"], dtype=float)
    else:
        wspd_arr = np.full(n_steps, 2.5)

    if "pressao" in cenario_clima:
        p_raw = np.asarray(cenario_clima["pressao"], dtype=float)
        pres_atm_arr = p_raw / 1013.25 if np.nanmean(p_raw) > 500.0 else p_raw
    elif "press" in cenario_clima:
        p_raw = np.asarray(cenario_clima["press"], dtype=float)
        pres_atm_arr = p_raw / 1013.25 if np.nanmean(p_raw) > 500.0 else p_raw
    else:
        pres_atm_arr = np.ones(n_steps)

    n_sca = cfg["n_sca"]
    n_loops_f = float(cfg["n_loops"])
    ap_tot = cfg["ap_tot"]

    phi = math.radians(latitude)
    shift = (longitude - tz * 15.0) * pi / 180.0
    theta_dep = max(math.radians(10.0), 1e-6)
    theta_stow = max(math.radians(170.0), 1e-6)
    iam_c0, iam_c1, iam_c2 = 1.0, 0.0327, -0.1351
    col_opt_eff_0 = 0.988 * 0.952 * 0.93 * 0.97 * 1.0  # SAM: Tracking, geometry, clean refl, dirt, general = 0.848386416

    t_field_ini = 0.5 * (cfg["t_fp_k"] + cfg["t_in_des_k"])
    state_conv = {
        "t_sys_c_t_end_last": t_field_ini,
        "t_sys_h_t_end_last": t_field_ini,
        "t_htf_out_t_end_last": [t_field_ini] * n_sca,
        "op_mode": 0,
    }

    out_q_inc_sf = np.zeros(n_steps)
    out_cos_th = np.zeros(n_steps)
    out_q_inc_costh = np.zeros(n_steps)
    out_q_rec_inc = np.zeros(n_steps)
    out_q_rec_loss = np.zeros(n_steps)
    out_q_rec_abs = np.zeros(n_steps)
    out_e_int_energy = np.zeros(n_steps)
    out_q_htf_out = np.zeros(n_steps)
    out_q_fp = np.zeros(n_steps)
    out_scas_def = np.ones(n_steps)
    out_q_piping_loss = np.zeros(n_steps)

    def copy_state(s):
        return {
            "t_sys_c_t_end_last": s["t_sys_c_t_end_last"],
            "t_sys_h_t_end_last": s["t_sys_h_t_end_last"],
            "t_htf_out_t_end_last": list(s["t_htf_out_t_end_last"]),
            "op_mode": s["op_mode"],
        }

    def run_off_or_su_substeps(state_in, dt_total, is_startup, t_db_k, t_sky_k, v_wind, p_pa, dni_val, q_sca_base, col_eff_base, m_dot_ss):
        n_sub = max(1, int(math.ceil(dt_total / 600.0 - 1e-6)))
        dt_sub = dt_total / float(n_sub)
        st = copy_state(state_in)

        q_loss_acc = 0.0
        q_abs_acc = 0.0
        e_int_acc = 0.0
        q_fp_acc = 0.0
        q_pipe_loss_acc = 0.0
        time_used = 0.0
        reached_on = False

        for _ in range(n_sub):
            if is_startup:
                if dni_val > 50.0 and st["t_htf_out_t_end_last"][-1] > 0.5 * (cfg["t_fp_k"] + cfg["t_startup_k"]):
                    m_dot_loop = min(max(0.8 * m_dot_ss + 0.2 * cfg["m_dot_htfmin"], cfg["m_dot_htfmin"]), cfg["m_dot_htfmax"])
                else:
                    m_dot_loop = cfg["m_dot_htfmin"]
                q_sca_cur = q_sca_base
                col_eff_cur = col_eff_base
            else:
                m_dot_loop = cfg["m_dot_htfmin"]
                q_sca_cur = np.zeros(n_sca)
                col_eff_cur = np.zeros(n_sca)

            t_cold_in = st["t_sys_h_t_end_last"]
            res = loop_energy_balance_t_int(
                cfg, st, t_cold_in, m_dot_loop, t_db_k, t_sky_k, v_wind, p_pa, q_sca_cur, col_eff_cur, dt_sub
            )

            q_fp_sub = 0.0
            if res["t_htf_out_t_end"][-1] < cfg["t_fp_k"] + 10.0:
                q_dot_field_losses = (
                    res["q_dot_xover_loss_subts"]
                    + res["q_dot_hr_cold_loss_subts"]
                    + res["q_dot_hr_hot_loss_subts"]
                    - res["q_dot_sca_abs_subts"]
                )
                if q_dot_field_losses > 0.0:
                    t_cold_orig = t_cold_in
                    c_ave = vp1_cp_ave_k(t_cold_orig, res["t_sys_h_t_int"]) * 1000.0
                    t_cold_in = t_cold_orig + (q_dot_field_losses * 1e6) / (c_ave * res["m_dot_tot"])
                    res = loop_energy_balance_t_int(
                        cfg, st, t_cold_in, m_dot_loop, t_db_k, t_sky_k, v_wind, p_pa, q_sca_cur, col_eff_cur, dt_sub
                    )
                    q_dot_field_losses = (
                        res["q_dot_xover_loss_subts"]
                        + res["q_dot_hr_cold_loss_subts"]
                        + res["q_dot_hr_hot_loss_subts"]
                        - res["q_dot_sca_abs_subts"]
                    )
                    c_ave = vp1_cp_ave_k(t_cold_in, res["t_sys_h_t_int"]) * 1000.0
                    t_cold_in = t_cold_orig + (q_dot_field_losses * 1e6) / (c_ave * res["m_dot_tot"])
                    res = loop_energy_balance_t_int(
                        cfg, st, t_cold_in, m_dot_loop, t_db_k, t_sky_k, v_wind, p_pa, q_sca_cur, col_eff_cur, dt_sub
                    )
                    c_ave = vp1_cp_ave_k(t_cold_in, res["t_sys_h_t_int"]) * 1000.0
                    q_fp_sub = res["m_dot_tot"] * c_ave * (t_cold_in - t_cold_orig) * 1e-6

            q_loss_acc += res["q_dot_sca_loss_subts"] * dt_sub
            q_abs_acc += res["q_dot_sca_abs_subts"] * dt_sub
            e_int_sub = res["e_dot_sca_subts"] + res["e_dot_hr_cold_subts"] + res["e_dot_hr_hot_subts"]
            e_int_acc += e_int_sub * dt_sub
            q_pipe_loss_acc += (res["q_dot_hr_cold_loss_subts"] + res["q_dot_hr_hot_loss_subts"] + res["q_dot_xover_loss_subts"]) * dt_sub
            q_fp_acc += q_fp_sub * dt_sub
            time_used += dt_sub

            st["t_sys_c_t_end_last"] = res["t_sys_c_t_end"]
            st["t_sys_h_t_end_last"] = res["t_sys_h_t_end"]
            st["t_htf_out_t_end_last"] = list(res["t_htf_out_t_end"])

            su_delay_req = cfg.get("rec_su_delay_s", 720.0)
            if is_startup and res["t_sys_h_t_end"] > cfg["t_startup_k"] and time_used >= su_delay_req:
                reached_on = True
                st["op_mode"] = 2
                break

        return {
            "state_end": st,
            "q_loss": q_loss_acc / time_used,
            "q_abs": q_abs_acc / time_used,
            "e_int": e_int_acc / time_used,
            "q_fp": q_fp_acc / time_used,
            "q_pipe_loss": q_pipe_loss_acc / time_used,
            "time_used": time_used,
            "reached_on": reached_on,
        }

    def run_on_step(state_in, dt_on, t_cold_in_k, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base, col_eff_base):
        st = copy_state(state_in)
        t_des_out = cfg["t_out_des_k"]
        m_min = cfg["m_dot_htfmin"]
        m_max = cfg["m_dot_htfmax"]
        df_comp = 1.0

        res = loop_energy_balance_t_int(
            cfg, st, t_cold_in_k, m_min, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base, col_eff_base, dt_on
        )
        err_min = (res["t_htf_out_t_end"][-1] - t_des_out) / t_des_out

        if err_min > 0.001:
            res_max = loop_energy_balance_t_int(
                cfg, st, t_cold_in_k, m_max, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base, col_eff_base, dt_on
            )
            err_max = (res_max["t_htf_out_t_end"][-1] - t_des_out) / t_des_out
            if err_max > 0.001:
                # Limite térmico do loop a m_max (SAM C++ linhas 3761 e 3816: q_abs_maxOT = c_htf_ave * m_dot_htfmax * (T_out_des - T_in_des))
                c_ave_des = vp1_cp_ave_k(t_cold_in_k, t_des_out) * 1000.0
                q_abs_max_ot_loop = c_ave_des * m_max * (t_des_out - t_cold_in_k)
                q_check_loop = (res_max["q_dot_sca_abs_subts"] * 1e6) / n_loops_f
                df_comp = min(max(q_abs_max_ot_loop / max(q_check_loop, 1.0), 0.0), 1.0)
                res = loop_energy_balance_t_int(
                    cfg, st, t_cold_in_k, m_max, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base * df_comp, col_eff_base, dt_on
                )
            else:
                t_lo = res_max["t_htf_out_t_end"][-1]
                t_hi = res["t_htf_out_t_end"][-1]
                dt_in_des = max(t_des_out - t_cold_in_k, 1.0)
                m_guess = min(max(m_max * (t_lo - t_cold_in_k) / dt_in_des, m_min), m_max)
                m_a, f_a = m_min, t_hi - t_des_out
                m_b, f_b = m_max, t_lo - t_des_out
                for _ in range(5):
                    res = loop_energy_balance_t_int(
                        cfg, st, t_cold_in_k, m_guess, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base, col_eff_base, dt_on
                    )
                    f_g = res["t_htf_out_t_end"][-1] - t_des_out
                    if abs(f_g) / t_des_out < 0.0005:
                        break
                    if f_g > 0.0:
                        m_a, f_a = m_guess, f_g
                    else:
                        m_b, f_b = m_guess, f_g
                    m_sec = m_a - f_a * (m_b - m_a) / (f_b - f_a) if abs(f_b - f_a) > 1e-9 else 0.5 * (m_a + m_b)
                    if m_min < m_sec < m_max:
                        m_guess = m_sec
                    else:
                        m_guess = 0.5 * (m_a + m_b)

        st["t_sys_c_t_end_last"] = res["t_sys_c_t_end"]
        st["t_sys_h_t_end_last"] = res["t_sys_h_t_end"]
        st["t_htf_out_t_end_last"] = list(res["t_htf_out_t_end"])

        q_fp_on = 0.0
        e_int_on = res["e_dot_sca_subts"] + res["e_dot_hr_cold_subts"] + res["e_dot_hr_hot_subts"]
        q_htf_rep = max(0.0, res["q_dot_htf_to_sink_subts"])

        q_pipe_loss_on = res["q_dot_hr_cold_loss_subts"] + res["q_dot_hr_hot_loss_subts"] + res["q_dot_xover_loss_subts"]
        return {
            "state_end": st,
            "q_loss": res["q_dot_sca_loss_subts"],
            "q_abs": res["q_dot_sca_abs_subts"],
            "e_int": e_int_on,
            "q_htf": q_htf_rep,
            "q_fp": q_fp_on,
            "df_comp": df_comp,
            "t_sys_h_t_end": res["t_sys_h_t_end"],
            "q_pipe_loss": q_pipe_loss_on,
        }

    for idx in range(n_steps):
        dni = float(dni_arr[idx])
        t_db_k = float(tdry_arr[idx]) + 273.15
        v_wind = float(wspd_arr[idx])
        p_pa = float(pres_atm_arr[idx]) * 101325.0

        time_hr = float(idx + 1)
        hour = math.fmod(time_hr, 24.0)
        t_sky_k = t_db_k * ((0.711 + 0.0056 * float(tdew_arr[idx]) + 0.000073 * (float(tdew_arr[idx]) ** 2) + 0.013 * math.cos(hour * 15.0 * pi / 180.0)) ** 0.25)

        day_of_year = int(math.ceil(time_hr / 24.0))
        b_ang = (day_of_year - 1) * 360.0 / 365.0 * pi / 180.0
        eot = 229.2 * (0.000075 + 0.001868 * math.cos(b_ang) - 0.032077 * math.sin(b_ang) - 0.014615 * math.cos(2.0 * b_ang) - 0.04089 * math.sin(2.0 * b_ang))
        dec = 23.45 * math.sin(360.0 * (284.0 + day_of_year) / 365.0 * pi / 180.0) * pi / 180.0
        solar_noon = 12.0 - (shift * 180.0 / pi) / 15.0 - eot / 60.0

        dep_hr1 = math.cos(phi) / math.tan(theta_dep)
        dep_hr2 = -math.tan(dec) * math.sin(phi) / math.tan(theta_dep)
        arg_dep = min(max((dep_hr1 * dep_hr2 + math.sqrt(max(0.0, dep_hr1 * dep_hr1 - dep_hr2 * dep_hr2 + 1.0))) / (dep_hr1 * dep_hr1 + 1.0), -1.0), 1.0)
        dep_hr3 = math.copysign(1.0, math.tan(pi - theta_dep)) * math.acos(arg_dep) * 180.0 / pi / 15.0
        dep_time = solar_noon + dep_hr3

        stw_hr1 = math.cos(phi) / math.tan(theta_stow)
        stw_hr2 = -math.tan(dec) * math.sin(phi) / math.tan(theta_stow)
        arg_stw = min(max((stw_hr1 * stw_hr2 + math.sqrt(max(0.0, stw_hr1 * stw_hr1 - stw_hr2 * stw_hr2 + 1.0))) / (stw_hr1 * stw_hr1 + 1.0), -1.0), 1.0)
        stw_hr3 = math.copysign(1.0, math.tan(pi - theta_stow)) * math.acos(arg_stw) * 180.0 / pi / 15.0
        stow_time = solar_noon + stw_hr3

        hr_a = hour - 1.0
        hr_b = hour
        if hr_b > dep_time and hr_a < stow_time:
            if hr_a < dep_time:
                ftrack = (hr_b - dep_time) / 1.0
                mid_track = hr_b - ftrack * 0.5
            elif hr_b > stow_time:
                ftrack = (stow_time - hr_a) / 1.0
                mid_track = hr_a + ftrack * 0.5
            else:
                ftrack = 1.0
                mid_track = hr_a + 0.5
        else:
            ftrack = 0.0
            mid_track = hr_a + 0.5

        std_time = mid_track
        solar_time = std_time + (shift * 180.0 / pi) / 15.0 + eot / 60.0
        omega = (solar_time - 12.0) * 15.0 * pi / 180.0
        solar_alt = math.asin(min(max(math.sin(dec) * math.sin(phi) + math.cos(phi) * math.cos(dec) * math.cos(omega), -1.0), 1.0))

        q_inc_sf = dni * ap_tot * 1e-6
        out_q_inc_sf[idx] = q_inc_sf

        if solar_alt <= 0.0:
            solar_az = 0.0
            cos_th = 0.0
        else:
            if abs(omega) < 1e-6:
                solar_az = 0.0 if (dec <= phi) else pi
            else:
                arg_az = (math.cos(dec) * math.cos(omega) * math.sin(phi) - math.cos(phi) * math.sin(dec)) / math.cos(solar_alt)
                solar_az = math.copysign(math.acos(min(max(arg_az, -1.0), 1.0)), omega)
            if ftrack == 0.0:
                cos_th = 1.0 if dni > 0.0 else 0.0
            else:
                cos_th = math.sqrt(max(0.0, 1.0 - (math.cos(solar_alt) * math.cos(solar_az)) ** 2))

        out_cos_th[idx] = cos_th
        out_q_inc_costh[idx] = q_inc_sf * cos_th

        is_on_sun = (solar_alt > 0.0 and ftrack > 0.0 and dni >= 1.0 and v_wind < 25.0)
        if is_on_sun:
            theta = math.acos(min(max(cos_th, -1.0), 1.0))
            tan_th = math.tan(theta)
            iam = min(max(iam_c0 + iam_c1 * theta / max(cos_th, 1e-6) + iam_c2 * (theta ** 2) / max(cos_th, 1e-6), 0.0), 1.0)
            opt_derate = col_opt_eff_0 * iam * ftrack

            sin_z = math.cos(solar_alt)
            cos_z = math.sin(solar_alt)
            sin_a = math.sin(solar_az)
            phi_t = math.atan2(sin_z * sin_a, max(cos_z, 1e-9))
            row_shadow = abs(math.cos(phi_t)) * cfg["row_distance"] / W_APERTURE
            if row_shadow < 0.5:
                row_shadow = 0.0
            elif row_shadow > 1.0:
                row_shadow = 1.0

            sol_sul = abs(solar_az) <= (pi / 2.0)
            gain_u = max(L_FOCAL * tan_th - D_SCA, 0.0) / L_SCA
            end_loss = np.empty(n_sca)
            for a in range(n_sca):
                eg = 0.0
                if sol_sul:
                    if a != 0 and a != (n_sca - 1):
                        eg = gain_u
                else:
                    if a != (n_sca // 2 - 1) and a != (n_sca // 2):
                        eg = gain_u
                end_loss[a] = min(max(1.0 - L_FOCAL * tan_th / L_SCA + eg, 0.0), 1.0)

            col_eff_base = opt_derate * row_shadow * end_loss
            q_sca_base = np.full(n_sca, dni * cos_th * A_APERTURE / L_SCA)
            m_dot_ss = (dni * cos_th * opt_derate * row_shadow * float(np.mean(end_loss))) / (cfg["i_bn_des"] * cfg["opteff_des"]) * cfg["m_dot_loop_des"]
            q_rec_inc_base = float(np.sum(q_sca_base * L_SCA * col_eff_base * SHADOW_0 * DIRT_HCE_0 * TAU_ENV_0)) * n_loops_f * 1e-6
        else:
            col_eff_base = np.zeros(n_sca)
            q_sca_base = np.zeros(n_sca)
            m_dot_ss = cfg["m_dot_htfmin"]
            q_rec_inc_base = 0.0

        op_mode = state_conv["op_mode"]
        t_cold_in_des = cfg.get("t_cold_in_op_k", cfg["t_in_des_k"])

        if op_mode == 2:
            res_on = run_on_step(state_conv, 3600.0, t_cold_in_des, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base, col_eff_base)
            state_conv = res_on["state_end"]
            if res_on["t_sys_h_t_end"] < cfg["t_shutdown_k"]:
                state_conv["op_mode"] = 1 if is_on_sun else 0
            out_q_rec_inc[idx] = q_rec_inc_base * res_on["df_comp"]
            out_q_rec_loss[idx] = res_on["q_loss"]
            out_q_rec_abs[idx] = res_on["q_abs"]
            out_e_int_energy[idx] = res_on["e_int"]
            out_q_htf_out[idx] = res_on["q_htf"]
            out_q_fp[idx] = res_on["q_fp"]
            out_scas_def[idx] = res_on["df_comp"]
            out_q_piping_loss[idx] = res_on["q_pipe_loss"]
        else:
            if is_on_sun:
                res_su = run_off_or_su_substeps(state_conv, 3600.0, True, t_db_k, t_sky_k, v_wind, p_pa, dni, q_sca_base, col_eff_base, m_dot_ss)
                if res_su["reached_on"] and res_su["time_used"] < 3600.0:
                    dt_su = res_su["time_used"]
                    dt_on = 3600.0 - dt_su
                    res_on = run_on_step(res_su["state_end"], dt_on, t_cold_in_des, t_db_k, t_sky_k, v_wind, p_pa, q_sca_base, col_eff_base)
                    state_conv = res_on["state_end"]
                    if res_on["t_sys_h_t_end"] < cfg["t_shutdown_k"]:
                        state_conv["op_mode"] = 1 if is_on_sun else 0
                    else:
                        state_conv["op_mode"] = 2

                    out_q_rec_inc[idx] = (q_rec_inc_base * dt_su + q_rec_inc_base * res_on["df_comp"] * dt_on) / 3600.0
                    out_q_rec_loss[idx] = (res_su["q_loss"] * dt_su + res_on["q_loss"] * dt_on) / 3600.0
                    out_q_rec_abs[idx] = (res_su["q_abs"] * dt_su + res_on["q_abs"] * dt_on) / 3600.0
                    out_e_int_energy[idx] = (res_su["e_int"] * dt_su + res_on["e_int"] * dt_on) / 3600.0
                    out_q_htf_out[idx] = (0.0 * dt_su + res_on["q_htf"] * dt_on) / 3600.0
                    out_q_fp[idx] = (res_su["q_fp"] * dt_su + res_on["q_fp"] * dt_on) / 3600.0
                    out_scas_def[idx] = (1.0 * dt_su + res_on["df_comp"] * dt_on) / 3600.0
                    out_q_piping_loss[idx] = (res_su["q_pipe_loss"] * dt_su + res_on["q_pipe_loss"] * dt_on) / 3600.0
                else:
                    state_conv = res_su["state_end"]
                    if not res_su["reached_on"]:
                        state_conv["op_mode"] = 1
                    else:
                        state_conv["op_mode"] = 2

                    out_q_rec_inc[idx] = q_rec_inc_base
                    out_q_rec_loss[idx] = res_su["q_loss"]
                    out_q_rec_abs[idx] = res_su["q_abs"]
                    out_e_int_energy[idx] = res_su["e_int"]
                    out_q_htf_out[idx] = 0.0
                    out_q_fp[idx] = res_su["q_fp"]
                    out_scas_def[idx] = 1.0
                    out_q_piping_loss[idx] = res_su["q_pipe_loss"]
            else:
                res_off = run_off_or_su_substeps(state_conv, 3600.0, False, t_db_k, t_sky_k, v_wind, p_pa, 0.0, q_sca_base, col_eff_base, m_dot_ss)
                state_conv = res_off["state_end"]
                state_conv["op_mode"] = 0
                out_q_rec_inc[idx] = 0.0
                out_q_rec_loss[idx] = res_off["q_loss"]
                out_q_rec_abs[idx] = res_off["q_abs"]
                out_e_int_energy[idx] = res_off["e_int"]
                out_q_htf_out[idx] = 0.0
                out_q_fp[idx] = res_off["q_fp"]
                out_scas_def[idx] = 1.0
                out_q_piping_loss[idx] = res_off["q_pipe_loss"]

    return {
        "q_inc_sf": out_q_inc_sf,
        "cos_th": out_cos_th,
        "q_inc_costh": out_q_inc_costh,
        "q_rec_inc": out_q_rec_inc,
        "q_rec_loss": out_q_rec_loss,
        "q_rec_abs": out_q_rec_abs,
        "e_int_energy": out_e_int_energy,
        "q_htf_out": out_q_htf_out,
        "q_fp": out_q_fp,
        "scas_def": out_scas_def,
        "q_piping_loss": out_q_piping_loss,
    }
