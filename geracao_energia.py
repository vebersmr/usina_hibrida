# -*- coding: utf-8 -*-
"""
Módulo Unificado de Geração de Energia e Armazenamento:
1. Usina Termossolar (CSP Parabolic Trough + TES)
2. Usina Fotovoltaica (PV)
3. Usina Eólica (Wind Farm + Wake Models)
4. Sistema de Armazenamento por Baterias (BESS + Rainflow)

Todas as funções recebem o dicionário de cenário gerado por 'gerador_climatico.py'
e retornam vetores horários padronizados em Watts [W].
"""

from collections import Counter, defaultdict, deque
import math
from math import pi
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pvlib
from scipy.interpolate import interp1d

import modelagem_csp
from simulacao_eolica_isolada import (
    AdjustmentFactors,
    ConstantWakeModel,
    ParkWakeModel,
    SimpleWakeModel,
    WakeModelOptions,
    WindData,
    WindPowerCalculator,
    WindTurbine,
    _calculate_losses,
    _get_fixed_losses,
    # parametros_padrao_parque_100mw,
)


# ==============================================================================
# 1. MÓDULO CSP (CALHA PARABÓLICA + TES)
# ==============================================================================

# Constantes Físicas e de Engenharia CSP
LATITUDE_CSP = -13.27
LONGITUDE_CSP = -43.42
ZONE_CSP = -3
PHI_CSP = math.radians(LATITUDE_CSP)
PSI_CSP = LONGITUDE_CSP
THETA_COL = 0
GAMMA_COL = 0
THETA_DEP = math.radians(10)
THETA_STOW = math.radians(170)

C1 = 2.331e-20                     # [mmHg*cm³/K]
CSI = 2.4e-8                       # [cm]
SIGMA = 5.67e-8                    # Constante de Stefan-Boltzmann
GRAV = 9.81                        # Gravidade [m/s²]

Hitec = modelagem_csp.Fluid(1)
air = modelagem_csp.Fluid(2)
field = modelagem_csp.SolarField()
n_loops = field.actual_nLoops()

T_COLD_IN_DEFAULT = field.system.loop_inlet_htf_temperature   # 293 °C
T_HOT_OUT_DEFAULT = field.system.loop_outlet_htf_temperature  # 391 °C
T_HTF_IN = T_COLD_IN_DEFAULT + 273.15
T_HTF_MED_K = (T_COLD_IN_DEFAULT + T_HOT_OUT_DEFAULT) * 0.5 + 273.15

A_APERTURE = field.col.reflective_aperture_area
L_FOCAL = field.col.average_surface_to_focus_path_length
L_SCA = field.col.length_collector_assembly
D_SCA = field.col.piping_distance_between_assemblies

DIRT_HCE = field.rec.Dirt_HCE
ALPHA_ABS = field.rec.alpha_abs
TAU_ENV = field.rec.tau_env
SHADOW = field.rec.Shadowing
P_A = field.rec.P_a[0]
D2 = field.rec.absorber_tube_inner_diameter
D3 = field.rec.absorber_tube_outer_diameter
D4 = field.rec.glass_envelope_inner_diameter
D5 = field.rec.glass_envelope_outer_diameter

EPS_MATRIX = np.array(field.rec.epsilon_3_11)
EPS_4 = field.rec.epsilon_env[0]
INTERP_EPS_3 = interp1d(EPS_MATRIX[:, 0], EPS_MATRIX[:, 1], kind='linear')

CYCLE_THERMAL_POWER = field.system.cycle_thermal_power()
TES_CAP_DEFAULT = CYCLE_THERMAL_POWER * field.system.hours_of_storage_at_design_point
PC_EFF_DEFAULT = field.system.cycle_thermal_efficiency
P_GROSS_DEFAULT = CYCLE_THERMAL_POWER * PC_EFF_DEFAULT        # 111.0 MWe para 100 MWe líquidos
DEGRADATION_CSP_DEFAULT = 0.2                                 # [%/ano]
FATOR_LIQUIDO_CSP = 0.89                                      # Fator de conversão bruto -> líquido


def calcular_perda_tes_w(
    TES_hours: float,
    P_gross: float = P_GROSS_DEFAULT,
    eta: float = PC_EFF_DEFAULT,
    T_cold_in: float = T_COLD_IN_DEFAULT,
    T_hot_out: float = T_HOT_OUT_DEFAULT,
    dT_approach: float = 5.0,
    h_min: float = 1.0,
    h_tank_in: float = 12.0,
    tank_pairs: int = 1,
    u_tank: float = 0.4,
) -> float:
    """
    Calcula a perda térmica dos tanques de armazenamento térmico (TES) em Watts [Wt]
    seguindo exatamente a modelagem física do SAM com o fluido Hitec (Fluid(1)).
    """
    if TES_hours <= 0.0 or P_gross <= 0.0:
        return 0.0

    cycle_power = P_gross / eta
    Q_tes_des = cycle_power * TES_hours  # C na fórmula [MWht]

    T_tes_hot = T_hot_out - dT_approach + 273.15
    T_tes_cold = T_cold_in + dT_approach + 273.15

    T_tes_ave = 0.5 * (T_tes_hot + T_tes_cold) - 273.15

    rho_ave = Hitec.dens(T_tes_ave, 1)
    cp_ave = Hitec.Cp_ave(T_tes_cold, T_tes_hot)

    vol_one_temp_avail = Q_tes_des * 3600.0 / (
        rho_ave * cp_ave / 1000.0 * (T_tes_hot - T_tes_cold)
    )
    vol_one_temp_total = vol_one_temp_avail / (1.0 - h_min / h_tank_in)

    A_cs = vol_one_temp_total / (h_tank_in * tank_pairs)
    d_tank_out = pow(A_cs / pi, 0.5) * 2.0

    UA_tanks_one_temp = u_tank * (A_cs + pi * d_tank_out * h_tank_in) * tank_pairs

    T_amb_des = 15.0 + 273.15
    q_dot_loss_cold = UA_tanks_one_temp * (T_tes_cold - T_amb_des) * 1.0e-6
    q_dot_loss_hot = UA_tanks_one_temp * (T_tes_hot - T_amb_des) * 1.0e-6
    q_dot_loss_des = q_dot_loss_cold + q_dot_loss_hot  # [MWt]

    return q_dot_loss_des * 1.0e6  # Converte de [MWt] para [Wt]


HTF_LOSS_DEFAULT = calcular_perda_tes_w(
    TES_hours=field.system.hours_of_storage_at_design_point,
    P_gross=P_GROSS_DEFAULT,
    eta=PC_EFF_DEFAULT,
)


def _calc_efficiency_csp(theta, azi, elev, omega_col, ftrack, iam):
    n_sca = field.nSCA
    end_gain = np.zeros(n_sca)

    row_shadow = abs(math.cos(omega_col)) * field.row_spacing / field.col.aperture_width
    row_shadow = max(0, min(row_shadow, 1)) if elev >= 0 else 0

    for a in range(n_sca):
        if abs(azi - 180) <= 90:
            if a != 0 and a != (n_sca - 1):
                end_gain[a] = max(L_FOCAL * math.tan(theta) - D_SCA, 0) / L_SCA
        else:
            if a != (math.floor(n_sca / 2) - 1) and a != math.floor(n_sca / 2):
                end_gain[a] = max(L_FOCAL * math.tan(theta) - D_SCA, 0) / L_SCA

    end_loss = 1 - L_FOCAL * math.tan(theta) / L_SCA + end_gain
    col_opt_eta = field.col.Col_opt_eff * row_shadow * end_loss * ftrack * iam * math.cos(theta)
    return col_opt_eta * DIRT_HCE[0] * SHADOW[0] * TAU_ENV[0] * ALPHA_ABS[0]


def ler_metadados_nsrdb(path_pasta: str, prefixo_arquivo: str = "") -> dict:
    """
    Lê as duas primeiras linhas do arquivo climático NSRDB para extrair
    Latitude, Longitude, Time Zone, Elevation e definir a inclinação PV (|Latitude|).
    """
    arquivos = [
        f for f in os.listdir(path_pasta)
        if f.endswith(".csv") and (prefixo_arquivo in f)
    ]
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV com prefixo '{prefixo_arquivo}' encontrado em '{path_pasta}'."
        )

    caminho_csv = os.path.join(path_pasta, sorted(arquivos)[0])
    df_meta = pd.read_csv(caminho_csv, nrows=1)

    lat = float(df_meta["Latitude"].iloc[0])
    lon = float(df_meta["Longitude"].iloc[0])
    tz = int(float(df_meta["Time Zone"].iloc[0]))
    alt = float(df_meta["Elevation"].iloc[0])

    return {
        "location_id": str(df_meta["Location ID"].iloc[0]),
        "latitude": lat,
        "longitude": lon,
        "tz": tz,
        "altitude": alt,
        "bs_pv": abs(lat),  # Inclinação dos módulos igual ao módulo da latitude
    }


def calc_angles_eff_csp(
    zone: float = ZONE_CSP,
    phi: float = PHI_CSP,
    psi: float = PSI_CSP,
    theta_col: float = THETA_COL,
    gamma_col: float = GAMMA_COL,
    theta_dep: float = THETA_DEP,
    theta_stow: float = THETA_STOW,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> np.ndarray:
    """Calcula a matriz (8760, 8) de eficiência óptica horária do campo solar CSP."""
    if latitude is not None:
        phi = math.radians(latitude)
    if longitude is not None:
        psi = longitude

    eff = np.empty((8760, 8))
    dt = 1.0
    iam_matrix = [1, 0.0327, -0.1351]

    for i in range(8760):
        hour = (i % 24) + 1
        day = math.ceil(i / 24) + 1
        b_angle = math.radians((day - 1) * 360 / 365)
        eot = 229.2 * (
            0.000075
            + 0.001868 * math.cos(b_angle)
            - 0.032077 * math.sin(b_angle)
            - 0.014615 * math.cos(2 * b_angle)
            - 0.04089 * math.sin(2 * b_angle)
        )
        shift = 15 * zone - psi
        delta = math.radians(23.45 * math.sin(math.radians((284 + day) * 360 / 365)))
        noon = 12 - shift / 15 - eot / 60

        dep_hr1 = math.cos(phi) / math.tan(theta_dep)
        dep_hr2 = -math.tan(delta) * math.sin(phi) / math.tan(theta_dep)
        dep_hr3 = (
            np.sign(math.tan(math.pi - theta_dep))
            * math.acos((dep_hr1 * dep_hr2 + math.sqrt(dep_hr1**2 - dep_hr2**2 + 1)) / (dep_hr1**2 + 1))
            * 180 / math.pi / 15
        )
        dep_time = noon + dep_hr3

        stw_hr1 = math.cos(phi) / math.tan(theta_stow)
        stw_hr2 = -math.tan(delta) * math.sin(phi) / math.tan(theta_stow)
        stw_hr3 = (
            np.sign(math.tan(math.pi - theta_stow))
            * math.acos((stw_hr1 * stw_hr2 + math.sqrt(stw_hr1**2 - stw_hr2**2 + 1)) / (stw_hr1**2 + 1))
            * 180 / math.pi / 15
        )
        stw_time = noon + stw_hr3

        hr_a, hr_b = hour - dt, hour

        if hr_b > dep_time and hr_a < stw_time:
            if hr_a < dep_time:
                ftrack = (hr_b - dep_time) / dt
                mid_track = hr_b - ftrack * 0.5 * dt
            elif hr_b > stw_time:
                ftrack = (stw_time - hr_a) / dt
                mid_track = hr_a + ftrack * 0.5 * dt
            else:
                ftrack, mid_track = 1, hr_a + 0.5 * dt
        else:
            ftrack, mid_track = 0, hr_a + 0.5 * dt

        t_sol = mid_track + shift / 15 + eot / 60
        omega = math.radians(15 * (t_sol - 12))
        elev_calc = math.asin(
            math.sin(delta) * math.sin(phi) + math.cos(phi) * math.cos(delta) * math.cos(omega)
        )
        zen_calc = math.pi / 2 - elev_calc
        azi_calc = np.sign(omega) * abs(
            math.acos((math.cos(zen_calc) * math.sin(phi) - math.sin(delta)) / (math.sin(zen_calc) * math.cos(phi)))
        )
        omega_col_calc = math.atan(
            math.cos(elev_calc)
            * math.sin(azi_calc - gamma_col)
            / (
                math.sin(elev_calc - theta_col)
                + math.sin(theta_col) * math.cos(elev_calc) * (1 - math.cos(azi_calc - gamma_col))
            )
        )

        if ftrack == 0:
            costh_calc = 1
        else:
            costh_calc = math.sqrt(
                1
                - (
                    math.cos(elev_calc - theta_col)
                    - math.cos(theta_col) * math.cos(elev_calc) * (1 - math.cos(azi_calc - gamma_col))
                ) ** 2
            )
        theta_calc = math.acos(costh_calc)

        iam = iam_matrix[0] + sum(
            iam_matrix[j] * (theta_calc**j) / costh_calc for j in range(1, len(iam_matrix))
        )
        iam = max(0, min(iam, 1))

        eff[i] = _calc_efficiency_csp(theta_calc, azi_calc, elev_calc, omega_col_calc, ftrack, iam)
    return eff


def fq_34conv(t_3, t_4):
    t_34 = (t_3 + t_4) / 2
    lam = C1 * t_34 / (P_A * CSI**2)
    cp_34 = field.annulus_gas.Cp(t_34 - 273.15) * 1e3
    cv_34 = field.annulus_gas.Cv(t_34) * 1e3
    gamma_gas = cp_34 / cv_34
    b_val = (9 * gamma_gas - 5) / (2 * gamma_gas + 2)
    k_34 = field.annulus_gas.cond(t_34 - 273.15)
    h_conv = k_34 / (D3 / 2 * math.log(D4 / D3) + b_val * lam / 100 * (D3 / D4 + 1))
    return math.pi * D3 * h_conv * (t_3 - t_4)


def fq_34rad(t_3, t_4):
    eps_3 = INTERP_EPS_3(t_3 - 273.15)
    h_rad = SIGMA * (t_3**2 + t_4**2) * (t_3 + t_4) / (1 / eps_3 + D3 / D4 * (1 / EPS_4 - 1))
    return math.pi * D3 * h_rad * (t_3 - t_4)


def fq_cond_bracket(t_3, t_6, p_6, v_6):
    p_brac, d_brac, a_cs_brac, k_brac_steel = 0.2032, 0.0508, 0.00016129, 48.0
    t_base = t_3 - 10.0
    t_brac = (t_base + t_6) / 2
    t_brac6 = (t_brac + t_6) / 2.0

    if v_6 <= 0.1:
        mu_brac6 = air.visc(t_brac6)
        rho_brac6 = air.dens(t_brac6, p_6)
        cp_brac6 = air.Cp(t_brac6) * 1000
        k_brac6 = air.cond(t_brac6)
        nu_brac6 = mu_brac6 / rho_brac6
        alpha_brac6 = k_brac6 / (cp_brac6 * rho_brac6)
        beta_brac6 = 1 / t_brac6
        ra_dbrac = GRAV * beta_brac6 * abs(t_brac - t_6) * d_brac**3 / (alpha_brac6 * nu_brac6)
        pr_brac6 = nu_brac6 / alpha_brac6
        nu_bar = pow(
            0.60 + (0.387 * pow(ra_dbrac, 0.1667)) / pow(1 + pow(0.559 / pr_brac6, 0.5625), 0.2963),
            2,
        )
        h_brac6 = nu_bar * k_brac6 / d_brac
    else:
        mu_brac = air.visc(t_brac)
        mu_6 = air.visc(t_6)
        rho_6 = air.dens(t_6, p_6)
        rho_brac = air.dens(t_brac, p_6)
        k_brac = air.cond(t_brac)
        k_6 = air.cond(t_6)
        k_brac6 = air.cond(t_brac6)
        cp_brac = air.Cp(t_brac) * 1000
        cp_6 = air.Cp(t_6) * 1000
        nu_6 = mu_6 / rho_6
        nu_brac = mu_brac / rho_brac
        alpha_brac = k_brac / (cp_brac * rho_brac)
        alpha_6 = k_6 / (cp_6 * rho_6)
        re_dbrac = v_6 * d_brac / nu_6
        pr_brac = nu_brac / alpha_brac
        pr_6 = nu_6 / alpha_6

        n_val = 0.37 if pr_6 <= 10 else 0.36
        if re_dbrac < 40:
            c_val, m_val = 0.75, 0.4
        elif 40 <= re_dbrac < 1e3:
            c_val, m_val = 0.51, 0.5
        elif 1e3 <= re_dbrac < 2e5:
            c_val, m_val = 0.26, 0.6
        else:
            c_val, m_val = 0.076, 0.7

        nu_bar = c_val * pow(re_dbrac, m_val) * pow(pr_6, n_val) * pow(pr_6 / pr_brac, 0.25)
        h_brac6 = nu_bar * k_brac6 / d_brac

    l_hce = 4.06
    return math.sqrt(h_brac6 * p_brac * k_brac_steel * a_cs_brac) * (t_base - t_6) / l_hce


def simular_receptor_csp(
    cenario_clima: dict,
    opt_eff: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa o LOOP 1 (Potência Térmica do Receptor CSP).
    Retorna: (p_rec [MWt], t_in_celsius [°C], p_rec_loss [MWt]).
    """
    dni_vetor = cenario_clima["dni"]
    wind_vetor = cenario_clima["vento_vel"]
    press_vetor = cenario_clima["press"]
    temp_vetor = cenario_clima["temp"]
    n_horas = len(dni_vetor)

    p_rec = np.zeros(n_horas)
    p_rec_loss = np.zeros(n_horas)
    t_in = np.zeros(n_horas)
    t_in[0] = 221.0 + 273.15
    fator_exp = 0.983955

    for i in range(n_horas):
        q_sca = dni_vetor[i] * A_APERTURE / L_SCA

        if i > 0:
            if dni_vetor[i] < 75:
                t_amb = temp_vetor[i] + 273.15
                t_in[i] = t_amb + (t_in[i - 1] - t_amb) * fator_exp
                t_1 = t_in[i]
            else:
                t_in[i] = T_HTF_IN
                t_1 = T_HTF_MED_K
        else:
            t_1 = t_in[i]

        t_2 = t_1 + 2.0
        t_3 = t_2 + 5.0
        t_4 = t_3 - 0.9 * (t_3 - (temp_vetor[i] + 273.15))

        q_3solabs = q_sca * opt_eff[i % 8760]
        q_abs_total = sum(q_3solabs * L_SCA * 1e-6)

        q_34conv = fq_34conv(t_3, t_4)
        q_34rad = fq_34rad(t_3, t_4)
        q_cond_bracket = fq_cond_bracket(t_3, temp_vetor[i], press_vetor[i], wind_vetor[i])

        q_heatloss = (q_34conv + q_34rad + q_cond_bracket) * L_SCA * 1e-6
        q_loss_total = q_heatloss * field.nSCA * n_loops
        q_abs_total *= n_loops

        df = 1.0 if q_abs_total < 544.0 else (544.0 / q_abs_total)

        p_rec[i] = max(0.0, q_abs_total * df - q_loss_total)
        p_rec_loss[i] = q_loss_total

    return p_rec, (t_in - 273.15), p_rec_loss


def despachar_csp_tes(
    p_in: np.ndarray,
    t_htf_celsius: np.ndarray,
    tes_capacity: float = TES_CAP_DEFAULT,
    cycle_power: float = CYCLE_THERMAL_POWER,
    htf_loss: Optional[float] = None,
    htf_ini: float = 0.0,
    pc_eff: float = PC_EFF_DEFAULT,
    deg: float = DEGRADATION_CSP_DEFAULT,
    start: float = 0.2,
    fator_campo: float = 1.0,
) -> dict:
    """
    Executa a lógica de despacho térmico do CSP+TES corrigindo a escala de
    partida da turbina (cycle_power), de inércia térmica do campo (fator_campo)
    e calculando dinamicamente 'htf_loss' caso não seja informado.
    """
    if htf_loss is None:
        horas_tes_calc = (tes_capacity / cycle_power) if cycle_power > 0 else 0.0
        p_gross_calc = cycle_power * pc_eff
        htf_loss = calcular_perda_tes_w(TES_hours=horas_tes_calc, P_gross=p_gross_calc, eta=pc_eff)

    out_fraction = 1
    pc_on = 0
    tes_capacity_inicial = htf_ini
    receiver_power_htf = p_in * 1e6
    p_cycle_local = cycle_power * 1e6

    start_pc = aux_pc_on = start * cycle_power * 1e6
    n_total = len(receiver_power_htf)
    pout = np.zeros(8760)
    pout_horario_bruto_w = np.zeros(n_total)
    pot_anual = []

    p_htf_old = 0.0
    deg_acumulado = 1.0

    for j in range(n_total):
        kk = j % 8760
        if j > 0 and kk == 0:
            receiver_power_htf = p_in * 1e6
            p_htf_old = 0.0
            deg_acumulado *= (1.0 - deg / 100.0)
            pout = np.zeros(8760)

        p_cycle_local = out_fraction * cycle_power * 1e6
        p_htf_new = (-3.6 * t_htf_celsius[j] + 1170.0) * 1e6 * fator_campo
        aux_p_htf = p_htf_old * 0.6

        if p_htf_new < p_htf_old:
            for ii in range(j, n_total):
                if aux_p_htf <= 0:
                    break
                if receiver_power_htf[ii] > aux_p_htf:
                    receiver_power_htf[ii] -= aux_p_htf
                    aux_p_htf = 0.0
                else:
                    aux_p_htf -= receiver_power_htf[ii]
                    receiver_power_htf[ii] = 0.0
        p_htf_old = p_htf_new

        if out_fraction > 0:
            if receiver_power_htf[j] > 0:
                if pc_on == 0:
                    if receiver_power_htf[j] > aux_pc_on:
                        if receiver_power_htf[j] > p_cycle_local:
                            pout[kk] = (p_cycle_local - aux_pc_on) * pc_eff * deg_acumulado
                            tes_capacity_inicial += receiver_power_htf[j] - p_cycle_local - htf_loss
                        else:
                            pout[kk] = (receiver_power_htf[j] - aux_pc_on) * pc_eff * deg_acumulado
                        aux_pc_on = 0.0
                        pc_on = 1
                    else:
                        aux_pc_on -= receiver_power_htf[j]
                        pout[kk] = 0.0
                else:
                    if receiver_power_htf[j] > p_cycle_local:
                        pout[kk] = p_cycle_local * pc_eff * deg_acumulado
                        tes_capacity_inicial += receiver_power_htf[j] - p_cycle_local - htf_loss
                        if tes_capacity_inicial > tes_capacity * 1e6:
                            receiver_power_htf[j] -= (tes_capacity_inicial - tes_capacity * 1e6)
                            tes_capacity_inicial = tes_capacity * 1e6
                    else:
                        if tes_capacity_inicial > (p_cycle_local - receiver_power_htf[j]):
                            pout[kk] = (
                                receiver_power_htf[j] + (p_cycle_local - receiver_power_htf[j])
                            ) * pc_eff * deg_acumulado
                            tes_capacity_inicial -= (p_cycle_local - receiver_power_htf[j] + htf_loss)
                            if tes_capacity_inicial < 0:
                                tes_capacity_inicial = 0.0
                        else:
                            if tes_capacity_inicial > 0:
                                pout[kk] = (receiver_power_htf[j] + tes_capacity_inicial) * pc_eff * deg_acumulado
                                tes_capacity_inicial = 0.0
                            else:
                                pout[kk] = receiver_power_htf[j] * pc_eff * deg_acumulado
            elif tes_capacity_inicial > 0:
                if pc_on == 0:
                    if tes_capacity_inicial > aux_pc_on:
                        if tes_capacity_inicial > p_cycle_local:
                            pout[kk] = (p_cycle_local - aux_pc_on) * pc_eff * deg_acumulado
                            tes_capacity_inicial -= (p_cycle_local + htf_loss)
                        else:
                            pout[kk] = (tes_capacity_inicial - aux_pc_on) * pc_eff * deg_acumulado
                            tes_capacity_inicial = 0.0
                        aux_pc_on = 0.0
                        pc_on = 1
                    else:
                        pout[kk] = 0.0
                        pc_on = 0
                        aux_pc_on = start_pc
                else:
                    if tes_capacity_inicial > p_cycle_local:
                        pout[kk] = p_cycle_local * pc_eff * deg_acumulado
                        tes_capacity_inicial -= (p_cycle_local + htf_loss)
                        if tes_capacity_inicial < 0:
                            tes_capacity_inicial = 0.0
                    else:
                        if tes_capacity_inicial > 0.02 * cycle_power * 1e6:
                            pout[kk] = tes_capacity_inicial * pc_eff * deg_acumulado
                            tes_capacity_inicial = 0.0
                        else:
                            pout[kk] = 0.0
                            pc_on = 0
                            aux_pc_on = start_pc
            else:
                pout[kk] = 0.0
                pc_on = 0
                aux_pc_on = start_pc

        pout_horario_bruto_w[j] = pout[kk]

        if j > 0 and ((j + 1) % 8760) == 0:
            pot_anual.append(sum(pout))

    pot_anual_bruta = pd.Series(pot_anual)
    pot_anual_liquida = pot_anual_bruta * FATOR_LIQUIDO_CSP
    pout_horario_liquido_w = pout_horario_bruto_w * FATOR_LIQUIDO_CSP

    return {
        "p_horaria_bruta_w": pout_horario_bruto_w,
        "p_horaria_liquida_w": pout_horario_liquido_w,
        "pot_anual_bruta_wh": pot_anual_bruta,
        "pot_anual_liquida_wh": pot_anual_liquida,
    }


# ==============================================================================
# 2. MÓDULO FOTOVOLTAICO (PV PURO VETORIZADO)
# ==============================================================================

def preparar_geometria_solar_pv(
    n_horas: int,
    vida_util: int,
    latitude: float = -13.27,
    longitude: float = -43.42,
    altitude: float = 446.0,
    tz: int = -3,
    ys: float = 180.0,
    bs: Optional[float] = None,
    ano_base: int = 1998,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pré-calcula o Ângulo de Incidência (AOI) e o Zênite Solar sem dias bissextos."""
    if bs is None:
        bs = abs(latitude)

    # No padrão POSIX (Etc/GMT), o sinal do fuso é invertido (ex: UTC-3 -> 'Etc/GMT+3')
    etc_gmt_str = f"Etc/GMT{-int(tz):+d}"
    site = pvlib.location.Location(latitude, longitude, etc_gmt_str, altitude, 'America/Sao_Paulo')
    times = pd.date_range(
        start=f'{ano_base}-01-01 00:30:00',
        periods=n_horas + ((vida_util // 4) + 1) * 24,
        freq='h',
        name='America/Sao_Paulo',
        tz=site.tz,
    )
    solar_pos = pvlib.solarposition.get_solarposition(times, site.latitude, site.longitude, site.altitude)

    z_rad = solar_pos["zenith"].to_numpy() * np.pi / 180.0
    y_rad = solar_pos["azimuth"].to_numpy() * np.pi / 180.0
    ys_rad = ys * np.pi / 180.0
    bs_rad = bs * np.pi / 180.0

    a_aoi = np.sin(z_rad) * np.cos(y_rad - ys_rad) * np.sin(bs_rad) + np.cos(z_rad) * np.cos(bs_rad)

    mask_sem_bissexto = ~((solar_pos.index.month == 2) & (solar_pos.index.day == 29))
    a_aoi_filtrado = np.clip(a_aoi[mask_sem_bissexto][:n_horas], -1.0, 1.0)
    aoi_graus = np.arccos(a_aoi_filtrado) * 180.0 / np.pi
    zenith_graus = solar_pos["zenith"].to_numpy()[mask_sem_bissexto][:n_horas]

    return aoi_graus, zenith_graus


def simular_geracao_pv(
    cenario_clima: dict,
    aoi_graus: np.ndarray,
    zenith_graus: np.ndarray,
    n_mod: int = 471139,
    pot_nom_w: float = 550.0,
    a_mod: float = 2.584386,
    p_nom_ac_w: float = 190000400.38600048,
    deg_anual_pct: float = 0.55,
    bs: float = 13.27,
) -> dict:
    """
    Calcula a geração fotovoltaica horária (CC e CA pós-inversor com clipping)
    de forma 100% vetorizada. Retorna potências em Watts [W].
    """
    dni = cenario_clima["dni"]
    dhi = cenario_clima["dhi"]
    albedo = cenario_clima["albedo"]
    temp = cenario_clima["temp"]
    vento = cenario_clima["vento_vel"]
    n_passos = len(temp)

    npv_nom = pot_nom_w / (1000.0 * a_mod)
    y_temp_coeff = -0.35 / 100.0
    tc_ref = 25.0
    n_inv = 0.984

    # Perdas CC e CA
    xl1 = 1.0 - 2.0 / 100.0   # Mismatch
    xl2 = 1.0 - 0.5 / 100.0   # Diodos
    xl3 = 1.0 - 2.0 / 100.0   # Cabeamento CC
    xl5 = 1.0 - 1.0 / 100.0   # Cabeamento CA
    fpv_dc = xl1 * xl2 * xl3
    fpv_ac = xl5 * n_inv

    # Irradiâncias no plano do painel (POA)
    aoi_rad = aoi_graus * np.pi / 180.0
    bs_rad = bs * np.pi / 180.0
    z_rad = zenith_graus * np.pi / 180.0

    ib = np.maximum(0.0, 0.95 * dni * np.cos(aoi_rad))
    id_irr = 0.95 * dhi * (1.0 + np.cos(bs_rad)) / 2.0
    ir = 0.95 * albedo * (dni * np.cos(z_rad) + dhi) * (1.0 - np.cos(bs_rad)) / 2.0
    gi = ib + id_irr + ir

    # Temperatura de operação e potência CC sem degradação
    tc = gi * np.exp(-3.56 - 0.075 * vento) + temp + (gi / 1000.0) * 3.0
    npv = npv_nom * (1.0 + y_temp_coeff * (tc - tc_ref))
    pot_dc_sem_deg = n_mod * a_mod * gi * npv * fpv_dc

    # Aplicação da degradação anual e clipping do inversor CA
    anos_transcorridos = np.arange(n_passos) // 8760
    fator_deg_vetor = (1.0 - (deg_anual_pct / 100.0)) ** anos_transcorridos

    p_pv_dc_w = pot_dc_sem_deg * fator_deg_vetor
    cap_inversor_ac_w = p_nom_ac_w * xl5
    p_pv_ac_w = np.where(p_pv_dc_w * n_inv > p_nom_ac_w, cap_inversor_ac_w, p_pv_dc_w * fpv_ac)

    pot_anual_wh = pd.Series(p_pv_ac_w.reshape(-1, 8760).sum(axis=1))

    return {
        "p_pv_dc_w": p_pv_dc_w,
        "p_pv_ac_w": p_pv_ac_w,
        "pot_anual_pv_wh": pot_anual_wh,
    }


# ==============================================================================
# 3. MÓDULO EÓLICO (ADAPTADO PARA SÉRIES EM MEMÓRIA)
# ==============================================================================

def _ler_arquivo_vento_sam(caminho_arquivo: str, hub_height: float, shear: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lê o arquivo meteorológico padrão do SAM (.srw / .txt) com múltiplas alturas,
    interpolando linearmente para 'hub_height' (ou extrapolando via shear se fora da faixa).
    """
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        linhas_cabecalho = [next(f).strip().split(",") for _ in range(5)]

    variaveis = [v.strip() for v in linhas_cabecalho[2]]
    alturas = np.array([float(h) for h in linhas_cabecalho[4]])
    dados = np.loadtxt(caminho_arquivo, delimiter=",", skiprows=5)

    alturas_unicas = sorted(list(set(alturas)))

    def extrair_var_altura(nome_var: str, h_alvo: float) -> np.ndarray:
        for idx, (v, h) in enumerate(zip(variaveis, alturas)):
            if v == nome_var and math.isclose(h, h_alvo):
                return dados[:, idx]
        raise ValueError(f"Variável {nome_var} na altura {h_alvo}m não encontrada.")

    # Se hub_height coincide exatamente com uma das alturas do arquivo (ex: 80 m)
    for h_u in alturas_unicas:
        if math.isclose(hub_height, h_u):
            return (
                extrair_var_altura("Temperature", h_u),
                extrair_var_altura("Pressure", h_u),
                extrair_var_altura("Speed", h_u),
                extrair_var_altura("Direction", h_u),
            )

    # Se hub_height está entre duas alturas medidas (interpolação linear igual ao SAM)
    if alturas_unicas[0] < hub_height < alturas_unicas[-1]:
        h_inf = max(h for h in alturas_unicas if h < hub_height)
        h_sup = min(h for h in alturas_unicas if h > hub_height)
        peso = (hub_height - h_inf) / (h_sup - h_inf)

        temp = extrair_var_altura("Temperature", h_inf) + peso * (extrair_var_altura("Temperature", h_sup) - extrair_var_altura("Temperature", h_inf))
        press = extrair_var_altura("Pressure", h_inf) + peso * (extrair_var_altura("Pressure", h_sup) - extrair_var_altura("Pressure", h_inf))
        vel = extrair_var_altura("Speed", h_inf) + peso * (extrair_var_altura("Speed", h_sup) - extrair_var_altura("Speed", h_inf))
        direcao = extrair_var_altura("Direction", h_inf) + peso * (extrair_var_altura("Direction", h_sup) - extrair_var_altura("Direction", h_inf))
        return temp, press, vel, direcao

    # Extrapolação caso hub_height esteja fora da faixa [50, 140]
    h_ref = alturas_unicas[0] if hub_height < alturas_unicas[0] else alturas_unicas[-1]
    temp = extrair_var_altura("Temperature", h_ref)
    press = extrair_var_altura("Pressure", h_ref)
    vel = extrair_var_altura("Speed", h_ref) * ((hub_height / h_ref) ** shear)
    direcao = extrair_var_altura("Direction", h_ref)
    return temp, press, vel, direcao


def simular_geracao_eolica(
    cenario_clima: dict,
    parametros_json: Optional[str] = None,
    deg_anual_pct: float = 0.0,
    arquivo_sam_srw: Optional[str] = None,
) -> dict:
    """
    Simula a geração horária do parque eólico a partir de 'cenario_clima' ou,
    opcionalmente, direto do arquivo meteorológico do SAM ('arquivo_sam_srw') para validação.
    """
    import json

    if parametros_json and os.path.isfile(parametros_json):
        with open(parametros_json, "r", encoding="utf-8") as f:
            inputs = json.load(f)
    else:
        raise ValueError("O arquivo de parâmetros JSON da usina eólica não foi encontrado.")

    hub_height = inputs["wind_turbine_hub_ht"]
    shear = inputs.get("wind_resource_shear", 0.14)
    n_steps = len(cenario_clima["temp"])

    if arquivo_sam_srw and os.path.isfile(arquivo_sam_srw):
        temp_1a, press_atm_1a, ws_hub_1a, wdir_1a = _ler_arquivo_vento_sam(
            arquivo_sam_srw, hub_height, shear
        )
        n_anos = max(1, n_steps // 8760)
        temp = np.tile(temp_1a[:8760], n_anos)
        pressure_atm = np.tile(press_atm_1a[:8760], n_anos)  # Já está em [atm] no arquivo do SAM
        wind_speed_hub = np.tile(ws_hub_1a[:8760], n_anos)
        wind_dir = np.tile(wdir_1a[:8760], n_anos)
    else:
        temp = cenario_clima["temp"]
        wind_speed10 = cenario_clima["vento_vel"]
        wind_speed_hub = wind_speed10 * ((hub_height / 10.0) ** shear)
        wind_dir = cenario_clima["vento_dir"]
        pressure_atm = cenario_clima["press"] / 1013.25

    # 1. Configurar WindTurbine
    wt = WindTurbine()
    wt.shear_exponent = shear
    wt.hub_height = hub_height
    wt.measurement_height = hub_height
    wt.rotor_diameter = inputs["wind_turbine_rotor_diameter"]
    wt.set_power_curve(
        list(inputs["wind_turbine_powercurve_windspeeds"]),
        list(inputs["wind_turbine_powercurve_powerout"]),
    )
    if "wind_turbine_ct_curve" in inputs:
        wt.set_ct_curve(list(inputs["wind_turbine_ct_curve"]))

    # 2. Configurar WindPowerCalculator
    wpc = WindPowerCalculator()
    wpc.wind_turb = wt
    ti_coeff = inputs["wind_resource_turbulence_coeff"]
    wpc.turbulence_intensity = ti_coeff / 100.0 if ti_coeff > 1.0 else ti_coeff
    wpc.x_coords = np.array(inputs["wind_farm_xCoordinates"], dtype=float)
    wpc.y_coords = np.array(inputs["wind_farm_yCoordinates"], dtype=float)
    wpc.n_turbines = len(wpc.x_coords)

    # 3. Configurar Modelo de Esteira e Perdas
    wake_model_choice = inputs["wind_farm_wake_model"]
    if wake_model_choice == WakeModelOptions.SIMPLE:
        wake_model = SimpleWakeModel(wpc.n_turbines, wt)
    elif wake_model_choice == WakeModelOptions.PARK:
        wake_model = ParkWakeModel(wpc.n_turbines, wt)
    else:
        derate = (100.0 - inputs.get("wake_int_loss", 0.0)) / 100.0
        wake_model = ConstantWakeModel(wpc.n_turbines, wt, derate)

    wpc.initialize_model(wake_model)
    haf = AdjustmentFactors(inputs, "adjust")
    haf.setup(n_steps)
    loss_multiplier = _get_fixed_losses(inputs)

    gen_kw = np.zeros(n_steps)
    power = np.zeros(wpc.n_turbines)
    thrust = np.zeros(wpc.n_turbines)
    eff = np.zeros(wpc.n_turbines)
    ad_ws = np.zeros(wpc.n_turbines)
    ti = np.zeros(wpc.n_turbines)
    dist_down = np.zeros(wpc.n_turbines)
    dist_cross = np.zeros(wpc.n_turbines)

    for i in range(n_steps):
        farm_p, _ = wpc.wind_power_using_resource(
            wind_speed_hub[i], wind_dir[i], pressure_atm[i], temp[i],
            power, thrust, eff, ad_ws, ti, dist_down, dist_cross
        )
        gen_kw[i] = farm_p * loss_multiplier * haf(i)

    anos_transcorridos = np.arange(n_steps) // 8760
    fator_deg_vetor = (1.0 - (deg_anual_pct / 100.0)) ** anos_transcorridos
    p_eolica_w = gen_kw * 1000.0 * fator_deg_vetor

    pot_anual_wh = pd.Series(p_eolica_w.reshape(-1, 8760).sum(axis=1))

    return {
        "p_eolica_w": p_eolica_w,
        "pot_anual_eolica_wh": pot_anual_wh,
    }


# ==============================================================================
# 4. MÓDULO BESS (DESPACHO DE BATERIA INDEPENDENTE DA FONTE)
# ==============================================================================

def _reversals(series):
    series = iter(series)
    x_last, x = next(series, None), next(series, None)
    if x_last is None or x is None:
        return
    d_last = x - x_last
    yield 0, x_last
    index = None
    for index, x_next in enumerate(series, start=1):
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            yield index, x
        x_last, x = x, x_next
        d_last = d_next
    if index is not None:
        yield index + 1, x_next


def _extract_cycles(series):
    points = deque()

    def format_output(point1, point2, count):
        _, x1 = point1
        _, x2 = point2
        return abs(x1 - x2), count

    for point in _reversals(series):
        points.append(point)
        while len(points) >= 3:
            x1, x2, x3 = points[-3][1], points[-2][1], points[-1][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            if X < Y:
                break
            elif len(points) == 3:
                yield format_output(points[0], points[1], 0.5)
                points.popleft()
            else:
                yield format_output(points[-3], points[-2], 1.0)
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)
    else:
        while len(points) > 1:
            yield format_output(points[0], points[1], 0.5)
            points.popleft()


def _voltage_bat(Vo, K, Qfull, it, i, R, A, B):
    return Vo - (R * i) - (K * (Qfull / (Qfull - it))) + (A * np.exp(-B * it))


def _capacidade_bat(x):
    return (-25.0 / 6000.0) * x + 100.0


def simular_despacho_bess(
    p_entrada_ac_w: np.ndarray,
    desejo_horario_w: float = 100e6,
    n_bat: int = 78125,
) -> dict:
    """
    Recebe qualquer curva horária de geração CA (PV, Eólica, CSP ou a soma delas)
    e opera o banco de baterias RB48V10 para tentar manter 'desejo_horario_w'.
    Mantém 100% da matemática original de count_cycles e tensão terminal.
    """
    c_rate = 0.25
    t_drain = 1.0 / c_rate
    q_full = 100.0
    i_abs = abs(q_full / t_drain)
    A = 2.0168
    B = 4.8525
    R = 0.0199
    v_nom = 51.2
    q_nom = 80.0
    soc_max = 100.0
    soc_min = 20.0
    q_inicial = 50.0
    Q = 50.0
    q_max = 100.0
    soc_percent = 100.0 * Q / q_full
    capacidade_troca_bat = 72.2
    v_full = 54.5
    n_inv = 0.984

    K = (v_full - v_nom + A * (np.e ** (-B * q_nom) - 1.0)) * (q_full - q_nom) / q_nom
    Vo = v_full + K + R * i_abs - A

    n_passos = len(p_entrada_ac_w)
    SOC = []
    Carga = []
    Pbat = []
    Corrente = []
    Ciclo = []
    Vbat = []
    CapacidadeBAT = []
    vet_capacidade_bat = []

    v_ciclo = []
    ini_ciclo = 0
    cont_ciclo = 0
    auxj = -1
    reset = 0

    def count_cycles_local(series_slice):
        nonlocal reset, cont_ciclo, v_ciclo
        if reset == 1 and cont_ciclo == 0:
            series_slice[0] = q_inicial
        if cont_ciclo != 0:
            reset = 0

        counts = defaultdict(float)
        for rng, count in _extract_cycles(series_slice):
            counts[rng] += count

        resp = sorted(counts.items())
        resp += v_ciclo
        aux1_dup = [y[0] for y in resp]
        aux2_dup = Counter(aux1_dup)
        flag_dup = 0
        dup_num = None
        for numero, repeticoes in aux2_dup.items():
            if repeticoes > 1:
                dup_num = numero
                flag_dup = 1
        if flag_dup == 1:
            cont_dup = sum(item[1] for item in resp if item[0] == dup_num)
            resp = list(set(resp))
            for idx_r in range(len(resp)):
                if resp[idx_r][0] == dup_num:
                    resp.pop(idx_r)
                    resp.append((dup_num, cont_dup))
            v_ciclo = []

        cont = 0.0
        flag = 0
        for r in resp:
            if r[1] >= 1:
                flag = 1
                if r[1] % 1 == 0.5:
                    cont += r[1] - 0.5
                else:
                    cont += r[1]
        if flag == 1:
            v_ciclo = []
            for r in resp:
                if r[1] < 1:
                    v_ciclo.append(r)
                if r[1] > 1 and r[1] % 1 != 0:
                    v_ciclo.append((r[0], 0.5))
        return cont

    for j in range(n_passos):
        auxj += 1
        saldo_energia = p_entrada_ac_w[j] - desejo_horario_w

        if saldo_energia < 0:
            # --- MODO DESCARGA ---
            p_out_bateria = -saldo_energia
            i = i_abs
            Qc = q_full - Q
            V = _voltage_bat(Vo, K, q_full, Qc, i, R, A, B)
            Vbat.append(V)

            if i > 0 and soc_percent != soc_min:
                I = p_out_bateria / n_inv / V / n_bat
                if I > i:
                    I = i
                Qdis = I
                soc_percent = (Q - Qdis) * 100.0 / q_max

                if Q - I < soc_min * _capacidade_bat(cont_ciclo) / 100.0:
                    Qdis = Q - soc_min * (q_max / 100.0)
                    I = Qdis
                    soc_percent = soc_min

                Q -= Qdis
                Pbat.append(I * V * n_bat * n_inv)
                SOC.append(soc_percent)
                Corrente.append(I)

                aux_ciclo = count_cycles_local(SOC[ini_ciclo : auxj + 1])
                if aux_ciclo > 0:
                    ini_ciclo = auxj
                    cont_ciclo += aux_ciclo
                    q_max = (_capacidade_bat(cont_ciclo) / 100.0) * q_full
                    aux_ciclo = 0
                    Q -= (25.0 / 6000.0) * aux_ciclo

                Carga.append(Q)
                Ciclo.append(cont_ciclo)
                CapacidadeBAT.append(q_max)

                if q_max <= capacidade_troca_bat:
                    vet_capacidade_bat.append((j + 1) / 8760.0)
                    Q, soc_percent = 50.0, 50.0
                    v_ciclo, cont_ciclo = [], 0
                    q_max = 100.0
            else:
                Pbat.append(0.0)
                SOC.append(soc_percent)
                Carga.append(Q)
                Corrente.append(0.0)
                Ciclo.append(cont_ciclo)
                CapacidadeBAT.append(q_max)

        elif saldo_energia > 0:
            # --- MODO RECARGA ---
            p_in_bateria = saldo_energia
            i = -i_abs
            Qc = q_full - Q
            V = _voltage_bat(Vo, K, q_full, Qc, i, R, A, B)
            Vbat.append(V)

            if i < 0 and soc_percent != soc_max:
                I = -p_in_bateria * n_inv / V / n_bat
                if I < i:
                    I = i
                Qchar = I
                soc_percent = (Q - Qchar) * 100.0 / q_max

                if Q - I > soc_max * _capacidade_bat(cont_ciclo) / 100.0:
                    Qchar = -(soc_max * q_max / 100.0 - Q)
                    I = Qchar
                    soc_percent = soc_max

                Q -= Qchar
                Pbat.append(I * V * n_bat * n_inv)
                SOC.append(soc_percent)
                Corrente.append(I)

                aux_ciclo = count_cycles_local(SOC[ini_ciclo : auxj + 1])
                if aux_ciclo > 0:
                    ini_ciclo = auxj
                    cont_ciclo += aux_ciclo
                    q_max = (_capacidade_bat(cont_ciclo) / 100.0) * q_full
                    aux_ciclo = 0
                    Q -= (25.0 / 6000.0) * aux_ciclo

                Carga.append(Q)
                Ciclo.append(cont_ciclo)
                CapacidadeBAT.append(q_max)

                if q_max <= capacidade_troca_bat:
                    vet_capacidade_bat.append((j + 1) / 8760.0)
                    Q, soc_percent = 50.0, 50.0
                    v_ciclo, cont_ciclo = [], 0
                    q_max = 100.0
            else:
                Pbat.append(0.0)
                SOC.append(soc_percent)
                Carga.append(Q)
                Corrente.append(0.0)
                Ciclo.append(cont_ciclo)
                CapacidadeBAT.append(q_max)
        else:
            # --- MODO REPOUSO ---
            Qc = q_full - Q
            V = _voltage_bat(Vo, K, q_full, Qc, i_abs, R, A, B)
            Vbat.append(V)
            Pbat.append(0.0)
            SOC.append(soc_percent)
            Carga.append(Q)
            Corrente.append(0.0)
            Ciclo.append(cont_ciclo)
            CapacidadeBAT.append(q_max)

    p_bat_w = np.array(Pbat, dtype=float)
    p_saida_rede_w = p_entrada_ac_w + p_bat_w
    pot_anual_wh = pd.Series(p_saida_rede_w.reshape(-1, 8760).sum(axis=1))

    return {
        "p_bat_w": p_bat_w,
        "p_saida_rede_w": p_saida_rede_w,
        "pot_anual_hibrido_wh": pot_anual_wh,
        "soc": np.array(SOC),
        "trocas_bateria_anos": vet_capacidade_bat,
    }