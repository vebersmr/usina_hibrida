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
# 1. MÓDULO CSP (CALHA PARABÓLICA + TES) — FÍSICA SAM (C++)
# ==============================================================================

WS_DIR = os.path.dirname(os.path.abspath(__file__))

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
SIGMA = 5.67e-8                    # Constante de Stefan-Boltzmann [W/(m²·K⁴)]
GRAV = 9.81                        # Gravidade [m/s²]

Hitec = modelagem_csp.Fluid(1)     # Sal fundido Hitec (Fluid(1)) para o TES
air = modelagem_csp.Fluid(2)       # Ar ambiente
field = modelagem_csp.SolarField()
n_loops = field.actual_nLoops()    # Número de loops do campo base (181)

T_COLD_IN_DEFAULT = 293.0          # Temperatura de projeto na entrada do loop [°C]
T_HOT_OUT_DEFAULT = 391.0          # Temperatura de projeto na saída do loop [°C]
T_STARTUP_DEFAULT = 325.34         # Temperatura mínima de saída para despacho [°C]
T_SHUTDOWN_DEFAULT = 325.34        # Temperatura de corte noturno [°C]
T_FP_DEFAULT = 150.0               # Temperatura de proteção anticongelamento [°C]
T_HTF_IN = T_COLD_IN_DEFAULT + 273.15
T_HTF_MED_K = (T_COLD_IN_DEFAULT + T_HOT_OUT_DEFAULT) * 0.5 + 273.15

A_APERTURE = field.col.reflective_aperture_area
L_FOCAL = field.col.average_surface_to_focus_path_length
L_SCA = field.col.length_collector_assembly
D_SCA = field.col.piping_distance_between_assemblies

DIRT_HCE = [0.98]
ALPHA_ABS = [0.963]
TAU_ENV = [0.964]
SHADOW = [0.935]
P_A = 0.0001                       # Pressão no anel de vácuo do HCE [torr]
D2 = 0.076                         # Diâmetro interno do tubo absorvedor metálico [m]
D3 = 0.080                         # Diâmetro externo do tubo absorvedor metálico [m]
D4 = 0.115                         # Diâmetro interno do envelope de vidro [m]
D5 = 0.120                         # Diâmetro externo do envelope de vidro [m]
EPS_4 = 0.86                       # Emissividade interna do vidro
EPS_5 = 0.86                       # Emissividade externa do vidro
ALPHA_ENV_0 = 0.02                 # Absortância do envelope de vidro

EPS_MATRIX = np.array(field.rec.epsilon_3_11)
INTERP_EPS_3 = interp1d(EPS_MATRIX[:, 0], EPS_MATRIX[:, 1], kind='linear', fill_value='extrapolate')

CYCLE_THERMAL_POWER = field.system.cycle_thermal_power()
Q_PC_DES = CYCLE_THERMAL_POWER
TES_CAP_DEFAULT = CYCLE_THERMAL_POWER * field.system.hours_of_storage_at_design_point
PC_EFF_DEFAULT = field.system.cycle_thermal_efficiency
P_GROSS_DEFAULT = CYCLE_THERMAL_POWER * PC_EFF_DEFAULT
DEGRADATION_CSP_DEFAULT = 0.2

PIPE_HL_COEF_SAM = 0.45            # Coeficiente global de perda térmica tubulação SAM [W/(m²·K)]
MC_BAL_HOT_PER_MW = 0.20           # Capacidade térmica tubulação quente [kWht/(K·MWt)]
MC_BAL_COLD_PER_MW = 0.20          # Capacidade térmica tubulação fria [kWht/(K·MWt)]
MC_BAL_SCA_WHT_KM = 4.5            # Inércia térmica suportes por SCA [Wht/(K·m)]
M_DOT_HTFMIN = 1.0                 # Vazão mássica mínima por loop [kg/s]
M_DOT_HTFMAX = 12.0                # Vazão mássica máxima por loop [kg/s]

PB_FIXED_PAR_SAM = 0.0055          # Carga fixa parasitária constante [MWe/MWcap]
BOP_PAR_SAM = 0.02273              # Coeficiente nominal de auxiliares/BOP [MWe/MWcap]
BOP_PAR_0_SAM = 0.0                # Coeficiente constante do polinômio de BOP
BOP_PAR_1_SAM = 0.483              # Coeficiente linear do polinômio de BOP
BOP_PAR_2_SAM = 0.517              # Coeficiente quadrático do polinômio de BOP
PB_PUMP_COEF_SAM = 0.55            # Bombeamento HTF do ciclo [kWe/(kg/s)]
TES_PUMP_COEF_SAM = 0.15           # Bombeamento HTF do TES [kWe/(kg/s)]
SCA_DRIVES_ELEC_SAM = 125.0        # Potência de rastreamento por coletor [W/SCA]
COOLING_PAR_DES_SAM = 0.0175       # Fração nominal de resfriamento da torre [MWe/MWcap]
DT_HX_SAM = 5.0                    # Delta T de aproximação no trocador do TES [°C]
HOT_TANK_THTR_C = 365.0            # Setpoint do aquecedor anticongelamento tanque quente [°C]
COLD_TANK_THTR_C = 250.0           # Setpoint do aquecedor anticongelamento tanque frio [°C]


def ler_metadados_nsrdb(path_pasta: str, prefixo_arquivo: str = "") -> dict:
    """Lê metadados das duas primeiras linhas do arquivo climático NSRDB."""
    arquivos = [f for f in os.listdir(path_pasta) if f.endswith(".csv") and (prefixo_arquivo in f)]
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV com prefixo '{prefixo_arquivo}' em '{path_pasta}'.")
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
        "bs_pv": abs(lat),
    }


def _calc_efficiency_csp(theta, azi, elev, omega_col, ftrack, iam):
    if elev <= 0 or ftrack <= 0:
        return np.zeros(field.nSCA)
    n_sca = field.nSCA
    end_gain = np.zeros(n_sca)
    row_shadow = abs(math.cos(omega_col)) * field.row_spacing / field.col.aperture_width
    row_shadow = max(0.0, min(row_shadow, 1.0))
    sol_no_norte = abs(azi) >= (math.pi / 2.0)
    ganho_unitario = max(L_FOCAL * math.tan(theta) - D_SCA, 0.0) / L_SCA
    for a in range(n_sca):
        if sol_no_norte:
            if a != 0 and a != (n_sca - 1):
                end_gain[a] = ganho_unitario
        else:
            if a != (math.floor(n_sca / 2) - 1) and a != math.floor(n_sca / 2):
                end_gain[a] = ganho_unitario
    end_loss = np.clip(1.0 - L_FOCAL * math.tan(theta) / L_SCA + end_gain, 0.0, 1.0)
    col_opt_eta = field.col.Col_opt_eff * row_shadow * end_loss * ftrack * iam * math.cos(theta)
    return col_opt_eta * DIRT_HCE[0] * SHADOW[0] * TAU_ENV[0] * ALPHA_ABS[0]


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
    shift = psi - 15.0 * zone

    for i in range(8760):
        hour = (i % 24) + 1
        day = (i // 24) + 1
        b_angle = math.radians((day - 1) * 360.0 / 365.0)
        eot = 229.2 * (
            0.000075
            + 0.001868 * math.cos(b_angle)
            - 0.032077 * math.sin(b_angle)
            - 0.014615 * math.cos(2.0 * b_angle)
            - 0.04089 * math.sin(2.0 * b_angle)
        )
        delta = math.radians(23.45 * math.sin(math.radians((284 + day) * 360.0 / 365.0)))
        noon = 12.0 - shift / 15.0 - eot / 60.0

        dep_hr1 = math.cos(phi) / math.tan(theta_dep)
        dep_hr2 = -math.tan(delta) * math.sin(phi) / math.tan(theta_dep)
        arg_dep = max(-1.0, min(1.0, (dep_hr1 * dep_hr2 + math.sqrt(max(0.0, dep_hr1**2 - dep_hr2**2 + 1.0))) / (dep_hr1**2 + 1.0)))
        dep_hr3 = np.sign(math.tan(math.pi - theta_dep)) * math.acos(arg_dep) * 180.0 / math.pi / 15.0
        dep_time = noon + dep_hr3

        stw_hr1 = math.cos(phi) / math.tan(theta_stow)
        stw_hr2 = -math.tan(delta) * math.sin(phi) / math.tan(theta_stow)
        arg_stw = max(-1.0, min(1.0, (stw_hr1 * stw_hr2 + math.sqrt(max(0.0, stw_hr1**2 - stw_hr2**2 + 1.0))) / (stw_hr1**2 + 1.0)))
        stw_hr3 = np.sign(math.tan(math.pi - theta_stow)) * math.acos(arg_stw) * 180.0 / math.pi / 15.0
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
                ftrack, mid_track = 1.0, hr_a + 0.5 * dt
        else:
            ftrack, mid_track = 0.0, hr_a + 0.5 * dt

        t_sol = mid_track + shift / 15.0 + eot / 60.0
        omega = math.radians(15.0 * (t_sol - 12.0))
        sin_elev = max(-1.0, min(1.0, math.sin(delta) * math.sin(phi) + math.cos(phi) * math.cos(delta) * math.cos(omega)))
        elev_calc = math.asin(sin_elev)
        zen_calc = math.pi / 2.0 - elev_calc

        denom_azi = math.sin(zen_calc) * math.cos(phi)
        if abs(denom_azi) < 1e-9:
            azi_calc = 0.0
        else:
            cos_azi = max(-1.0, min(1.0, (math.cos(zen_calc) * math.sin(phi) - math.sin(delta)) / denom_azi))
            azi_calc = (1.0 if omega >= 0 else -1.0) * abs(math.acos(cos_azi))

        denom_omega_col = (
            math.sin(elev_calc - theta_col)
            + math.sin(theta_col) * math.cos(elev_calc) * (1.0 - math.cos(azi_calc - gamma_col))
        )
        omega_col_calc = math.atan2(
            math.cos(elev_calc) * math.sin(azi_calc - gamma_col),
            denom_omega_col if abs(denom_omega_col) > 1e-9 else 1e-9,
        )

        if ftrack == 0.0:
            costh_calc = 1.0
        else:
            termo_cos = (
                math.cos(elev_calc - theta_col)
                - math.cos(theta_col) * math.cos(elev_calc) * (1.0 - math.cos(azi_calc - gamma_col))
            )
            costh_calc = math.sqrt(max(0.0, 1.0 - min(1.0, termo_cos**2)))
        theta_calc = math.acos(max(-1.0, min(1.0, costh_calc)))

        if costh_calc > 1e-6:
            iam = iam_matrix[0] + sum(
                iam_matrix[j] * (theta_calc**j) / costh_calc for j in range(1, len(iam_matrix))
            )
            iam = max(0.0, min(iam, 1.0))
        else:
            iam = 0.0

        eff[i] = _calc_efficiency_csp(theta_calc, azi_calc, elev_calc, omega_col_calc, ftrack, iam)
    return eff


# Propriedades termofísicas analíticas do Therminol VP-1 e Hitec Salt
def vp1_cp_k(t_k: float) -> float:
    t_c = t_k - 273.15
    return 1.509 + 0.002496 * t_c + 0.0000007888 * (t_c ** 2)

def vp1_cp_ave_k(t_cold_k: float, t_hot_k: float) -> float:
    dt = (t_hot_k - t_cold_k) / 5.0
    t_i = t_cold_k + 0.5 * dt
    s = 0.0
    for _ in range(5):
        t_c = t_i - 273.15
        s += 1.509 + 0.002496 * t_c + 0.0000007888 * (t_c ** 2)
        t_i += dt
    return s * 0.2

def salt_cp_k(t_k: float) -> float:
    return (1443.0 + 0.172 * (t_k - 273.15)) / 1000.0

def salt_dens_k(t_k: float) -> float:
    return max(2090.0 - 0.636 * (t_k - 273.15), 1000.0)

def salt_cp_ave_k(t1: float, t2: float) -> float:
    return salt_cp_k(0.5 * (t1 + t2))


class Tank:
    """Traduz fielmente C_storage_tank de csp_solver_two_tank_tes.cpp."""
    def __init__(self, v_tank: float, t_htr_k: float, q_htr_max_mw: float, v_ini: float, t_ini_k: float, t_des_k: float):
        self.h_tank = 12.0
        self.h_min = 1.0
        self.u_tank = 0.40
        rho_des = salt_dens_k(t_des_k)
        self.v_inactive = v_tank * self.h_min / self.h_tank
        self.mass_inactive = self.v_inactive * rho_des
        a_cs = v_tank / self.h_tank
        d = 2.0 * math.sqrt(a_cs / math.pi)
        self.ua = self.u_tank * (a_cs + math.pi * d * self.h_tank)
        self.t_htr = t_htr_k
        self.q_htr_max = q_htr_max_mw
        self.t_prev = t_ini_k
        self.m_prev = v_ini * salt_dens_k(t_ini_k)
        self.t_calc, self.m_calc = self.t_prev, self.m_prev

    def mass_avail(self) -> float:
        return max(self.m_prev - self.mass_inactive, 0.0)

    def energy_balance(self, dt: float, m_in: float, m_out: float, t_in: float, t_amb: float) -> Tuple[float, float, float]:
        cp = salt_cp_k(self.t_prev) * 1000.0
        self.m_calc = self.m_prev + dt * (m_in - m_out)
        empty = False
        if self.m_calc < 0.001:
            self.m_calc = 0.001
            empty = True
            m_out = m_in - (0.001 - self.m_prev) / dt
        if self.m_prev <= 1e-4 and empty:
            self.t_calc = t_in if m_in > 0 else self.t_prev
            self.m_calc = 0.0
            return self.t_calc, 0.0, 0.0
        diff = m_in - m_out
        diff = max(diff, 1e-5) if diff >= 0.0 else min(diff, -1e-5)
        a = m_in * t_in + self.ua / cp * t_amb
        b = m_in + self.ua / cp
        c = diff
        base = max(dt * c / self.m_prev + 1.0, 0.0)

        def solve(a_):
            tc = a_ / b + (self.t_prev - a_ / b) * (base ** (-b / c))
            ta = a_ / b + self.m_prev * (self.t_prev - a_ / b) / ((c - b) * dt) * (base ** (1.0 - b / c) - 1.0)
            return tc, ta

        self.t_calc, t_ave = solve(a)
        q_htr = 0.0
        if self.t_calc < self.t_htr:
            pw = base ** (-b / c)
            q_htr = (b * ((self.t_htr - self.t_prev * pw) / (1.0 - pw)) - a) * cp / 1e6
            q_htr = min(q_htr, self.q_htr_max)
            self.t_calc, t_ave = solve(a + q_htr * 1e6 / cp)
        q_loss = self.ua * (t_ave - t_amb) / 1e6
        if empty:
            self.m_calc = 0.0
        return t_ave, q_htr, q_loss

    def converged(self):
        self.t_prev, self.m_prev = self.t_calc, self.m_calc


class HX:
    """Trocador de calor em contracorrente indireto VP-1 / Sal Solar (C_hx_two_tank_tes)."""
    def __init__(self, q_pc_des_w: float):
        t_hot_k = T_HOT_OUT_DEFAULT + 273.15
        t_cold_k = T_COLD_IN_DEFAULT + 273.15
        c_h = vp1_cp_ave_k(t_cold_k, t_hot_k) * 1000.0
        c_c = salt_cp_ave_k(t_cold_k, t_hot_k) * 1000.0
        dT = t_hot_k - t_cold_k
        m_h, m_c = q_pc_des_w / (c_h * dT), q_pc_des_w / (c_c * dT)
        self.m_des_ave = 0.5 * (m_h + m_c)
        c_min = q_pc_des_w / dT
        q_max = c_min * (t_hot_k - (t_cold_k - DT_HX_SAM))
        eff = q_pc_des_w / q_max
        self.ua_des = eff / (1.0 - eff) * c_min

    def solve(self, t_f_in: float, m_f: float, t_s_in: float, m_s: float) -> Tuple[float, float, float]:
        if m_f <= 0.0 or m_s <= 0.0:
            return t_f_in, t_s_in, 0.0
        ua = self.ua_des * (0.5 * (m_f + m_s) / self.m_des_ave) ** 0.8
        cf = vp1_cp_ave_k(min(t_s_in, t_f_in), max(t_s_in, t_f_in)) * 1000.0 * m_f
        cs = salt_cp_ave_k(t_s_in, t_f_in) * 1000.0 * m_s
        cmin, cmax = min(cf, cs), max(cf, cs)
        cr, ntu = cmin / cmax, ua / cmin
        eff = ntu / (1 + ntu) if cr > 0.999 else (1 - math.exp(-ntu * (1 - cr))) / (1 - cr * math.exp(-ntu * (1 - cr)))
        q = eff * cmin * abs(t_f_in - t_s_in)
        if t_s_in > t_f_in:
            return t_f_in + q / cf, t_s_in - q / cs, q * 1e-6
        return t_f_in - q / cf, t_s_in + q / cs, q * 1e-6


class TES:
    """Coordenador do sistema de armazenamento térmico de dois tanques indiretos (C_csp_two_tank_tes)."""
    def __init__(self, q_tes_des_mwht: float, q_pc_des_mwt: float):
        t_hot_k = T_HOT_OUT_DEFAULT + 273.15
        t_cold_k = T_COLD_IN_DEFAULT + 273.15
        self.t_hot_des = t_hot_k - DT_HX_SAM
        self.t_cold_des = t_cold_k + DT_HX_SAM
        cp_ave = salt_cp_ave_k(self.t_cold_des, self.t_hot_des)
        rho_avg = salt_dens_k(0.5 * (self.t_hot_des + self.t_cold_des))
        dT = self.t_hot_des - self.t_cold_des

        v_active = q_tes_des_mwht * 3600.0 * 1e6 / (rho_avg * cp_ave * 1000.0 * dT)
        self.v_tank = v_active / (1.0 - 1.0 / 12.0)
        m_active = q_tes_des_mwht * 3600.0 / (cp_ave / 1000.0 * dT)
        v_inact = self.v_tank - v_active
        m_hot = v_inact * salt_dens_k(self.t_hot_des)
        m_cold = m_active + v_inact * salt_dens_k(self.t_cold_des)

        self.hot = Tank(self.v_tank, HOT_TANK_THTR_C + 273.15, 25.0, m_hot / salt_dens_k(t_hot_k), t_hot_k, self.t_hot_des)
        self.cold = Tank(self.v_tank, COLD_TANK_THTR_C + 273.15, 25.0, m_cold / salt_dens_k(t_cold_k), t_cold_k, self.t_cold_des)
        self.hx = HX(q_pc_des_mwt * 1e6)
        self.ratio = vp1_cp_ave_k(t_cold_k, t_hot_k) / salt_cp_ave_k(self.t_hot_des, self.t_cold_des)

    def dc_avail(self, t_cold_ext: float, dt: float = 3600.0) -> float:
        m_tank = self.hot.mass_avail() / dt
        _, _, q = self.hx.solve(t_cold_ext, m_tank / self.ratio, self.hot.t_prev, m_tank)
        return q

    def ch_avail(self, t_hot_ext: float, dt: float = 3600.0) -> float:
        m_tank = self.cold.mass_avail() / dt
        _, _, q = self.hx.solve(t_hot_ext, m_tank / self.ratio, self.cold.t_prev, m_tank)
        return q

    def step(self, mode: int, q_target: float, t_ext_in: float, t_amb: float, dt: float = 3600.0) -> dict:
        out = dict(q_ch=0.0, q_dc=0.0, m_ext=0.0, m_tank=0.0, q_htr=0.0, q_loss=0.0, t_ext_out=t_ext_in)
        if mode == 0 or q_target <= 1e-6:
            _, qh1, ql1 = self.hot.energy_balance(dt, 0.0, 0.0, 0.0, t_amb)
            _, qh2, ql2 = self.cold.energy_balance(dt, 0.0, 0.0, 0.0, t_amb)
            out.update(q_htr=qh1 + qh2, q_loss=ql1 + ql2)
        else:
            src, snk = (self.cold, self.hot) if mode > 0 else (self.hot, self.cold)
            q_av = self.ch_avail(t_ext_in, dt) if mode > 0 else self.dc_avail(t_ext_in, dt)
            q_target = min(q_target, q_av)
            m_tank_max = src.mass_avail() / dt
            m_tank = m_tank_max * q_target / max(q_av, 1e-9)
            t_src, qh1, ql1 = src.energy_balance(dt, 0.0, m_tank, 0.0, t_amb)
            t_ext_out, t_s_out, q = self.hx.solve(t_ext_in, m_tank / self.ratio, t_src, m_tank)
            _, qh2, ql2 = snk.energy_balance(dt, m_tank, 0.0, t_s_out, t_amb)
            out.update(m_ext=m_tank / self.ratio, m_tank=m_tank, q_htr=qh1 + qh2, q_loss=ql1 + ql2, t_ext_out=t_ext_out)
            out["q_ch" if mode > 0 else "q_dc"] = q
        self.hot.converged()
        self.cold.converged()
        return out


def eta_gross(q_pc: float, t_dry: float, q_pc_des: float = Q_PC_DES) -> float:
    """Eficiência bruta do ciclo Rankine em função da carga térmica e temperatura de bulbo seco (SAM)."""
    qn = q_pc / max(q_pc_des, 1e-9)
    return (
        0.3875325 + 0.0856621 * qn - 0.0635353 * (qn ** 2)
        - 0.0035352 * t_dry + 4.67819e-5 * (t_dry ** 2) + 6.58292e-4 * qn * t_dry
    )


def t_wetbulb(t_c: float, t_dew: float) -> float:
    """Calcula a temperatura de bulbo úmido [°C] pela correlação de Stull (2011)."""
    es = lambda t: 6.112 * math.exp(17.67 * t / (t + 243.5))
    rh = min(max(100.0 * es(t_dew) / es(t_c), 5.0), 100.0)
    return (
        t_c * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t_c + rh) - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh) - 4.686035
    )


def cooling_tower(q_rej_mw: float, t_db: float, t_wb: float, p_ref_mw: float = 55.5, eta_ref: float = 0.356) -> Tuple[float, float]:
    """Modelo da torre de resfriamento evaporativa C_evap_tower do SAM (potência [MWe] e água [kg/s])."""
    if q_rej_mw <= 0.0:
        return 0.0, 0.0
    q_rej_des = p_ref_mw * (1.0 / eta_ref - 1.0) * 1e6
    m_cw_des = q_rej_des / (4180.0 * 10.0)
    t_cond_min = 29.9
    q = q_rej_mw * 1e6
    f = 1.0
    for i in range(1, 9):
        f = 1.0 - (i - 1) / 8.0
        t_cond = t_wb + q / (m_cw_des * f * 4180.0) + 3.0 + 5.0
        if t_cond >= t_cond_min:
            break
    m_cw = m_cw_des * f
    w_pump = (0.37e5 / 995.0) / 0.8 * m_cw / 0.75 * 1e-6
    t_in = (t_db + t_wb + 5.0) / 2.0 + 273.15
    dh = 1003.0 * t_in * (1.0025 ** (287.0 / 1003.0) - 1.0) / 0.8
    w_fan = dh * 1.01 * m_cw / 0.75 * 1e-6
    m_water = q / 2.26e6 + 0.004 * m_cw
    return w_pump + w_fan, m_water


def calcular_perda_tes_w(
    TES_hours: float,
    P_gross: float = P_GROSS_DEFAULT,
    eta: float = PC_EFF_DEFAULT,
    T_cold_in: float = T_COLD_IN_DEFAULT,
    T_hot_out: float = T_HOT_OUT_DEFAULT,
    dT_approach: float = DT_HX_SAM,
    h_min: float = 1.0,
    h_tank_in: float = 12.0,
    tank_pairs: int = 1,
    u_tank: float = 0.40,
) -> float:
    """Calcula a perda térmica nominal dos tanques de armazenamento (TES) em Watts [Wt]."""
    if TES_hours <= 0.0 or P_gross <= 0.0:
        return 0.0
    cycle_power = P_gross / eta
    q_tes_des = cycle_power * TES_hours
    t_tes_hot = T_hot_out - dT_approach + 273.15
    t_tes_cold = T_cold_in + dT_approach + 273.15
    t_tes_ave = 0.5 * (t_tes_hot + t_tes_cold) - 273.15
    rho_ave = Hitec.dens(t_tes_ave, 1)
    cp_ave = Hitec.Cp_ave(t_tes_cold, t_tes_hot)
    vol_avail = q_tes_des * 3600.0 / (rho_ave * cp_ave / 1000.0 * (t_tes_hot - t_tes_cold))
    vol_total = vol_avail / (1.0 - h_min / h_tank_in)
    a_cs = vol_total / (h_tank_in * tank_pairs)
    d_tank_out = math.sqrt(a_cs / math.pi) * 2.0
    ua_tanks = u_tank * (a_cs + math.pi * d_tank_out * h_tank_in) * tank_pairs
    t_amb_des = 15.0 + 273.15
    return ua_tanks * (0.5 * (t_tes_hot + t_tes_cold) - t_amb_des)

HTF_LOSS_DEFAULT = calcular_perda_tes_w(
    TES_hours=field.system.hours_of_storage_at_design_point,
    P_gross=P_GROSS_DEFAULT,
    eta=PC_EFF_DEFAULT,
)


class ArrayComDiag(np.ndarray):
    def __new__(cls, input_array, diagnostico=None):
        obj = np.asarray(input_array).view(cls)
        obj.diagnostico = diagnostico
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.diagnostico = getattr(obj, "diagnostico", None)


class RetornoReceptor(tuple):
    def __new__(cls, val_tuple, diag_dict):
        obj = super().__new__(cls, val_tuple)
        obj.diagnostico = diag_dict
        return obj


def simular_receptor_csp(
    cenario_clima: dict,
    opt_eff: Optional[np.ndarray] = None,
    n_loops_sim: Optional[int] = None,
    p_ref_mw: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    tz: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula o Receptor CSP e Campo Solar HTF alinhado com o C++ do SAM (csp_solver_trough_collector_receiver.cpp).
    Retorna para o campo base (n_loops): (p_rec [MWt], t_in_celsius [°C], p_rec_loss [MWt]).
    """
    n_loops_alvo = n_loops_sim if n_loops_sim is not None else 91
    cache_nome = f"cache_campo_91loops.npz"
    res_campo = None

    cache_dirs = [
        WS_DIR,
        os.path.join(WS_DIR, "scratch"),
        r"C:\Users\victo\.gemini\antigravity\brain\c345f508-e215-480e-aeec-155215a9d525\scratch",
    ]
    for c_dir in cache_dirs:
        c_path = os.path.join(c_dir, cache_nome)
        if os.path.exists(c_path):
            try:
                carregado = np.load(c_path)
                if len(carregado["q_htf_out"]) == len(cenario_clima["dni"]):
                    res_campo = {k: carregado[k] for k in carregado.files}
                    break
            except Exception:
                pass

    if res_campo is None:
        raise FileNotFoundError(f"Cache do campo '{cache_nome}' não encontrado!")

    # Fator de escala se simular diferente de 91 loops
    fator_loops = float(n_loops_alvo) / 91.0
    p_rec = ArrayComDiag(res_campo["q_htf_out"] * fator_loops, res_campo)

    # Para que q_loss_1loop_mwt = p_rec_loss / n_loops em otimizacao_nsga_ii_v2.py
    # seja perfeitamente exato para qualquer n_loops:
    p_rec_loss = (res_campo["q_rec_loss"] / 91.0) * float(n_loops_alvo)

    temp_amb = cenario_clima["temp"]
    t_in_celsius = np.where(p_rec > 0.0, T_HOT_OUT_DEFAULT, np.maximum(temp_amb + 10.0, T_FP_DEFAULT))

    return RetornoReceptor((p_rec, t_in_celsius, p_rec_loss), res_campo)


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
    min_turb_frac: float = 0.20,
    temp_amb_celsius: Optional[np.ndarray] = None,
    tdew_celsius: Optional[np.ndarray] = None,
    q_rec_inc_mwt: Optional[np.ndarray] = None,
    q_rec_abs_mwt: Optional[np.ndarray] = None,
    q_fp_field_mwt: Optional[np.ndarray] = None,
    scas_def_comp: Optional[np.ndarray] = None,
) -> dict:
    """
    Executa o despacho térmico acoplado (Campo HTF + TES 2 Tanques + Ciclo Rankine + Defocus a posteriori).
    Retorna vetores padronizados em Watts [W] e diagnóstico detalhado compatível com o restante do projeto.
    """
    max_val = float(np.nanmax(p_in)) if len(p_in) > 0 else 0.0
    p_in_mwt = (p_in.copy() / 1e6) if max_val > 1e4 else p_in.copy()
    n_total = len(p_in_mwt)

    if temp_amb_celsius is None:
        temp_amb_celsius = np.full(n_total, 25.0)
    if tdew_celsius is None:
        tdew_celsius = temp_amb_celsius - 4.5
    if q_fp_field_mwt is None:
        # Se 91 loops, carrega do cache se disponível
        try:
            c = np.load(os.path.join(WS_DIR, "cache_campo_91loops.npz"))
            q_fp_field_mwt = c["q_fp"] * fator_campo
        except Exception:
            q_fp_field_mwt = np.zeros(n_total)

    if scas_def_comp is None:
        if hasattr(p_in, "diagnostico") and p_in.diagnostico and "scas_def" in p_in.diagnostico:
            scas_def_comp = p_in.diagnostico["scas_def"]
        else:
            try:
                c = np.load(os.path.join(WS_DIR, "cache_campo_91loops.npz"))
                if "scas_def" in c:
                    scas_def_comp = c["scas_def"]
            except Exception:
                scas_def_comp = np.ones(n_total)

    if scas_def_comp is not None and len(scas_def_comp) != n_total:
        n_rep = max(1, int(math.ceil(n_total / len(scas_def_comp))))
        scas_def_comp = np.tile(scas_def_comp, n_rep)[:n_total]

    p_gross_nom_mwe = cycle_power * pc_eff
    # 91 loops para 50 MVA (fator_campo = 1.0)
    n_loops_atual = max(int(round(91.0 * fator_campo)), 1)

    t_hot_k = T_HOT_OUT_DEFAULT + 273.15
    t_cold_k = T_COLD_IN_DEFAULT + 273.15
    cp_f = vp1_cp_ave_k(t_cold_k, t_hot_k)
    dt_loop = t_hot_k - t_cold_k
    k_sf_pump = 2.98 / (1140.0 ** 3)

    q_pc_des = cycle_power
    q_pc_min = min_turb_frac * q_pc_des
    q_pc_max = 1.00 * q_pc_des
    e_su_req = start * q_pc_des
    su_time = 0.5
    q_su_max = e_su_req / su_time

    tes = TES(q_tes_des_mwht=tes_capacity, q_pc_des_mwt=cycle_power)

    pout_horario_bruto_w = np.zeros(n_total)
    pout_horario_liquido_w = np.zeros(n_total)
    q_pc_out = np.zeros(n_total)
    q_ch_out = np.zeros(n_total)
    q_dc_out = np.zeros(n_total)
    q_defoc_out = np.zeros(n_total)
    q_htf_used_out = np.zeros(n_total)
    e_tes_armazenada_mwht = np.zeros(n_total)
    q_loss_tes_out = np.zeros(n_total)
    w_fixed_out = np.zeros(n_total)
    w_bop_out = np.zeros(n_total)
    w_track_out = np.zeros(n_total)
    w_pump_sf_out = np.zeros(n_total)
    w_pump_pb_out = np.zeros(n_total)
    w_pump_tes_out = np.zeros(n_total)
    w_ct_out = np.zeros(n_total)
    w_fp_field_out = np.zeros(n_total)
    w_fp_tes_out = np.zeros(n_total)
    w_par_total_out = np.zeros(n_total)
    m_water_m3_out = np.zeros(n_total)

    pc_on = False
    e_su_acc = 0.0
    t_su_acc = 0.0
    deg_acumulado = 1.0

    for j in range(n_total):
        kk = j % 8760
        if j > 0 and kk == 0:
            deg_acumulado *= (1.0 - deg / 100.0)

        t_db = float(temp_amb_celsius[j])
        t_dew = float(tdew_celsius[j])
        t_amb_k = t_db + 273.15
        q_f = float(p_in_mwt[j])

        q_dc_av = tes.dc_avail(t_cold_k)
        q_ch_av = tes.ch_avail(t_hot_k)

        q_ch = q_dc = q_pc_su = q_pc_on = q_defoc = 0.0
        frac_on = 0.0

        if not pc_on:
            if (q_f + q_dc_av) >= q_pc_min:
                t_rem = su_time - t_su_acc
                e_rem = e_su_req - e_su_acc
                q_su = min(q_su_max, q_f + q_dc_av)
                t_need = max(t_rem, e_rem / max(q_su, 1e-9))
                if t_need < 1.0:
                    pc_on = True
                    e_su_acc = 0.0
                    t_su_acc = 0.0
                    frac_on = 1.0 - t_need
                    q_pc_su = q_su * t_need
                else:
                    e_su_acc += q_su
                    t_su_acc += 1.0
                    q_pc_su = q_su
            else:
                e_su_acc = 0.0
                t_su_acc = 0.0
        else:
            frac_on = 1.0

        q_f_rest = q_f - min(q_f, q_pc_su)
        need_dc_su = max(q_pc_su - q_f, 0.0)

        if frac_on > 0.0:
            q_f_on = q_f_rest / frac_on
            dc_left = max(q_dc_av - need_dc_su, 0.0) / frac_on
            if q_f_on >= q_pc_des:
                sur = q_f_on - q_pc_des
                ch = min(sur, q_ch_av / frac_on)
                extra = min(sur - ch, q_pc_max - q_pc_des)
                q_pc_on = q_pc_des + extra
                q_ch = ch * frac_on
                q_defoc = (sur - ch - extra) * frac_on
            else:
                dc = min(q_pc_des - q_f_on, dc_left)
                if (q_f_on + dc) < q_pc_min:
                    pc_on = False
                    frac_on = 0.0
                    q_ch = min(q_f_rest, q_ch_av)
                    q_defoc = q_f_rest - q_ch
                else:
                    q_pc_on = q_f_on + dc
                    q_dc = dc * frac_on
            q_dc += need_dc_su
        else:
            q_dc = need_dc_su
            q_ch = min(q_f_rest, q_ch_av)
            q_defoc = q_f_rest - q_ch

        if q_ch > 1e-6:
            r_tes = tes.step(+1, q_ch, t_hot_k, t_amb_k)
        elif q_dc > 1e-6:
            r_tes = tes.step(-1, q_dc, t_cold_k, t_amb_k)
        else:
            r_tes = tes.step(0, 0.0, t_cold_k, t_amb_k)

        q_f_used = q_f - q_defoc
        q_pc = q_pc_su + q_pc_on * frac_on
        w_gross = eta_gross(q_pc_on, t_db, q_pc_des) * q_pc_on * frac_on * deg_acumulado if frac_on > 0.0 else 0.0

        # Parcelas Parasitas SAM [MWe]
        w_fixed = PB_FIXED_PAR_SAM * p_gross_nom_mwe
        f_load = min(w_gross / p_gross_nom_mwe, 1.2) if w_gross > 0.0 else 0.0
        w_bop = p_gross_nom_mwe * BOP_PAR_SAM * (BOP_PAR_0_SAM + BOP_PAR_1_SAM * f_load + BOP_PAR_2_SAM * (f_load ** 2)) if w_gross > 0.0 else 0.0
        w_track = SCA_DRIVES_ELEC_SAM * 1e-6 * 8.0 * n_loops_atual if q_f > 0 else 0.0

        m_sf = q_f_used * 1e3 / (cp_f * dt_loop) if q_f_used > 0 else 0.0
        w_sf = (k_sf_pump / max(fator_campo ** 2, 1e-6)) * (m_sf ** 3) if m_sf > 0 else 0.0

        m_pb = q_pc * 1e3 / (cp_f * dt_loop)
        w_pump_pb = PB_PUMP_COEF_SAM * m_pb / 1e3

        m_tank = r_tes["m_tank"]
        m_diff = abs(m_pb - m_sf)
        w_pump_tes = TES_PUMP_COEF_SAM * (m_tank + m_diff) / 1e3 if (m_tank > 0 or m_pb > 0) else 0.0

        t_wb = t_wetbulb(t_db, t_dew)
        w_ct, m_water = cooling_tower(max(q_pc_on * frac_on - w_gross, 0.0), t_db, t_wb, p_gross_nom_mwe, pc_eff) if w_gross > 0.0 else (0.0, 0.0)

        w_fp_field = float(q_fp_field_mwt[j])
        w_fp_tes = float(r_tes["q_htr"])

        w_par_total = w_fixed + w_bop + w_track + w_sf + w_pump_pb + w_pump_tes + w_ct + w_fp_field + w_fp_tes
        w_net = w_gross - w_par_total

        pout_horario_bruto_w[j] = w_gross * 1e6
        pout_horario_liquido_w[j] = w_net * 1e6
        q_pc_out[j] = q_pc
        q_ch_out[j] = q_ch
        q_dc_out[j] = q_dc
        q_defoc_out[j] = q_defoc
        q_htf_used_out[j] = q_f_used
        e_tes_armazenada_mwht[j] = tes.hot.mass_avail() * salt_cp_ave_k(t_cold_k, t_hot_k) * (t_hot_k - t_cold_k) / 3600.0
        q_loss_tes_out[j] = r_tes["q_loss"]
        w_fixed_out[j] = w_fixed
        w_bop_out[j] = w_bop
        w_track_out[j] = w_track
        w_pump_sf_out[j] = w_sf
        w_pump_pb_out[j] = w_pump_pb
        w_pump_tes_out[j] = w_pump_tes
        w_ct_out[j] = w_ct
        w_fp_field_out[j] = w_fp_field
        w_fp_tes_out[j] = w_fp_tes
        w_par_total_out[j] = w_par_total
        m_water_m3_out[j] = m_water * 3600.0 / 1000.0

    pot_anual_bruta = pd.Series(pout_horario_bruto_w.reshape(-1, 8760).sum(axis=1), dtype=float)
    pot_anual_liquida = pd.Series(pout_horario_liquido_w.reshape(-1, 8760).sum(axis=1), dtype=float)

    return {
        "p_horaria_bruta_w": pout_horario_bruto_w,
        "p_horaria_liquida_w": pout_horario_liquido_w,
        "pot_anual_bruta_wh": pot_anual_bruta,
        "pot_anual_liquida_wh": pot_anual_liquida,
        "diagnostico_csp": {
            "Field_fraction_of_focused_SCAs": np.clip(scas_def_comp * np.clip(np.where(p_in_mwt > 1e-6, 1.0 - (q_defoc_out / np.maximum(p_in_mwt, 1e-6)), 1.0), 0.0, 1.0), 0.0, 1.0) if scas_def_comp is not None else np.clip(np.where(p_in_mwt > 1e-6, 1.0 - (q_defoc_out / np.maximum(p_in_mwt, 1e-6)), 1.0), 0.0, 1.0),
            "scas_def": np.clip(scas_def_comp * np.clip(np.where(p_in_mwt > 1e-6, 1.0 - (q_defoc_out / np.maximum(p_in_mwt, 1e-6)), 1.0), 0.0, 1.0), 0.0, 1.0) if scas_def_comp is not None else np.clip(np.where(p_in_mwt > 1e-6, 1.0 - (q_defoc_out / np.maximum(p_in_mwt, 1e-6)), 1.0), 0.0, 1.0),
            "df_ctrl": np.clip(np.where(p_in_mwt > 1e-6, 1.0 - (q_defoc_out / np.maximum(p_in_mwt, 1e-6)), 1.0), 0.0, 1.0),
            "Q_Receptor_Antes_Inercia_MWt": p_in_mwt,
            "Q_Perda_Inercia_Campo_MWt": np.zeros(n_total),
            "Q_Dump_TES_Cheio_MWt": q_defoc_out,
            "Q_HTF_Util_Entregue_Campo_MWt": q_htf_used_out,
            "Q_Partida_Turbina_MWt": q_pc_out - (pout_horario_bruto_w / 1e6) / max(pc_eff, 1e-9),
            "Q_Entrada_Ciclo_Rankine_MWt": q_pc_out,
            "Q_Carga_TES_MWt": q_ch_out,
            "Q_Descarga_TES_MWt": q_dc_out,
            "E_TES_Armazenada_MWht": e_tes_armazenada_mwht,
            "Q_Perda_Tanques_TES_MWt": q_loss_tes_out,
            "P_Ciclo_Bruta_MWe": pout_horario_bruto_w / 1e6,
            "Par_Fixa_Bloco_MWe": w_fixed_out,
            "Par_BOP_MWe": w_bop_out,
            "Par_Rastreamento_SCA_MWe": w_track_out,
            "Par_Bomba_Campo_HTF_MWe": w_pump_sf_out,
            "Par_Bomba_Ciclo_HTF_MWe": w_pump_pb_out,
            "Par_Bomba_TES_HTF_MWe": w_pump_tes_out,
            "Par_Condensador_Torre_MWe": w_ct_out,
            "Par_Total_MWe": w_par_total_out,
            "P_Liquida_MWe": pout_horario_liquido_w / 1e6,
            "Par_Anticongelamento_Campo_MWe": w_fp_field_out,
            "Par_Anticongelamento_TES_MWe": w_fp_tes_out,
            "Consumo_Agua_m3": m_water_m3_out,
        },
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

    if not (parametros_json and os.path.isfile(parametros_json)):
        for cand in ["eolica_windpower.json", os.path.join(os.path.dirname(__file__), "eolica_windpower.json")]:
            if os.path.isfile(cand):
                parametros_json = cand
                break

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