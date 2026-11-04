# -*- coding: utf-8 -*-
"""
Otimização Multiobjetivo (NSGA-II) sobre Malha Paramétrica de 10 em 10 MVA (Total = 100 MVA).
Variáveis de Projeto (4):
  1. Capacidade TES [horas]
  2. Capacidade BESS [MWh]
  3. Múltiplo Solar da usina CSP [-]
  4. Razão DC/AC (ou AC/DC) da usina PV [-]

Utiliza os métodos financeiros de 'analise_fin.py' para obter VPL, LCOE, TIRM e PBD,
e avalia a função objetivo: Fator de Capacidade / LCOE com cálculo exato de perdas do TES.
"""

from dataclasses import dataclass
import time
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Biblioteca para NSGA-II (pip install pymoo)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from geracao_series_sinteticas import gerar_series_climaticas
from geracao_energia_v2 import (
    CYCLE_THERMAL_POWER,
    DEGRADATION_CSP_DEFAULT,
    PC_EFF_DEFAULT,
    calc_angles_eff_csp,
    calcular_perda_tes_w,  # Opcional, caso queira inspecionar a perda fora do despacho
    despachar_csp_tes,
    field,
    ler_metadados_nsrdb,
    n_loops,
    preparar_geometria_solar_pv,
    simular_despacho_bess,
    simular_geracao_eolica,
    simular_geracao_pv,
    simular_receptor_csp,
)

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS DA OTIMIZAÇÃO
# ==============================================================================

USAR_RAZAO_AC_DC_INVERSA = False
VIDA_UTIL_FINANCEIRA = 30


# ==============================================================================
# 2. FUNÇÕES FINANCEIRAS INTEGRADAS DE 'analise_fin.py'
# ==============================================================================

def calcular_valor_presente(y, x, taxa):
    serie = pd.concat([pd.Series(x), pd.Series(y)], ignore_index=True)
    indices = np.arange(len(serie))
    return serie / ((1 + taxa) ** indices)


def calcular_tirm(y, x, taxa1, taxa2, qual):
    serie = pd.concat([pd.Series(x), pd.Series(y)], ignore_index=True)
    n_len = len(serie)
    indices = np.arange(n_len)
    taxa_negativa = taxa2 if qual == 1 else taxa1
    mask_pos = serie > 0
    mask_neg = serie <= 0
    vf_positivo = (serie[mask_pos] * ((1 + taxa1) ** (30 - indices[mask_pos]))).sum()
    vp_negativo = (serie[mask_neg] / ((1 + taxa_negativa) ** indices[mask_neg])).sum()
    if vp_negativo == 0:
        return 0
    return (((vf_positivo / abs(vp_negativo)) ** (1 / 30)) - 1) * 100


def calcular_pbd(lista1, qtd_anos_total1):
    soma_acumulada = np.cumsum(lista1)
    if soma_acumulada.iloc[-1] < 0:
        return qtd_anos_total1
    idx = np.where(soma_acumulada >= 0)[0][0]
    if idx == 0:
        return 0
    saldo_anterior = abs(soma_acumulada.iloc[idx - 1])
    fracao = saldo_anterior / lista1.iloc[idx]
    return (idx - 1) + fracao


def analise_fin_otimizada_hibrida(
    vida_util: int,
    horas_tes: float,
    energia_ano1_gwh: float,
    pot_anual_array_wh: np.ndarray,
    p_csp_mva: float = 100.0,
    p_pv_mva: float = 0.0,
    p_eol_mva: float = 0.0,
    n_loops_csp: int = 0,
    n_mod_pv: int = 0,
    bess_mwh: float = 0.0,
    qtd_trocas_bat: int = 0,
    k_preco: float = 1.0,
    k_capex: float = 1.0,
    k_wacc: float = 1.0,
    k_dolar: float = 1.0,
    k_cap_t: float = 1.0,
    k_contrato: float = 1.0,
    flag_print: bool = False,
) -> Tuple[float, float, float, float]:
    vida_hibrido = vida_util
    conv_dolar_reais = 5.62 * k_dolar
    adesao_ccee = 8335.0
    valor_terra_nua = 2501.93  # R$/ha

    if p_csp_mva > 0:
        fator_csp = p_csp_mva / 100.0
        fator_campo_csp = n_loops_csp / float(n_loops)

        reflective_area = 949888.0 * fator_campo_csp
        area_total_csp = reflective_area * 1.4 * 15.0 / 6.0 / 1e4
        valor_area_csp = valor_terra_nua * area_total_csp

        site_improvements = 25.0
        solar_field = 150.0
        htf_system = 60.0
        storage = 62.0
        power_plant = 910.0
        balance_of_plant = 90.0
        contingency = 0.07
        epc = 0.11

        p_ref = 111.0 * fator_csp
        tes_capacity_val = (p_ref / 0.356) * horas_tes

        cost_site = site_improvements * reflective_area
        cost_solar_field = solar_field * reflective_area
        cost_htf_system = htf_system * reflective_area
        cost_tes = storage * tes_capacity_val * 1000.0
        cost_power_cycle = power_plant * p_ref * 1000.0
        cost_balance = balance_of_plant * p_ref * 1000.0

        sub_total_csp = (
            cost_site + cost_tes + cost_power_cycle
            + cost_balance + cost_solar_field + cost_htf_system
        )
        total_direct_cost_csp = sub_total_csp * (1.0 + contingency)
        total_indirect_cost_csp = (total_direct_cost_csp * epc) + (valor_area_csp / conv_dolar_reais)
        capex_dolar_csp = total_direct_cost_csp + total_indirect_cost_csp

        gross_net_conversion_factor = 0.9
        taxa_oem_fixo_csp = 70.0
        oem_fixo_csp_brl = p_ref * gross_net_conversion_factor * 1e6 * taxa_oem_fixo_csp / 1000.0
    else:
        total_direct_cost_csp = 0.0
        capex_dolar_csp = 0.0
        oem_fixo_csp_brl = 0.0

    if p_pv_mva > 0 and n_mod_pv > 0:
        area_ha_pv = n_mod_pv * 2.584386 * 0.0001
        valor_area_pv_usd = (valor_terra_nua * area_ha_pv) / conv_dolar_reais
        custo_modulos_usd = (n_mod_pv * 785.10) / conv_dolar_reais
        custo_inversor_pv_usd = (p_pv_mva * 1e6) * 0.05
        direct_cost_pv_usd = custo_modulos_usd + custo_inversor_pv_usd
        capex_dolar_pv = direct_cost_pv_usd + valor_area_pv_usd
        oem_fixo_pv_brl = 0.02 * (capex_dolar_pv * conv_dolar_reais)
    else:
        direct_cost_pv_usd = 0.0
        capex_dolar_pv = 0.0
        oem_fixo_pv_brl = 0.0

    if p_eol_mva > 0:
        direct_cost_eol_usd = p_eol_mva * 1000.0 * 1079.0
        capex_dolar_eol = direct_cost_eol_usd
        oem_fixo_eol_brl = 0.04 * (capex_dolar_eol * conv_dolar_reais)
    else:
        direct_cost_eol_usd = 0.0
        capex_dolar_eol = 0.0
        oem_fixo_eol_brl = 0.0

    if bess_mwh > 0.5:
        pot_max_bat_kw = min(100000.0, bess_mwh * 0.5 * 1000.0)
        custo_baterias_usd = bess_mwh * 1000.0 * 299.0
        custo_inv_bat_usd = pot_max_bat_kw * 244.0
        custo_reposicoes_usd = qtd_trocas_bat * (bess_mwh * 1000.0 * 224.1)
        direct_cost_bess_usd = custo_baterias_usd + custo_inv_bat_usd + custo_reposicoes_usd
        capex_dolar_bess = direct_cost_bess_usd
        oem_fixo_bess_brl = 0.02 * ((custo_baterias_usd + custo_inv_bat_usd) * conv_dolar_reais)
    else:
        direct_cost_bess_usd = 0.0
        capex_dolar_bess = 0.0
        oem_fixo_bess_brl = 0.0

    total_direct_cost_usd = (
        total_direct_cost_csp + direct_cost_pv_usd + direct_cost_eol_usd + direct_cost_bess_usd
    )
    capex_dolar_total = capex_dolar_csp + capex_dolar_pv + capex_dolar_eol + capex_dolar_bess
    capex_usina_brl = capex_dolar_total * conv_dolar_reais * k_capex
    capex = capex_usina_brl + adesao_ccee

    qtd_anos_gerar = vida_util
    price_acr = 445.6 * k_preco
    vet_price_acr = pd.Series([price_acr] * qtd_anos_gerar)

    contrato_anual = energia_ano1_gwh * 1e9 * k_contrato
    limite_superior = 0.15
    limite_inferior = 0.10

    pot_anual = np.array(pot_anual_array_wh, dtype=float)

    m_excedente = np.where(
        pot_anual > contrato_anual * (1 + limite_superior),
        pot_anual - contrato_anual * (1 + limite_superior),
        0.0,
    )
    m_acumulado = np.where(
        pot_anual > contrato_anual * (1 + limite_superior),
        pot_anual - m_excedente - contrato_anual,
        np.where(pot_anual >= contrato_anual, pot_anual - contrato_anual, 0.0),
    )
    m_fixo = np.full(qtd_anos_gerar, contrato_anual)
    m_ress1 = np.where(
        (pot_anual < contrato_anual) & (pot_anual >= contrato_anual * (1 - limite_inferior)),
        contrato_anual - pot_anual,
        np.where(pot_anual < contrato_anual * (1 - limite_inferior), contrato_anual * limite_inferior, 0.0),
    )
    m_ress2 = np.where(
        pot_anual < contrato_anual * (1 - limite_inferior),
        contrato_anual * (1 - limite_inferior) - pot_anual,
        0.0,
    )

    rbo = (
        (m_excedente / 1e6) * 0.3 * vet_price_acr
        + (m_acumulado / 1e6) * vet_price_acr
        + (m_fixo / 1e6) * vet_price_acr
        - (m_ress1 / 1e6) * 1.06 * vet_price_acr
        - (m_ress2 / 1e6) * 1.15 * vet_price_acr
    )

    energia_transportada_pico = 100000000.0
    taxa_tust = 5.205
    desconto_tust = 0.5
    tust = pd.Series([energia_transportada_pico * taxa_tust * desconto_tust * 12.0 / 1000.0] * vida_hibrido)

    pis, cofins = 0.0165, 0.076
    deducao = pis * (rbo - tust) + cofins * (rbo - tust)
    rol = rbo - deducao

    oem_fixo_total_brl = oem_fixo_csp_brl + oem_fixo_pv_brl + oem_fixo_eol_brl + oem_fixo_bess_brl
    taxa_oem_var_csp = (4.0 * conv_dolar_reais) * (p_csp_mva / 100.0)
    oem_var_brl = taxa_oem_var_csp * pd.Series(pot_anual) / 1e6
    oem_total = pd.Series([oem_fixo_total_brl] * vida_hibrido) + oem_var_brl

    taxa_tfsee = 0.004
    tfsee = (rol - tust) * taxa_tfsee

    taxa_seguro = 0.003
    seguro = pd.Series([capex * taxa_seguro] * vida_hibrido)

    taxa_pd = 0.01
    pd_tax = rol * taxa_pd

    taxa_ccee = 0.091
    ccee = pot_anual * taxa_ccee / 1e6

    ebitda = rol - oem_total - tust - tfsee - seguro - pd_tax - ccee
    dep_anual = (total_direct_cost_usd * conv_dolar_reais) / vida_hibrido
    lbo = ebitda - dep_anual
    dep = pd.Series([dep_anual] * len(rbo))

    taxa_csll = 0.09
    taxa_ir1 = 0.15
    taxa_ir2 = 0.10

    rem_t = 0.0427
    rem_c = 0.1026 * k_wacc
    cap_t = 0.4513 * k_cap_t
    cap_p = 1.0 - cap_t
    anos_prest = 20

    saldo_devedor, amort, juros = [], [], []
    amort_valor = cap_t * capex / anos_prest
    for ii in range(vida_hibrido + 1):
        if ii == 0:
            saldo_devedor.append(cap_t * capex)
        elif 0 < ii <= anos_prest:
            saldo_devedor.append(saldo_devedor[ii - 1] - amort_valor)
            amort.append(amort_valor)
            juros.append(saldo_devedor[ii - 1] * rem_t)
        else:
            saldo_devedor.append(0.0)
            amort.append(0.0)
            juros.append(0.0)

    juros = pd.Series(juros)
    amort = pd.Series(amort)

    lbo_a = lbo - juros
    csll_a = np.where(lbo_a > 0, taxa_csll * lbo_a, 0.0)
    ir_a = np.where(
        lbo_a > 0,
        np.where(lbo_a >= 240000.0, (lbo_a * taxa_ir1) + ((lbo_a - 240000.0) * taxa_ir2), lbo_a * taxa_ir1),
        0.0,
    )

    fcla = lbo_a - pd.Series(csll_a) - pd.Series(ir_a) + dep - amort
    fcla_vp = calcular_valor_presente(fcla, -capex * cap_p, rem_c)
    vpl_a = float(fcla_vp.sum())

    tirm_a = float(calcular_tirm(fcla, -capex * cap_p, rem_c, rem_t, 1))

    vp_energia_a = calcular_valor_presente(pd.Series(pot_anual), 0, rem_c).sum()
    vp_oem_a = calcular_valor_presente(oem_total, 0, rem_c).sum()
    vp_juros_a = calcular_valor_presente(juros, 0, rem_c).sum()

    lcoe_a = (capex + vp_oem_a + vp_juros_a) / max(vp_energia_a / 1000.0, 1e-9)
    lcoe_a_dolar = float(lcoe_a / conv_dolar_reais)

    pbd_a = float(calcular_pbd(fcla_vp, qtd_anos_gerar))

    return vpl_a, lcoe_a_dolar, tirm_a, pbd_a


# ==============================================================================
# 3. MALHA DE 10 EM 10 MVA E PRÉ-CÁLCULO FÍSICO UNITÁRIO (COM A LÓGICA DO SAM)
# ==============================================================================

def gerar_malha_potencias(potencia_total_mva: int = 100, passo_mva: int = 10) -> List[Dict[str, int]]:
    combinacoes = []
    id_comb = 1
    for p_csp in range(0, potencia_total_mva + 1, passo_mva):
        for p_pv in range(0, potencia_total_mva - p_csp + 1, passo_mva):
            p_eol = potencia_total_mva - p_csp - p_pv
            combinacoes.append({
                "id": f"Caso_{id_comb:02d}_CSP{p_csp}_PV{p_pv}_EOL{p_eol}",
                "p_csp_mva": p_csp,
                "p_pv_mva": p_pv,
                "p_eol_mva": p_eol,
            })
            id_comb += 1
    return combinacoes


@dataclass
class CacheFisicoUnitario:
    n_horas: int
    vida_util_sim: int
    q_abs_1loop_mwt: np.ndarray
    q_loss_1loop_mwt: np.ndarray
    t_in_celsius: np.ndarray
    p_dc_1modulo_w: np.ndarray
    p_eol_ref_100mw_w: np.ndarray
    temp_amb_celsius: np.ndarray


def precalcular_perfis_unitarios(
    cenario_clima: dict,
    vida_util_sim: int,
    metadados_loc: Optional[dict] = None,
    ano_base_solar: int = 2021,
    json_eolica: str = r'hibrido\eolica\sam\eolica_windpower.json',
    arquivo_sam_eolica: str = None,
) -> CacheFisicoUnitario:
    n_horas = len(cenario_clima["temp"])

    if metadados_loc is not None:
        lat = metadados_loc["latitude"]
        lon = metadados_loc["longitude"]
        tz = metadados_loc["tz"]
        alt = metadados_loc["altitude"]
        bs_pv = metadados_loc["bs_pv"]
    else:
        lat, lon, tz, alt, bs_pv = -13.27, -43.42, -3, 446.0, 13.27

    opt_eff = calc_angles_eff_csp(zone=tz, latitude=lat, longitude=lon)
    _, t_in_c, p_loss_base = simular_receptor_csp(cenario_clima, opt_eff)
    dni_vetor = cenario_clima["dni"]
    temp_amb_celsius = cenario_clima["temp"]
    a_aperture = field.col.reflective_aperture_area
    soma_opt_eff_horaria = opt_eff.sum(axis=1)
    q_abs_1loop_mwt = np.array([
        (dni_vetor[i] * a_aperture * soma_opt_eff_horaria[i % 8760] * 1e-6)
        for i in range(n_horas)
    ])
    q_loss_1loop_mwt = p_loss_base / n_loops

    aoi_pv, zenith_pv = preparar_geometria_solar_pv(
        n_horas=n_horas,
        vida_util=vida_util_sim,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        tz=tz,
        bs=bs_pv,
        ano_base=ano_base_solar,
    )
    res_pv_1mod = simular_geracao_pv(
        cenario_clima,
        aoi_pv,
        zenith_pv,
        n_mod=1,
        pot_nom_w=550.0,
        p_nom_ac_w=1e9,
        bs=bs_pv,
    )
    p_dc_1modulo_w = res_pv_1mod["p_pv_dc_w"]

    res_eol = simular_geracao_eolica(
        cenario_clima,
        parametros_json=json_eolica,
        arquivo_sam_srw=arquivo_sam_eolica,
    )
    p_eol_ref_100mw_w = res_eol["p_eolica_w"]

    return CacheFisicoUnitario(
        n_horas=n_horas,
        vida_util_sim=vida_util_sim,
        q_abs_1loop_mwt=q_abs_1loop_mwt,
        q_loss_1loop_mwt=q_loss_1loop_mwt,
        t_in_celsius=t_in_c,
        p_dc_1modulo_w=p_dc_1modulo_w,
        p_eol_ref_100mw_w=p_eol_ref_100mw_w,
        temp_amb_celsius=temp_amb_celsius,
    )


# ==============================================================================
# 4. SIMULADOR ESCALÁVEL E CÁLCULO DE (FATOR DE CAPACIDADE / LCOE)
# ==============================================================================

def avaliar_configuracao_hibrida(
    cache: CacheFisicoUnitario,
    p_csp_mva: float,
    p_pv_mva: float,
    p_eol_mva: float,
    horas_tes: float = 12.0,
    bess_mwh: float = 0.0,
    sm_csp: float = 2.0,
    razao_pv: float = 1.36,
    demanda_alvo_w: float = 100e6,
) -> dict:
    n_horas = cache.n_horas
    vida_sim = cache.vida_util_sim

    # 1. USINA CSP + TES
    if p_csp_mva > 0:
        fator_bloco_csp = p_csp_mva / 100.0
        n_loops_escalado = int(math.ceil(n_loops * fator_bloco_csp * (sm_csp / 2.0)))
        fator_campo_real = n_loops_escalado / float(n_loops)
        p_max_abs_mwt = 544.0 * fator_campo_real

        q_abs_total = cache.q_abs_1loop_mwt * n_loops_escalado
        q_loss_total = cache.q_loss_1loop_mwt * n_loops_escalado
        df_vec = np.where(q_abs_total < p_max_abs_mwt, 1.0, p_max_abs_mwt / np.maximum(q_abs_total, 1e-9))
        p_rec_escalado = np.maximum(0.0, q_abs_total * df_vec - q_loss_total)

        cycle_thermal_power_esc = CYCLE_THERMAL_POWER * fator_bloco_csp
        tes_cap_esc = cycle_thermal_power_esc * horas_tes

        res_csp = despachar_csp_tes(
            p_in=p_rec_escalado,
            t_htf_celsius=cache.t_in_celsius,
            tes_capacity=tes_cap_esc,
            cycle_power=cycle_thermal_power_esc,
            htf_loss=None,
            pc_eff=PC_EFF_DEFAULT,
            deg=DEGRADATION_CSP_DEFAULT,
            fator_campo=fator_campo_real,
            temp_amb_celsius=cache.temp_amb_celsius,
        )
        p_csp_w = res_csp["p_horaria_liquida_w"]

    
        print(f'Pot Bruta Soma: {res_csp["p_horaria_bruta_w"].sum() / 1e9:.3f}')
        print(f'Pot Líquida Soma: {res_csp["pot_anual_liquida_wh"].sum() / 1e9:.3f}')
        print(f'Energia Bruta: {res_csp["pot_anual_bruta_wh"][0] / 1e9:.3f}')
        print(f'Energia Líquida: {res_csp["pot_anual_liquida_wh"][0] / 1e9:.3f}')

    else:
        p_csp_w = np.zeros(n_horas)
        n_loops_escalado = 0
        horas_tes = 0.0

    # 2. USINA PV + BESS (ACOPLADO NO LADO DC)
    if p_pv_mva > 0:
        p_nom_ac_w = p_pv_mva * 1e6
        fator_dc_ac = (1.0 / razao_pv) if USAR_RAZAO_AC_DC_INVERSA else razao_pv
        p_dc_instalada_w = p_nom_ac_w * fator_dc_ac
        n_mod_escalado = int(round(p_dc_instalada_w / 550.0))

        n_inv = 0.984
        xl5 = 0.99
        fpv_ac = xl5 * n_inv

        p_pv_dc_w = cache.p_dc_1modulo_w * n_mod_escalado
        p_lim_dc_inversor_w = p_nom_ac_w / n_inv

        if bess_mwh > 0.5:
            n_bat_escalado = max(1, int(round(bess_mwh / 0.00512)))
            res_bess = simular_despacho_bess(
                p_entrada_ac_w=p_pv_dc_w,
                desejo_horario_w=p_lim_dc_inversor_w,
                n_bat=n_bat_escalado,
            )
            p_dc_barramento_pv_w = np.maximum(0.0, res_bess["p_saida_rede_w"])
            qtd_trocas_bat = len(res_bess["trocas_bateria_anos"])
        else:
            p_dc_barramento_pv_w = p_pv_dc_w
            qtd_trocas_bat = 0
            bess_mwh = 0.0

        # Passagem pelo inversor PV e corte de clipping residual (caso bateria encha)
        p_pv_ac_w = np.where(
            p_dc_barramento_pv_w * n_inv > p_nom_ac_w,
            p_nom_ac_w * xl5,
            p_dc_barramento_pv_w * fpv_ac,
        )
    else:
        p_pv_ac_w = np.zeros(n_horas)
        n_mod_escalado = 0
        bess_mwh = 0.0
        qtd_trocas_bat = 0

    # 3. USINA EÓLICA (Múltiplos de 2,5 MW por turbina)
    if p_eol_mva > 0:
        p_eol_w = cache.p_eol_ref_100mw_w * (p_eol_mva / 100.0)
    else:
        p_eol_w = np.zeros(n_horas)

    # 4. BARRAMENTO CA UNIFICADO
    p_rede_final_w = p_csp_w + p_pv_ac_w + p_eol_w

    pot_anual_sim_wh = p_rede_final_w.reshape(vida_sim, 8760).sum(axis=1)
    energia_ano1_gwh = float(pot_anual_sim_wh[0] / 1e9)

    if vida_sim < VIDA_UTIL_FINANCEIRA:
        deg_media = (
            (p_csp_mva * 0.002 + p_pv_mva * 0.0055 + p_eol_mva * 0.002)
            / max(p_csp_mva + p_pv_mva + p_eol_mva, 1.0)
        )
        pot_anual_30a_wh = pot_anual_sim_wh[0] * ((1.0 - deg_media) ** np.arange(VIDA_UTIL_FINANCEIRA))
        qtd_trocas_30a = int(round(qtd_trocas_bat * (VIDA_UTIL_FINANCEIRA / vida_sim)))
    else:
        pot_anual_30a_wh = pot_anual_sim_wh[:VIDA_UTIL_FINANCEIRA]
        qtd_trocas_30a = qtd_trocas_bat

    pot_nominal_total_w = (p_csp_mva + p_pv_mva + p_eol_mva) * 1e6
    fc_medio_pct = float(np.mean(pot_anual_30a_wh) / (pot_nominal_total_w * 8760.0) * 100.0)
    fc_ano1_pct = float(pot_anual_sim_wh[0] / (pot_nominal_total_w * 8760.0) * 100.0)

    p_firme_w = np.minimum(p_rede_final_w, pot_nominal_total_w)
    fc_firme_100mw_pct = float(np.sum(p_firme_w) / vida_sim / (pot_nominal_total_w * 8760.0) * 100.0)

    vpl_brl, lcoe_usd_kwh, tirm_pct, pbd_anos = analise_fin_otimizada_hibrida(
        vida_util=VIDA_UTIL_FINANCEIRA,
        horas_tes=horas_tes,
        energia_ano1_gwh=energia_ano1_gwh,
        pot_anual_array_wh=pot_anual_30a_wh,
        p_csp_mva=p_csp_mva,
        p_pv_mva=p_pv_mva,
        p_eol_mva=p_eol_mva,
        n_loops_csp=n_loops_escalado,
        n_mod_pv=n_mod_escalado,
        bess_mwh=bess_mwh,
        qtd_trocas_bat=qtd_trocas_30a,
    )

    lcoe_usd_mwh = lcoe_usd_kwh * 1000.0
    vpl_musd = vpl_brl / 5.62 / 1e6

    fo_fc_sobre_lcoe_kwh = fc_medio_pct / max(lcoe_usd_kwh, 1e-9)
    fo_fc_sobre_lcoe_mwh = fc_medio_pct / max(lcoe_usd_mwh, 1e-9)

    return {
        "vpl_musd": vpl_musd,
        "lcoe_usd_kwh": lcoe_usd_kwh,
        "lcoe_usd_mwh": lcoe_usd_mwh,
        "tirm_pct": tirm_pct,
        "pbd_anos": pbd_anos,
        "fc_medio_pct": fc_medio_pct,
        "fc_ano1_pct": fc_ano1_pct,
        "fc_firme_100mw_pct": fc_firme_100mw_pct,
        "fo_fc_sobre_lcoe": fo_fc_sobre_lcoe_kwh,
        "fo_fc_sobre_lcoe_mwh": fo_fc_sobre_lcoe_mwh,
        "energia_ano1_gwh": energia_ano1_gwh,
        "n_loops_csp": int(n_loops_escalado),
        "n_modulos_pv": n_mod_escalado,
        "n_turbinas_eol": int(round(p_eol_mva / 2.5)),
    }


# ==============================================================================
# 5. PROBLEMA NSGA-II COM AS 4 VARIÁVEIS DE PROJETO
# ==============================================================================

class ProblemaHibridoNSGA2(ElementwiseProblem):
    def __init__(
        self,
        cache: CacheFisicoUnitario,
        p_csp_mva: float,
        p_pv_mva: float,
        p_eol_mva: float,
        permitir_tes: bool = True,
    ):
        self.cache = cache
        self.p_csp = p_csp_mva
        self.p_pv = p_pv_mva
        self.p_eol = p_eol_mva
        self.permitir_tes = permitir_tes

        lim_pv_min = 0.625 if USAR_RAZAO_AC_DC_INVERSA else 1.0
        lim_pv_max = 1.000 if USAR_RAZAO_AC_DC_INVERSA else 3.0

        tem_csp = p_csp_mva > 0
        tem_tes = tem_csp and permitir_tes
        tem_pv = p_pv_mva > 0

        xl = [
            0.0,
            0.0,
            1.2 if tem_csp else 2.0,
            lim_pv_min if tem_pv else 1.0,
        ]
        xu = [
            16.0 if tem_tes else 0.001,
            400.0 if tem_pv else 0.001,
            3.0 if tem_csp else 2.001,
            lim_pv_max if tem_pv else 1.001,
        ]

        super().__init__(n_var=4, n_obj=2, n_ieq_constr=0, xl=np.array(xl), xu=np.array(xu))

    def _evaluate(self, x, out, *args, **kwargs):
        horas_tes = float(x[0]) if (self.p_csp > 0 and self.permitir_tes) else 0.0
        bess_mwh = float(x[1]) if self.p_pv > 0 else 0.0
        sm_csp = float(x[2]) if self.p_csp > 0 else 0.0
        razao_pv = float(x[3]) if self.p_pv > 0 else 1.0

        res = avaliar_configuracao_hibrida(
            cache=self.cache,
            p_csp_mva=self.p_csp,
            p_pv_mva=self.p_pv,
            p_eol_mva=self.p_eol,
            horas_tes=horas_tes,
            bess_mwh=bess_mwh,
            sm_csp=sm_csp,
            razao_pv=razao_pv,
        )

        out["F"] = [res["lcoe_usd_kwh"], -res["fc_medio_pct"]]


# ==============================================================================
# 6. VALIDAÇÃO COM O SAM E EXECUÇÃO PRINCIPAL DA OTIMIZAÇÃO
# ==============================================================================

def executar_validacao_sam(cache: CacheFisicoUnitario) -> pd.DataFrame:
    casos_validacao = [
        {"Tec": "CSP_Isolado", "p_csp": 50,  "p_pv": 0,   "p_eol": 0,   "tes": 5.0, "sm": 2.0, "raz_pv": 1.0},
        {"Tec": "CSP_Isolado", "p_csp": 100, "p_pv": 0,   "p_eol": 0,   "tes": 5.0, "sm": 2.0, "raz_pv": 1.0},
        {"Tec": "PV_Isolado",  "p_csp": 0,   "p_pv": 50,  "p_eol": 0,   "tes": 0.0,  "sm": 0.0, "raz_pv": 1.36},
        {"Tec": "PV_Isolado",  "p_csp": 0,   "p_pv": 100, "p_eol": 0,   "tes": 0.0,  "sm": 0.0, "raz_pv": 1.36},
        {"Tec": "EOL_Isolada", "p_csp": 0,   "p_pv": 0,   "p_eol": 50,  "tes": 0.0,  "sm": 0.0, "raz_pv": 1.0},
        {"Tec": "EOL_Isolada", "p_csp": 0,   "p_pv": 0,   "p_eol": 100, "tes": 0.0,  "sm": 0.0, "raz_pv": 1.0},
    ]

    linhas = []
    for c in casos_validacao:
        r = avaliar_configuracao_hibrida(
            cache=cache,
            p_csp_mva=c["p_csp"],
            p_pv_mva=c["p_pv"],
            p_eol_mva=c["p_eol"],
            horas_tes=c["tes"],
            bess_mwh=0.0,
            sm_csp=c["sm"],
            razao_pv=c["raz_pv"],
        )
        linhas.append({
            "Tecnologia": c["Tec"],
            "Pot_MVA": c["p_csp"] + c["p_pv"] + c["p_eol"],
            "SAM_Loops_CSP": r["n_loops_csp"],
            "SAM_Modulos_PV": r["n_modulos_pv"],
            "SAM_Turbinas_EOL": r["n_turbinas_eol"],
            "Energia_Ano1_GWh": round(r["energia_ano1_gwh"], 3),
            "FC_Ano1_%": round(r["fc_ano1_pct"], 2),
            "LCOE_USD_kWh": round(r["lcoe_usd_kwh"], 4),
            "FC_sobre_LCOE": round(r["fo_fc_sobre_lcoe"], 2),
            "VPL_MUSD": round(r["vpl_musd"], 2),
        })

    df_val = pd.DataFrame(linhas)
    print("\n" + "=" * 105)
    print("TABELA DE VALIDAÇÃO FÍSICA E FINANCEIRA (BASEADA EM 'analise_fin.py' E SAM)")
    print("=" * 105)
    print(df_val.to_string(index=False))
    print("=" * 105)
    return df_val


def exportar_auditoria_csp_sam(
    cenario_clima: dict,
    cache: CacheFisicoUnitario,
    p_csp_mva: float = 100.0,
    horas_tes: float = 1.0,
    sm_csp: float = 2.0,
    arquivo_saida: str = "auditoria_horaria_csp_vs_sam.xlsx",
) -> pd.DataFrame:
    """
    Exporta todos os passos físicos horários (8760h) e o balanço anual da usina CSP
    para comparação direta coluna a coluna com o 'Time Series' do SAM.
    """
    fator_bloco_csp = p_csp_mva / 100.0
    n_loops_escalado = int(math.ceil(n_loops * fator_bloco_csp * (sm_csp / 2.0)))
    fator_campo_real = n_loops_escalado / float(n_loops)
    p_max_abs_mwt = 544.0 * fator_campo_real

    a_aperture_loop = field.col.reflective_aperture_area * field.nSCA
    area_total_m2 = a_aperture_loop * n_loops_escalado
    dni_w_m2 = cenario_clima["dni"][:8760]

    # Etapa 1: Incidência e Óptica
    q_incidente_campo_mwt = (dni_w_m2 * area_total_m2) * 1e-6
    q_abs_bruta_mwt = cache.q_abs_1loop_mwt[:8760] * n_loops_escalado
    eta_optica_efetiva = np.where(q_incidente_campo_mwt > 0, q_abs_bruta_mwt / q_incidente_campo_mwt, 0.0)

    # Etapa 2: Defocus de Pico e Perdas Térmicas no Receptor/Tubulações
    df_vec = np.where(q_abs_bruta_mwt < p_max_abs_mwt, 1.0, p_max_abs_mwt / np.maximum(q_abs_bruta_mwt, 1e-9))
    q_abs_pos_defocus_mwt = q_abs_bruta_mwt * df_vec
    q_perda_rec_tubos_mwt = cache.q_loss_1loop_mwt[:8760] * n_loops_escalado
    p_rec_escalado_mwt = np.maximum(0.0, q_abs_pos_defocus_mwt - q_perda_rec_tubos_mwt)

    # Etapa 3 e 4: Despacho Térmico, Ciclo Rankine e Parasitas
    cycle_thermal_power_esc = CYCLE_THERMAL_POWER * fator_bloco_csp
    tes_cap_esc = cycle_thermal_power_esc * horas_tes

    res_csp = despachar_csp_tes(
        p_in=p_rec_escalado_mwt,
        t_htf_celsius=cache.t_in_celsius[:8760],
        tes_capacity=tes_cap_esc,
        cycle_power=cycle_thermal_power_esc,
        htf_loss=None,
        pc_eff=PC_EFF_DEFAULT,
        deg=DEGRADATION_CSP_DEFAULT,
        fator_campo=fator_campo_real,
        temp_amb_celsius=cenario_clima["temp"][:8760],
    )
    diag = res_csp["diagnostico_csp"]

    df_horario = pd.DataFrame({
        "Hora_Ano": np.arange(1, 8761),
        "DNI_W_m2": dni_w_m2,
        "Temp_Amb_C": cenario_clima["temp"][:8760],
        "Vento_m_s": cenario_clima["vento_vel"][:8760],
        "T_HTF_Entrada_C": cache.t_in_celsius[:8760],
        "Eta_Optica_Campo": eta_optica_efetiva,
        "1_Q_Incidente_Campo_MWt": q_incidente_campo_mwt,
        "2_Q_Absorvida_Bruta_MWt": q_abs_bruta_mwt,
        "3_Fator_Defocus_Pico": df_vec,
        "4_Q_Absorvida_Pos_Defocus_MWt": q_abs_pos_defocus_mwt,
        "5_Q_Perda_Receptor_e_Tubos_MWt": q_perda_rec_tubos_mwt,
        "6_Q_Receptor_Antes_Inercia_MWt": diag["Q_Receptor_Antes_Inercia_MWt"],
        "7_Q_Perda_Inercia_Campo_MWt": diag["Q_Perda_Inercia_Campo_MWt"],
        "8_Q_Dump_TES_Cheio_MWt": diag["Q_Dump_TES_Cheio_MWt"],
        "9_Q_HTF_Util_Entregue_Campo_MWt": diag["Q_HTF_Util_Entregue_Campo_MWt"],
        "10_Q_Partida_Turbina_MWt": diag["Q_Partida_Turbina_MWt"],
        "11_Q_Carga_TES_MWt": diag["Q_Carga_TES_MWt"],
        "12_Q_Descarga_TES_MWt": diag["Q_Descarga_TES_MWt"],
        "13_E_TES_Armazenada_MWht": diag["E_TES_Armazenada_MWht"],
        "14_Q_Entrada_Ciclo_Rankine_MWt": diag["Q_Entrada_Ciclo_Rankine_MWt"],
        "15_P_Ciclo_Bruta_MWe": diag["P_Ciclo_Bruta_MWe"],
        "16_Par_Fixa_Bloco_MWe": diag["Par_Fixa_Bloco_MWe"],
        "17_Par_BOP_MWe": diag["Par_BOP_MWe"],
        "18_Par_Rastreamento_SCA_MWe": diag["Par_Rastreamento_SCA_MWe"],
        "19_Par_Bomba_Campo_HTF_MWe": diag["Par_Bomba_Campo_HTF_MWe"],
        "20_Par_Bomba_Ciclo_HTF_MWe": diag["Par_Bomba_Ciclo_HTF_MWe"],
        "21_Par_Bomba_TES_HTF_MWe": diag["Par_Bomba_TES_HTF_MWe"],
        "22_Par_Condensador_Torre_MWe": diag["Par_Condensador_Torre_MWe"],
        "23_Par_Total_MWe": diag["Par_Total_MWe"],
        "24_P_Liquida_MWe": diag["P_Liquida_MWe"],
    })

    # Resumo acumulado anual (GWh térmicos ou elétricos) para bater rápido com o SAM
    colunas_energia = [c for c in df_horario.columns if c.endswith("_MWt") or c.endswith("_MWe")]
    df_resumo = pd.DataFrame({
        "Etapa_Fisica": colunas_energia,
        "Total_Anual_GWh": [df_horario[c].sum() / 1000.0 for c in colunas_energia],
    })

    with pd.ExcelWriter(arquivo_saida) as writer:
        df_resumo.to_excel(writer, sheet_name="Resumo_Anual_GWh", index=False)
        df_horario.to_excel(writer, sheet_name="Horario_8760h", index=False)

    print(f"\n[OK] Auditoria CSP exportada para '{arquivo_saida}'!")
    print(df_resumo.to_string(index=False))
    return df_horario


if __name__ == "__main__":
    pasta_clima = r'historico_variaveis\nsrdb_macau'
    prefixo_loc = '2008167_-5.11_-36.62'

    # pasta_clima = r'historico_variaveis\nsrdb'
    # prefixo_loc = '1945275_-13.27_-43.42'

    metadados_loc = ler_metadados_nsrdb(path_pasta=pasta_clima, prefixo_arquivo=prefixo_loc)
    print(
        f"Localidade carregada (ID {metadados_loc['location_id']}): "
        f"Lat={metadados_loc['latitude']}°, Lon={metadados_loc['longitude']}°, "
        f"Alt={metadados_loc['altitude']}m, TZ={metadados_loc['tz']}, "
        f"Inclinação PV={metadados_loc['bs_pv']}°"
    )

    clima = gerar_series_climaticas(
        path_pasta=pasta_clima,
        prefixo_arquivo=prefixo_loc,
        vida_util=1,
        qtd_cenarios=1,
        modo_deterministico=True,
        ano_deterministico=2017,
        incluir_dhi_albedo=True,
    )
    cenario_0 = clima.obter_cenario(0)

    print("\nPré-calculando física unitária das 3 fontes (CSP, PV e Eólica)...")
    t0 = time.time()
    cache_fisico = precalcular_perfis_unitarios(
        cenario_0,
        vida_util_sim=1,
        metadados_loc=metadados_loc,
        ano_base_solar=2017
        # arquivo_sam_eolica=r'projeto_hibrido\WY Southern-Flat Lands.txt' # Descomentar para validação com o SAM
    )
    print(f"Física unitária pré-calculada em {time.time() - t0:.2f} s!")

    exportar_auditoria_csp_sam(
        cenario_clima=cenario_0,
        cache=cache_fisico,
        p_csp_mva=50.0,
        horas_tes=5.0,
        sm_csp=2.0,
        arquivo_saida="auditoria_csp_50mva.xlsx",
    )

    df_validacao = executar_validacao_sam(cache_fisico)

    # malha_100mva = gerar_malha_potencias(potencia_total_mva=100, passo_mva=10)

    # casos_para_rodar = [
    #     c for c in malha_100mva
    #     if (c["p_csp_mva"] % 50 == 0 and c["p_pv_mva"] % 50 == 0 and c["p_eol_mva"] % 50 == 0)
    # ]

    # fronteiras_pareto = []
    # melhores_fc_sobre_lcoe = []

    # for caso in casos_para_rodar:
    #     print(f"\nRodando NSGA-II para {caso['id']} "
    #           f"(CSP={caso['p_csp_mva']} MVA | PV={caso['p_pv_mva']} MVA | EOL={caso['p_eol_mva']} MVA)...")

    #     problema = ProblemaHibridoNSGA2(
    #         cache=cache_fisico,
    #         p_csp_mva=caso["p_csp_mva"],
    #         p_pv_mva=caso["p_pv_mva"],
    #         p_eol_mva=caso["p_eol_mva"],
    #     )

    #     algoritmo = NSGA2(pop_size=24)
    #     criterio_parada = get_termination("n_gen", 15)

    #     res_nsga = minimize(
    #         problema,
    #         algoritmo,
    #         criterio_parada,
    #         seed=42,
    #         verbose=False,
    #     )

    #     solucoes_caso = []
    #     for idx_sol in range(len(res_nsga.F)):
    #         x_sol = res_nsga.X[idx_sol]

    #         tes_opt = float(x_sol[0]) if caso["p_csp_mva"] > 0 else 0.0
    #         bess_opt = float(x_sol[1])
    #         sm_opt = float(x_sol[2]) if caso["p_csp_mva"] > 0 else 0.0
    #         raz_pv_opt = float(x_sol[3]) if caso["p_pv_mva"] > 0 else 0.0

    #         metricas = avaliar_configuracao_hibrida(
    #             cache=cache_fisico,
    #             p_csp_mva=caso["p_csp_mva"],
    #             p_pv_mva=caso["p_pv_mva"],
    #             p_eol_mva=caso["p_eol_mva"],
    #             horas_tes=tes_opt,
    #             bess_mwh=bess_opt,
    #             sm_csp=sm_opt,
    #             razao_pv=raz_pv_opt,
    #         )

    #         registro = {
    #             "Caso": caso["id"],
    #             "P_CSP_MVA": caso["p_csp_mva"],
    #             "P_PV_MVA": caso["p_pv_mva"],
    #             "P_EOL_MVA": caso["p_eol_mva"],
    #             "Cap_TES_h": round(tes_opt, 2),
    #             "Cap_BESS_MWh": round(bess_opt, 2),
    #             "SM_CSP": round(sm_opt, 2),
    #             "Razao_PV": round(raz_pv_opt, 3),
    #             "E_Ano1_GWh": round(metricas["energia_ano1_gwh"], 3),
    #             "FC_Ano1_%": round(metricas["fc_ano1_pct"], 2),
    #             "FC_Medio_%": round(metricas["fc_medio_pct"], 2),
    #             "LCOE_$_kWh": round(metricas["lcoe_usd_kwh"], 4),
    #             # "LCOE_USD_MWh": round(metricas["lcoe_usd_mwh"], 2),
    #             "FC/LCOE": round(metricas["fo_fc_sobre_lcoe"], 2),
    #             "VPL_MUSD": round(metricas["vpl_musd"], 2),
    #             # "TIRM_%": round(metricas["tirm_pct"], 2),
    #             "PBD": round(metricas["pbd_anos"], 2),
    #         }
    #         fronteiras_pareto.append(registro)
    #         solucoes_caso.append(registro)

    #     melhor_sol_caso = max(solucoes_caso, key=lambda d: d["FC/LCOE"])
    #     melhores_fc_sobre_lcoe.append(melhor_sol_caso)

    # df_pareto_total = pd.DataFrame(fronteiras_pareto)
    # df_melhores = pd.DataFrame(melhores_fc_sobre_lcoe).sort_values("FC/LCOE", ascending=False)

    # print("\n" + "=" * 120)
    # print("RANKING DAS MELHORES CONFIGURAÇÕES POR COMBINAÇÃO (MAXIMIZAÇÃO DE FC / LCOE)")
    # print("=" * 120)
    # print(df_melhores.to_string(index=False))
    # print("=" * 120)

    # with pd.ExcelWriter("resultados_otimizacao_fc_lcoe.xlsx") as writer:
    #     df_validacao.to_excel(writer, sheet_name="Validacao_SAM_Fin", index=False)
    #     df_melhores.to_excel(writer, sheet_name="Melhores_FC_sobre_LCOE", index=False)
    #     df_pareto_total.to_excel(writer, sheet_name="Fronteiras_Pareto", index=False)

    # print("\nArquivo 'resultados_otimizacao_fc_lcoe.xlsx' gerado com sucesso!")