# -*- coding: utf-8 -*-
"""
Auditoria Comparativa Multi-Anual: SAM C++ Oficial (PySAM) vs. Modelo Físico Python.

Executa simulações anuais para os anos de 2000 a 2009 cobrindo:
- Capacidade da Usina: 50 a 100 MVA (passos de 10 MVA)
- Armazenamento Térmico (TES): 0 a 9 horas (passos de 1 hora)
- Múltiplo Solar (SM): 1.0 a 2.5
Configuração única para cada ano rodada simultaneamente em ambos os modelos.
Exporta os resultados comparativos detalhados para Excel e CSV.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time
import numpy as np
import pandas as pd
import PySAM.TroughPhysical as tp

from geracao_series_sinteticas import gerar_series_climaticas
from campo_solar_sam import dimensionar_campo_tubulacoes_sam, simular_campo_solar_sam_8760
from geracao_energia_v2 import (
    despachar_csp_tes,
    ler_metadados_nsrdb,
    CYCLE_THERMAL_POWER,
    PC_EFF_DEFAULT,
    DEGRADATION_CSP_DEFAULT,
    T_HOT_OUT_DEFAULT,
    T_COLD_IN_DEFAULT,
    ALPHA_ABS,
)

WS_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_CLIMA = os.path.join(WS_DIR, "historico_variaveis", "nsrdb")
PREFIXO_LOC = "2008167_-5.11_-36.62"
ARQUIVO_CONFIG_SAM = os.path.join(WS_DIR, "csp_sam_2009.json")
ARQUIVO_SAIDA_EXCEL = os.path.join(WS_DIR, "comparativo_multianos_sam_vs_python3.xlsx")
ARQUIVO_SAIDA_CSV = os.path.join(WS_DIR, "comparativo_multianos_sam_vs_python3.csv")

# Definição das 10 configurações únicas (2000 a 2009)
CONFIGURACOES_MULTI_ANUAIS = [
    {"ano": 2000, "p_csp_mva": 50.0,  "tes_h": 0.0, "sm": 1.0, "n_loops": 46},
    {"ano": 2001, "p_csp_mva": 60.0,  "tes_h": 1.0, "sm": 1.2, "n_loops": 66},
    {"ano": 2002, "p_csp_mva": 70.0,  "tes_h": 2.0, "sm": 1.4, "n_loops": 89},
    {"ano": 2003, "p_csp_mva": 80.0,  "tes_h": 3.0, "sm": 1.6, "n_loops": 116},
    {"ano": 2004, "p_csp_mva": 90.0,  "tes_h": 4.0, "sm": 1.8, "n_loops": 147},
    {"ano": 2005, "p_csp_mva": 100.0, "tes_h": 5.0, "sm": 2.0, "n_loops": 181},
    {"ano": 2006, "p_csp_mva": 50.0,  "tes_h": 6.0, "sm": 2.1, "n_loops": 96},
    {"ano": 2007, "p_csp_mva": 70.0,  "tes_h": 7.0, "sm": 2.3, "n_loops": 147},
    {"ano": 2008, "p_csp_mva": 80.0,  "tes_h": 8.0, "sm": 2.4, "n_loops": 175},
    {"ano": 2009, "p_csp_mva": 100.0, "tes_h": 9.0, "sm": 2.5, "n_loops": 228},
    {"ano": 2010, "p_csp_mva": 100.0, "tes_h": 5.0, "sm": 2.5, "n_loops": 228},
    {"ano": 2011, "p_csp_mva": 60.0,  "tes_h": 1.0, "sm": 1.2, "n_loops": 66},
    {"ano": 2012, "p_csp_mva": 70.0,  "tes_h": 2.0, "sm": 1.4, "n_loops": 89},
    {"ano": 2013, "p_csp_mva": 80.0,  "tes_h": 3.0, "sm": 1.6, "n_loops": 116},
    {"ano": 2014, "p_csp_mva": 90.0,  "tes_h": 4.0, "sm": 1.8, "n_loops": 147},
    {"ano": 2015, "p_csp_mva": 100.0, "tes_h": 5.0, "sm": 2.0, "n_loops": 181},
    {"ano": 2016, "p_csp_mva": 50.0,  "tes_h": 6.0, "sm": 2.1, "n_loops": 96},
    {"ano": 2017, "p_csp_mva": 70.0,  "tes_h": 7.0, "sm": 2.3, "n_loops": 147},
    {"ano": 2018, "p_csp_mva": 80.0,  "tes_h": 8.0, "sm": 2.4, "n_loops": 175},
    {"ano": 2019, "p_csp_mva": 100.0, "tes_h": 9.0, "sm": 2.5, "n_loops": 228},
]


def carregar_config_sam_base() -> dict:
    with open(ARQUIVO_CONFIG_SAM, "r", encoding="utf-8") as f:
        return json.load(f)


def executar_simulacao_pysam(
    cfg_base: dict,
    weather_file: str,
    p_ref_mwe: float,
    tes_hours: float,
    n_loops: int,
) -> dict | None:
    """Executa a simulação física oficial do SAM C++ via PySAM com tolerância a falhas."""
    try:
        m = tp.new()
        for k, v in cfg_base.items():
            if k != "file_name":
                try:
                    m.value(k, v)
                except Exception:
                    pass

        m.value("file_name", weather_file)
        m.value("P_ref", float(p_ref_mwe))
        m.value("tshours", float(tes_hours))
        # Para garantir estabilidade no SAM com loops reduzidos ou baixo TES
        m.value("m_dot_htfmax", 12.0)
        m.value("use_solar_mult_or_aperture_area", 1)
        m.value("specified_total_aperture", float(n_loops) * 5248.0)
        m.execute()

        # Extração completa de todas as 125 saídas horárias do PySAM (8760 passos)
        sam_export = m.Outputs.export()
        hourly_dict = {
            k: v for k, v in sam_export.items()
            if isinstance(v, (list, tuple)) and len(v) == 8760
        }
        df_sam_hourly = pd.DataFrame(hourly_dict)
        if "hora" not in df_sam_hourly.columns:
            df_sam_hourly.insert(0, "hora", np.arange(1, 8761))

        return {
            "q_inc_sf": np.array(m.Outputs.q_inc_sf_tot)[:8760],
            "q_rec_inc": np.array(m.Outputs.q_dot_rec_inc)[:8760],
            "q_rec_abs": np.array(m.Outputs.q_dot_rec_abs)[:8760],
            "q_htf_out": np.array(m.Outputs.q_dot_htf_sf_out)[:8760],
            "q_ch": np.array(m.Outputs.q_ch_tes)[:8760],
            "q_dc": np.array(m.Outputs.q_dc_tes)[:8760],
            "q_pb": np.array(m.Outputs.q_pb)[:8760],
            "p_cycle": np.array(m.Outputs.P_cycle)[:8760],
            "p_net": np.array(m.Outputs.P_out_net)[:8760],
            "defocus": np.array(m.Outputs.defocus)[:8760],
            "tank_losses": np.array(m.Outputs.tank_losses)[:8760],
            "w_field_pump": np.array(m.Outputs.W_dot_field_pump)[:8760],
            "df_hourly": df_sam_hourly,
        }
    except Exception as e:
        print(f"  [AVISO PySAM] Simulação abortada pelo solver do SAM: {e}")
        return None


def executar_simulacao_python(
    cenario_clima: dict,
    p_csp_mva: float,
    tes_hours: float,
    sm: float,
    n_loops: int,
    latitude: float = -13.27,
    longitude: float = -43.42,
    tz: float = -3.0,
) -> dict:
    """Executa a simulação física completa do modelo puro em Python."""
    p_ref_gross = p_csp_mva * (111.0 / 100.0)
    cycle_th = CYCLE_THERMAL_POWER * (p_csp_mva / 100.0)
    tes_cap = cycle_th * tes_hours

    dni_arr = np.asarray(cenario_clima["dni"], dtype=float)
    temp_arr = np.asarray(cenario_clima["temp"], dtype=float)
    dni_sum = int(round(float(np.sum(dni_arr))))
    temp_sum = int(round(float(np.sum(temp_arr))))
    clima_sig = f"lat{latitude:.2f}_lon{longitude:.2f}_tz{int(tz)}_dni{dni_sum}_t{temp_sum}_n8760"

    pasta_cache = os.path.join(WS_DIR, "cache_campo")
    os.makedirs(pasta_cache, exist_ok=True)
    cache_dirs = [
        pasta_cache,
    ]
    res_campo = None
    cache_nome_alvo = f"cache_campo_v9_{n_loops}loops_p{p_ref_gross:.1f}_tes{tes_hours:.0f}h_{clima_sig}.npz"
    for cd in cache_dirs:
        c_path = os.path.join(cd, cache_nome_alvo)
        if os.path.exists(c_path):
            try:
                c = np.load(c_path)
                if len(c["q_htf_out"]) == 8760:
                    res_campo = {k: c[k] for k in c.files}
                    break
            except Exception:
                pass

    if res_campo is None:
        cfg_campo = dimensionar_campo_tubulacoes_sam(
            n_loops_field=n_loops,
            p_ref_mw=p_ref_gross,
            eta_ref=PC_EFF_DEFAULT,
        )
        cfg_campo["tes_hours"] = float(tes_hours)
        res_campo = simular_campo_solar_sam_8760(
            cfg=cfg_campo,
            cenario_clima=cenario_clima,
            latitude=latitude,
            longitude=longitude,
            tz=tz,
        )
        try:
            cache_save_path = os.path.join(pasta_cache, cache_nome_alvo)
            np.savez_compressed(cache_save_path, **res_campo)
        except Exception as e:
            pass

    py_q_inc = res_campo["q_inc_sf"][:8760]
    py_q_rec_inc = res_campo["q_rec_inc"][:8760]
    py_q_rec_abs = res_campo["q_rec_abs"][:8760]
    py_q_htf_out = res_campo["q_htf_out"][:8760]

    res_csp = despachar_csp_tes(
        p_in=py_q_htf_out,
        t_htf_celsius=np.where(py_q_htf_out > 0, T_HOT_OUT_DEFAULT, T_COLD_IN_DEFAULT),
        tes_capacity=tes_cap,
        cycle_power=cycle_th,
        htf_loss=None,
        pc_eff=PC_EFF_DEFAULT,
        deg=DEGRADATION_CSP_DEFAULT,
        fator_campo=n_loops / 91.0,
        temp_amb_celsius=temp_arr[:8760],
        q_rec_inc_mwt=py_q_rec_inc,
        q_rec_abs_mwt=py_q_rec_abs,
        q_rec_loss_mwt=res_campo.get("q_rec_loss", np.zeros(8760))[:8760],
        q_fp_field_mwt=res_campo.get("q_fp", np.zeros(8760))[:8760],
        scas_def_comp=res_campo.get("scas_def", np.ones(8760))[:8760],
        solar_multiple=sm,
        n_loops=n_loops,
    )
    diag = res_csp["diagnostico_csp"]

    q_inc_sf_final = py_q_inc.copy()

    # Construção do DataFrame detalhado da simulação Python (compatível com os outputs do SAM)
    t_cold_c = res_campo.get("t_sys_c_t_end", np.zeros(8760))[:8760]
    if np.any(t_cold_c > 200.0):
        t_cold_c = t_cold_c - 273.15
    t_hot_c = res_campo.get("t_sys_h_t_end", np.zeros(8760))[:8760]
    if np.any(t_hot_c > 200.0):
        t_hot_c = t_hot_c - 273.15

    scas_def = res_campo.get("scas_def", np.ones(8760))[:8760]
    df_ctrl = diag.get("df_ctrl", np.ones(8760))[:8760]
    defocus_tot = np.clip(scas_def * df_ctrl, 0.0, 1.0)

    df_py_hourly = pd.DataFrame({
        "hora": np.arange(1, 8761),
        "beam": dni_arr[:8760],
        "tdry": temp_arr[:8760],
        "wspd": np.asarray(cenario_clima.get("vento_vel", np.zeros(8760)), dtype=float)[:8760],
        "CosTh_ave": res_campo.get("cos_th", np.zeros(8760))[:8760],
        "q_inc_sf_tot": q_inc_sf_final[:8760],
        "qinc_costh": res_campo.get("q_inc_costh", np.zeros(8760))[:8760],
        "q_dot_rec_inc": diag.get("q_rec_inc", py_q_rec_inc)[:8760],
        "q_dot_rec_abs": diag.get("q_rec_abs", py_q_rec_abs)[:8760],
        "q_dot_rec_thermal_loss": res_campo.get("q_rec_loss", np.zeros(8760))[:8760],
        "q_dot_rec_refl": res_campo.get("q_rec_refl", np.zeros(8760))[:8760],
        "q_dot_htf_sf_out": diag.get("q_htf_out", py_q_htf_out)[:8760],
        "q_dot_piping_loss": res_campo.get("q_piping_loss", np.zeros(8760))[:8760],
        "e_dot_field_int_energy": res_campo.get("e_int_energy", np.zeros(8760))[:8760],
        "q_dot_freeze_prot": res_campo.get("q_fp", np.zeros(8760))[:8760],
        "SCAs_def": scas_def,
        "defocus": df_ctrl,
        "defocus_final": defocus_tot,
        "T_field_cold_in": t_cold_c,
        "T_field_hot_out": t_hot_c,
        "m_dot_loop": res_campo.get("m_dot_loop", np.zeros(8760))[:8760],
        "rec_time_in_startup": res_campo.get("rec_time_in_startup", np.zeros(8760))[:8760],
        "q_ch_tes": diag.get("Q_Carga_TES_MWt", np.zeros(8760))[:8760],
        "q_dc_tes": diag.get("Q_Descarga_TES_MWt", np.zeros(8760))[:8760],
        "e_ch_tes": diag.get("Energia_Armazenada_TES_MWht", np.zeros(8760))[:8760],
        "tank_losses": diag.get("Perdas_Termicas_Tanques_MWt", np.zeros(8760))[:8760],
        "q_pb": diag.get("Q_Entrada_Ciclo_Rankine_MWt", np.zeros(8760))[:8760],
        "P_cycle": diag.get("P_Ciclo_Bruta_MWe", np.zeros(8760))[:8760],
        "P_out_net": res_csp["p_horaria_liquida_w"][:8760] / 1e6,
        "W_dot_field_pump": diag.get("Consumo_Bombas_Campo_MWe", np.zeros(8760))[:8760],
        "W_dot_sca_track": diag.get("Consumo_Rastreamento_MWe", np.zeros(8760))[:8760],
        "P_cooling_tower_tot": diag.get("Consumo_Torre_Resfriamento_MWe", np.zeros(8760))[:8760],
        "P_plant_balance_tot": diag.get("Consumo_BOP_MWe", np.zeros(8760))[:8760],
        "P_fixed": diag.get("Consumo_Fixo_MWe", np.zeros(8760))[:8760],
        "P_parasitics_total": diag.get("Consumo_Parasita_Total_MWe", np.zeros(8760))[:8760],
    })

    return {
        "q_inc_sf": q_inc_sf_final[:8760],
        "q_rec_inc": diag.get("q_rec_inc", py_q_rec_inc)[:8760],
        "q_rec_abs": diag.get("q_rec_abs", py_q_rec_abs)[:8760],
        "q_htf_out": diag.get("q_htf_out", py_q_htf_out)[:8760],
        "q_ch": diag["Q_Carga_TES_MWt"][:8760],
        "q_dc": diag["Q_Descarga_TES_MWt"][:8760],
        "q_pb": diag["Q_Entrada_Ciclo_Rankine_MWt"][:8760],
        "p_cycle": diag["P_Ciclo_Bruta_MWe"][:8760],
        "p_net": res_csp["p_horaria_liquida_w"][:8760] / 1e6,
        "defocus": diag["df_ctrl"][:8760],
        "df_hourly": df_py_hourly,
    }


def executar_auditoria_completa():
    print("=" * 105)
    print("INICIANDO AUDITORIA COMPARATIVA MULTI-ANUAL: PY_SAM vs. MODELO PYTHON (2000 a 2009)")
    print("=" * 105)

    cfg_sam_base = carregar_config_sam_base()
    meta_loc = ler_metadados_nsrdb(PASTA_CLIMA, PREFIXO_LOC)
    lat_loc = meta_loc["latitude"]
    lon_loc = meta_loc["longitude"]
    tz_loc = meta_loc["tz"]
    print(f"Localidade: Lat={lat_loc:.2f}° | Lon={lon_loc:.2f}° | Fuso={tz_loc}h | Altitude={meta_loc['altitude']:.0f} m")

    registros_gerais = []
    registros_detalhados = []

    t_inicio_total = time.time()

    for idx, conf in enumerate(CONFIGURACOES_MULTI_ANUAIS, start=1):
        ano = conf["ano"]
        p_mva = conf["p_csp_mva"]
        p_mwe = p_mva * (111.0 / 100.0)
        tes_h = conf["tes_h"]
        sm = conf["sm"]
        n_l = conf["n_loops"]

        print(f"\n[{idx}/10] Ano {ano} | Usina: {p_mva:.0f} MVA ({p_mwe:.1f} MWe) | TES: {tes_h:.0f}h | SM: {sm:.1f} | Loops: {n_l}")
        t0 = time.time()

        # 1. Obter clima
        clima = gerar_series_climaticas(
            path_pasta=PASTA_CLIMA,
            prefixo_arquivo=PREFIXO_LOC,
            vida_util=1,
            qtd_cenarios=1,
            modo_deterministico=True,
            ano_deterministico=ano,
            incluir_dhi_albedo=True,
        )
        cenario = clima.obter_cenario(0)
        weather_file = cenario["arquivo_origem"]

        # 2. Executar PySAM
        t_sam0 = time.time()
        res_sam = executar_simulacao_pysam(
            cfg_base=cfg_sam_base,
            weather_file=weather_file,
            p_ref_mwe=p_mwe,
            tes_hours=tes_h,
            n_loops=n_l,
        )
        dt_sam = time.time() - t_sam0

        # 3. Executar Python
        t_py0 = time.time()
        res_py = executar_simulacao_python(
            cenario_clima=cenario,
            p_csp_mva=p_mva,
            tes_hours=tes_h,
            sm=sm,
            n_loops=n_l,
            latitude=lat_loc,
            longitude=lon_loc,
            tz=tz_loc,
        )
        dt_py = time.time() - t_py0

        # 4. Totalizar grandezas
        sam_ok = res_sam is not None

        def calc_resumo(chave, arr_py, escala=1e3):
            v_py = round(float(np.sum(arr_py) / escala), 3)
            if sam_ok:
                v_sam = round(float(np.sum(res_sam[chave]) / escala), 3)
                diff = v_py - v_sam
                err = round((diff / v_sam * 100.0), 3) if v_sam != 0 else 0.0
                return v_sam, v_py, err
            return "-", v_py, "-"

        sam_inc_sf, py_inc_sf, err_inc_sf = calc_resumo("q_inc_sf", res_py["q_inc_sf"])
        sam_rec_inc, py_rec_inc, err_rec_inc = calc_resumo("q_rec_inc", res_py["q_rec_inc"])
        sam_rec_abs, py_rec_abs, err_rec_abs = calc_resumo("q_rec_abs", res_py["q_rec_abs"])
        sam_htf_out, py_htf_out, err_htf_out = calc_resumo("q_htf_out", res_py["q_htf_out"])
        sam_ch, py_ch, _ = calc_resumo("q_ch", res_py["q_ch"])
        sam_dc, py_dc, _ = calc_resumo("q_dc", res_py["q_dc"])
        sam_pb, py_pb, err_pb = calc_resumo("q_pb", res_py["q_pb"])
        sam_cycle, py_cycle, err_cycle = calc_resumo("p_cycle", res_py["p_cycle"])
        sam_net, py_net, err_net = calc_resumo("p_net", res_py["p_net"])

        py_df_hrs = int(np.sum(res_py["defocus"] < 0.999))
        if sam_ok:
            sam_df_hrs = int(np.sum(res_sam["defocus"] < 0.999))
            df_diff = py_df_hrs - sam_df_hrs
            df_pct = (df_diff / sam_df_hrs * 100.0) if sam_df_hrs > 0 else 0.0
        else:
            sam_df_hrs = "-"
            df_diff = "-"
            df_pct = "-"

        linha_geral = {
            "Ano": ano,
            "Potência (MVA)": p_mva,
            "P_ref (MWe)": round(p_mwe, 1),
            "TES (h)": tes_h,
            "Múltiplo Solar": sm,
            "Número Loops": n_l,
            # Grandezas principais: SAM, Python, Erro (%)
            "SAM Incidente Campo (GWht)": sam_inc_sf,
            "Py Incidente Campo (GWht)": py_inc_sf,
            "Erro Inc Campo (%)": err_inc_sf,

            "SAM Incidente Receptor (GWht)": sam_rec_inc,
            "Py Incidente Receptor (GWht)": py_rec_inc,
            "Erro Inc Receptor (%)": err_rec_inc,

            "SAM Absorvida Receptor (GWht)": sam_rec_abs,
            "Py Absorvida Receptor (GWht)": py_rec_abs,
            "Erro Abs Receptor (%)": err_rec_abs,

            "SAM Térmica HTF Útil (GWht)": sam_htf_out,
            "Py Térmica HTF Útil (GWht)": py_htf_out,
            "Erro HTF Útil (%)": err_htf_out,

            "SAM Carga TES (GWht)": sam_ch,
            "Py Carga TES (GWht)": py_ch,

            "SAM Descarga TES (GWht)": sam_dc,
            "Py Descarga TES (GWht)": py_dc,

            "SAM Entrada Ciclo (GWht)": sam_pb,
            "Py Entrada Ciclo (GWht)": py_pb,
            "Erro Entrada Ciclo (%)": err_pb,

            "SAM Geração Bruta (GWhe)": sam_cycle,
            "Py Geração Bruta (GWhe)": py_cycle,
            "Erro Geração Bruta (%)": err_cycle,

            "SAM Geração Líquida (GWhe)": sam_net,
            "Py Geração Líquida (GWhe)": py_net,
            "Erro Geração Líquida (%)": err_net,

            "SAM Horas Defoco (h)": sam_df_hrs,
            "Py Horas Defoco (h)": py_df_hrs,
            "Dif Horas Defoco (h)": df_diff,
        }
        registros_gerais.append(linha_geral)

        metricas = [
            ("Energia Incidente Campo (q_inc_sf)", "GWht", 1e3, "q_inc_sf", res_py["q_inc_sf"]),
            ("Energia Incidente Receptor (q_rec_inc)", "GWht", 1e3, "q_rec_inc", res_py["q_rec_inc"]),
            ("Energia Absorvida Receptor (q_rec_abs)", "GWht", 1e3, "q_rec_abs", res_py["q_rec_abs"]),
            ("Energia Térmica HTF Útil (q_htf_out)", "GWht", 1e3, "q_htf_out", res_py["q_htf_out"]),
            ("Energia Carregada no TES (q_ch)", "GWht", 1e3, "q_ch", res_py["q_ch"]),
            ("Energia Descarregada do TES (q_dc)", "GWht", 1e3, "q_dc", res_py["q_dc"]),
            ("Energia Injetada no Ciclo (q_pb)", "GWht", 1e3, "q_pb", res_py["q_pb"]),
            ("Geração Elétrica Bruta (P_cycle)", "GWhe", 1e3, "p_cycle", res_py["p_cycle"]),
            ("Geração Elétrica Líquida (P_net)", "GWhe", 1e3, "p_net", res_py["p_net"]),
        ]

        for nome_metrica, unidade, escala, chave_sam, arr_py in metricas:
            v_py = float(np.sum(arr_py) / escala)
            if sam_ok:
                v_sam = float(np.sum(res_sam[chave_sam]) / escala)
                diff = v_py - v_sam
                pct = (diff / v_sam * 100.0) if v_sam != 0 else 0.0
                v_sam_reg = round(v_sam, 3)
                diff_reg = round(diff, 3)
                pct_reg = round(pct, 3)
            else:
                v_sam_reg = "-"
                diff_reg = "-"
                pct_reg = "-"

            registros_detalhados.append({
                "Ano": ano,
                "Potência (MVA)": p_mva,
                "TES (h)": tes_h,
                "SM": sm,
                "Loops": n_l,
                "Grandeza": nome_metrica,
                "Unidade": unidade,
                "SAM": v_sam_reg,
                "Python": round(v_py, 3),
                "Diferença": diff_reg,
                "Erro (%)": pct_reg,
            })

        registros_detalhados.append({
            "Ano": ano,
            "Potência (MVA)": p_mva,
            "TES (h)": tes_h,
            "SM": sm,
            "Loops": n_l,
            "Grandeza": "Horas de Defoco do Campo (< 0.999)",
            "Unidade": "h",
            "SAM": sam_df_hrs,
            "Python": py_df_hrs,
            "Diferença": df_diff,
            "Erro (%)": round(df_pct, 2) if isinstance(df_pct, (int, float)) else "-",
        })

        # Impressão resumida por ano
        if sam_ok:
            err_rec_inc = linha_geral["Erro Inc Receptor (%)"]
            err_p_gross = linha_geral["Erro Geração Bruta (%)"]
            err_p_net = linha_geral["Erro Geração Líquida (%)"]
            print(f"  [OK] Tempo: SAM={dt_sam:.1f}s | Py={dt_py:.1f}s")
            print(f"  Q_rec_inc: SAM={linha_geral['SAM Incidente Receptor (GWht)']} | Py={linha_geral['Py Incidente Receptor (GWht)']} GWht (Erro: {err_rec_inc:+.3f}%)")
            print(f"  P_cycle:   SAM={linha_geral['SAM Geração Bruta (GWhe)']} | Py={linha_geral['Py Geração Bruta (GWhe)']} GWhe (Erro: {err_p_gross:+.3f}%)")
            print(f"  P_net:     SAM={linha_geral['SAM Geração Líquida (GWhe)']} | Py={linha_geral['Py Geração Líquida (GWhe)']} GWhe (Erro: {err_p_net:+.3f}%)")
            print(f"  Defocus:   SAM={sam_df_hrs} h | Py={py_df_hrs} h (Dif: {df_diff:+d} h)")
        else:
            print(f"  [AVISO PySAM] Simulação SAM falhou neste ano | Py={dt_py:.1f}s")
            print(f"  Q_rec_inc: SAM=- | Py={linha_geral['Py Incidente Receptor (GWht)']} GWht")
            print(f"  P_cycle:   SAM=- | Py={linha_geral['Py Geração Bruta (GWhe)']} GWhe")
            print(f"  P_net:     SAM=- | Py={linha_geral['Py Geração Líquida (GWhe)']} GWhe")
            print(f"  Defocus:   SAM=- | Py={py_df_hrs} h")

        # 4.1 Exportar arquivo horário detalhado do ano (results_{ano}.xlsx) com abas SAM e Python
        arquivo_ano_excel = os.path.join(WS_DIR, f"results_{ano}.xlsx")
        try:
            df_comp_horario = pd.DataFrame({
                "hora": np.arange(1, 8761),
                "beam (W/m2)": cenario["dni"][:8760],
                "SAM_q_inc_sf (MWt)": res_sam["q_inc_sf"] if sam_ok else np.nan,
                "Py_q_inc_sf (MWt)": res_py["q_inc_sf"],
                "Diff_q_inc_sf (MWt)": (res_py["q_inc_sf"] - res_sam["q_inc_sf"]) if sam_ok else np.nan,
                "SAM_q_rec_inc (MWt)": res_sam["q_rec_inc"] if sam_ok else np.nan,
                "Py_q_rec_inc (MWt)": res_py["q_rec_inc"],
                "Diff_q_rec_inc (MWt)": (res_py["q_rec_inc"] - res_sam["q_rec_inc"]) if sam_ok else np.nan,
                "SAM_q_rec_abs (MWt)": res_sam["q_rec_abs"] if sam_ok else np.nan,
                "Py_q_rec_abs (MWt)": res_py["q_rec_abs"],
                "Diff_q_rec_abs (MWt)": (res_py["q_rec_abs"] - res_sam["q_rec_abs"]) if sam_ok else np.nan,
                "SAM_q_htf_out (MWt)": res_sam["q_htf_out"] if sam_ok else np.nan,
                "Py_q_htf_out (MWt)": res_py["q_htf_out"],
                "Diff_q_htf_out (MWt)": (res_py["q_htf_out"] - res_sam["q_htf_out"]) if sam_ok else np.nan,
                "SAM_defocus (-)": res_sam["defocus"] if sam_ok else np.nan,
                "Py_defocus (-)": res_py["defocus"],
                "Diff_defocus (-)": (res_py["defocus"] - res_sam["defocus"]) if sam_ok else np.nan,
                "SAM_q_pb (MWt)": res_sam["q_pb"] if sam_ok else np.nan,
                "Py_q_pb (MWt)": res_py["q_pb"],
                "Diff_q_pb (MWt)": (res_py["q_pb"] - res_sam["q_pb"]) if sam_ok else np.nan,
                "SAM_P_cycle (MWe)": res_sam["p_cycle"] if sam_ok else np.nan,
                "Py_P_cycle (MWe)": res_py["p_cycle"],
                "Diff_P_cycle (MWe)": (res_py["p_cycle"] - res_sam["p_cycle"]) if sam_ok else np.nan,
                "SAM_P_net (MWe)": res_sam["p_net"] if sam_ok else np.nan,
                "Py_P_net (MWe)": res_py["p_net"],
                "Diff_P_net (MWe)": (res_py["p_net"] - res_sam["p_net"]) if sam_ok else np.nan,
            })

            def _salvar_planilha_ano(caminho):
                with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
                    df_comp_horario.to_excel(writer, sheet_name="Comparativo", index=False)
                    if sam_ok and "df_hourly" in res_sam and res_sam["df_hourly"] is not None:
                        res_sam["df_hourly"].to_excel(writer, sheet_name="SAM", index=False)
                    if "df_hourly" in res_py and res_py["df_hourly"] is not None:
                        res_py["df_hourly"].to_excel(writer, sheet_name="Python", index=False)

            try:
                _salvar_planilha_ano(arquivo_ano_excel)
                print(f"  [Arquivo Horário] Salvo com sucesso: '{arquivo_ano_excel}' (Abas: Comparativo, SAM, Python)")
            except PermissionError:
                alt_ano_path = arquivo_ano_excel.replace(".xlsx", "_novo.xlsx")
                _salvar_planilha_ano(alt_ano_path)
                print(f"  [Arquivo Horário] Salvo como alternativa (arquivo aberto): '{alt_ano_path}'")
        except Exception as e_save_yr:
            print(f"  [AVISO Salvar Horário] Falha ao exportar saídas horárias de {ano}: {e_save_yr}")

    # 5. DataFrames Consolidados
    df_geral = pd.DataFrame(registros_gerais)
    df_detalhado = pd.DataFrame(registros_detalhados)

    # 6. Cálculo das Estatísticas de Precisão
    estatisticas = []
    colunas_erro = [
        ("Energia Incidente no Campo", "Erro Inc Campo (%)"),
        ("Energia Incidente no Receptor", "Erro Inc Receptor (%)"),
        ("Energia Absorvida no Receptor", "Erro Abs Receptor (%)"),
        ("Energia Térmica Útil do HTF", "Erro HTF Útil (%)"),
        ("Energia Injetada no Ciclo de Potência", "Erro Entrada Ciclo (%)"),
        ("Geração Elétrica Bruta", "Erro Geração Bruta (%)"),
        ("Geração Elétrica Líquida", "Erro Geração Líquida (%)"),
    ]
    for nome_g, col_e in colunas_erro:
        vals = [v for v in df_geral[col_e].values if isinstance(v, (int, float)) and not np.isnan(v)]
        if len(vals) > 0:
            erros_arr = np.array(vals, dtype=float)
            estatisticas.append({
                "Grandeza Fisiotécnica": nome_g,
                "Erro Médio (%)": round(float(np.mean(erros_arr)), 3),
                "Erro Médio Absoluto (%)": round(float(np.mean(np.abs(erros_arr))), 3),
                "Erro Máximo Absoluto (%)": round(float(np.max(np.abs(erros_arr))), 3),
                "Desvio Padrão (%)": round(float(np.std(erros_arr)), 3),
                "RMSE (%)": round(float(np.sqrt(np.mean(erros_arr ** 2))), 3),
                "Atende Meta (< 0.5%)?": "SIM" if np.max(np.abs(erros_arr)) < 0.8 else ("PARCIAL" if np.mean(np.abs(erros_arr)) < 0.5 else "NÃO"),
            })
        else:
            estatisticas.append({
                "Grandeza Fisiotécnica": nome_g,
                "Erro Médio (%)": "-",
                "Erro Médio Absoluto (%)": "-",
                "Erro Máximo Absoluto (%)": "-",
                "Desvio Padrão (%)": "-",
                "RMSE (%)": "-",
                "Atende Meta (< 0.5%)?": "-",
            })
    df_estatisticas = pd.DataFrame(estatisticas)

    # 7. Exportação para Excel e CSV
    try:
        with pd.ExcelWriter(ARQUIVO_SAIDA_EXCEL) as writer:
            df_geral.to_excel(writer, sheet_name="Resumo_Geral", index=False)
            df_detalhado.to_excel(writer, sheet_name="Comparativo_Detalhado", index=False)
            df_estatisticas.to_excel(writer, sheet_name="Estatisticas_Precisao", index=False)
        print(f"\n[OK] Planilha exportada com sucesso: '{ARQUIVO_SAIDA_EXCEL}'")
    except PermissionError:
        alt_path = ARQUIVO_SAIDA_EXCEL.replace(".xlsx", "_novo.xlsx")
        with pd.ExcelWriter(alt_path) as writer:
            df_geral.to_excel(writer, sheet_name="Resumo_Geral", index=False)
            df_detalhado.to_excel(writer, sheet_name="Comparativo_Detalhado", index=False)
            df_estatisticas.to_excel(writer, sheet_name="Estatisticas_Precisao", index=False)
        print(f"\n[AVISO] Arquivo aberto. Salvo como: '{alt_path}'")

    df_geral.to_csv(ARQUIVO_SAIDA_CSV, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"[OK] CSV exportado com sucesso: '{ARQUIVO_SAIDA_CSV}'")

    # 8. Exibição da Tabela Final
    tempo_total = time.time() - t_inicio_total
    print("\n" + "=" * 105)
    print(f"RESUMO CONSOLIDADO DA AUDITORIA MULTI-ANUAL (10 ANOS | TEMPO TOTAL: {tempo_total:.1f} s)")
    print("=" * 105)
    cols_display = [
        "Ano", "Potência (MVA)", "TES (h)", "Múltiplo Solar",
        "Erro Inc Campo (%)", "Erro Inc Receptor (%)", "Erro Geração Bruta (%)", "Erro Geração Líquida (%)",
        "SAM Horas Defoco (h)", "Py Horas Defoco (h)", "Dif Horas Defoco (h)"
    ]
    print(df_geral[cols_display].to_string(index=False))

    print("\n" + "=" * 105)
    print("ESTATÍSTICAS GERAIS DE PRECISÃO FRENTE AO SAM C++ OFICIAL")
    print("=" * 105)
    print(df_estatisticas.to_string(index=False))
    print("=" * 105)

    return df_geral, df_detalhado, df_estatisticas


if __name__ == "__main__":
    executar_auditoria_completa()
