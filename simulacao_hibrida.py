# -*- coding: utf-8 -*-
"""
Passo 3: Orquestrador de Simulação Híbrida e Isolada (CSP+TES, PV, Eólica e BESS).
Integra a geração climática unificada com os módulos físicos de geração e armazenamento,
permitindo simular múltiplas topologias sobre os mesmos cenários climáticos.
"""

import gc
from time import time
from typing import Dict

import numpy as np
import pandas as pd

from geracao_series_sinteticas import SeriesClimaticas, gerar_series_climaticas
from geracao_energia import (
    calc_angles_eff_csp,
    despachar_csp_tes,
    preparar_geometria_solar_pv,
    simular_despacho_bess,
    simular_geracao_eolica,
    simular_geracao_pv,
    simular_receptor_csp,
)

# ==============================================================================
# 1. PAINEL DE CONTROLE DA SIMULAÇÃO
# ==============================================================================

# Parâmetros Temporais e Climáticos
VIDA_UTIL_ANOS = 1                 # Ex: 1 para teste rápido ou 30 para vida útil completa
QTD_CENARIOS = 5                   # Quantidade de cenários de Monte Carlo
MODO_DETERMINISTICO = False         # True = lê 1 ano histórico fixo | False = MRM + Cholesky
ANO_DETERMINISTICO = 2020          # Ano usado caso MODO_DETERMINISTICO = True
SEMENTE_ALEATORIA = 42             # Inteiro para reprodutibilidade ou None para aleatório

# Caminhos de Arquivos
PATH_CLIMA_NSRDB = 'historico_variaveis/nsrdb_novo'
PREFIXO_ARQUIVO_CLIMA = '1945275_-13.27_-43.42'
ARQUIVO_JSON_EOLICA = r'hibrido\eolica\sam\eolica_windpower.json'

# Meta de Demanda Horária Flat no Barramento CA
DEMANDA_HORARIA_W = 100e6          # 100 MW flat [W]

# Dicionário de Combinações que serão avaliadas simultaneamente sobre o mesmo clima.
# Basta comentar/descomentar ou criar novas linhas conforme o estudo desejado:
COMBINACOES_ATIVAS: Dict[str, Dict[str, bool]] = {
    # --- Usinas Isoladas ---
    "01_PV_Isolado":           {"csp": False, "pv": True,  "eolica": False, "bess": False},
    "02_Eolica_Isolada":       {"csp": False, "pv": False, "eolica": True,  "bess": False},
    "03_CSP_TES_Isolado":      {"csp": True,  "pv": False, "eolica": False, "bess": False},

    # --- Híbridos Duplos ---
    "04_PV_BESS":              {"csp": False, "pv": True,  "eolica": False, "bess": True},
    "05_Eolica_BESS":          {"csp": False, "pv": False, "eolica": True,  "bess": True},
    "06_PV_Eolica":            {"csp": False, "pv": True,  "eolica": True,  "bess": False},
    "07_CSP_PV":               {"csp": True,  "pv": True,  "eolica": False, "bess": False},
    "08_CSP_Eolica":           {"csp": True,  "pv": False, "eolica": True,  "bess": False},

    # --- Híbridos Triplos e Completos ---
    "09_PV_Eolica_BESS":       {"csp": False, "pv": True,  "eolica": True,  "bess": True},
    "10_CSP_PV_BESS":          {"csp": True,  "pv": True,  "eolica": False, "bess": True},
    "11_CSP_PV_Eolica":        {"csp": True,  "pv": True,  "eolica": True,  "bess": False},
    "12_CSP_PV_Eolica_BESS":   {"csp": True,  "pv": True,  "eolica": True,  "bess": True},
}


# ==============================================================================
# 2. MOTOR ORQUESTRADOR DE CENÁRIOS E COMBINAÇÕES
# ==============================================================================

def executar_estudo_integrado(
    clima: SeriesClimaticas,
    combinacoes: Dict[str, Dict[str, bool]],
    demanda_horaria_w: float = 100e6,
    json_eolica: str = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Executa todas as combinações configuradas sobre os cenários climáticos gerados.
    Retorna:
      - df_resumo: Tabela consolidada com métricas médias de todos os cenários por arranjo.
      - dict_anuais_por_arranjo: Dicionário contendo os DataFrames de geração anual [GWh] de cada combinação.
    """
    n_horas = clima.n_horas
    vida_util = clima.vida_util
    qtd_cenarios = clima.qtd_cenarios

    # Identifica quais fontes físicas precisam ser pré-calculadas (evita cálculos desnecessários)
    precisa_csp = any(cfg.get("csp", False) for cfg in combinacoes.values())
    precisa_pv = any(cfg.get("pv", False) for cfg in combinacoes.values())
    precisa_eol = any(cfg.get("eolica", False) for cfg in combinacoes.values())

    # Pré-cálculos geométricos invariantes entre cenários (executados apenas 1 vez)
    opt_eff_csp = None
    aoi_pv, zenith_pv = None, None

    if precisa_csp:
        print("Pré-calculando geometria solar e eficiência óptica do CSP (8760h)...")
        opt_eff_csp = calc_angles_eff_csp()

    if precisa_pv:
        print("Pré-calculando geometria solar e ângulo de incidência do PV...")
        ano_base = ANO_DETERMINISTICO if MODO_DETERMINISTICO else 1998
        aoi_pv, zenith_pv = preparar_geometria_solar_pv(
            n_horas=n_horas, vida_util=vida_util, ano_base=ano_base
        )

    # Estruturas para armazenar resultados
    dict_anuais_por_arranjo = {nome: pd.DataFrame() for nome in combinacoes}
    registros_resumo = []

    # Loop pelos cenários climáticos (Monte Carlo ou Determinístico)
    for cen in range(qtd_cenarios):
        t_cen = time()
        print("\n" + "=" * 70)
        print(f"PROCESSANDO CENÁRIO CLIMÁTICO {cen + 1} / {qtd_cenarios}")
        print("=" * 70)

        cenario_clima = clima.obter_cenario(cen)

        # ----------------------------------------------------------------------
        # ETAPA A: Calcula a geração base das tecnologias ativas uma única vez
        # ----------------------------------------------------------------------
        if precisa_csp:
            t_op = time()
            print("  -> [1/3] Simulando campo solar, receptor e despacho térmico CSP+TES...")
            p_rec_csp, t_in_csp, _ = simular_receptor_csp(cenario_clima, opt_eff_csp)
            res_csp = despachar_csp_tes(p_in=p_rec_csp, t_htf_celsius=t_in_csp)
            p_csp_w = res_csp["p_horaria_liquida_w"]
            print(f"           CSP+TES concluído em {time() - t_op:.2f} s "
                  f"(Total: {np.sum(p_csp_w) / 1e9:.2f} GWh)")
        else:
            p_csp_w = np.zeros(n_horas)

        if precisa_pv:
            t_op = time()
            print("  -> [2/3] Simulando geração Fotovoltaica (PV)...")
            res_pv = simular_geracao_pv(cenario_clima, aoi_pv, zenith_pv)
            p_pv_w = res_pv["p_pv_ac_w"]
            print(f"           PV concluído em {time() - t_op:.2f} s "
                  f"(Total: {np.sum(p_pv_w) / 1e9:.2f} GWh)")
        else:
            p_pv_w = np.zeros(n_horas)

        if precisa_eol:
            t_op = time()
            print("  -> [3/3] Simulando geração do Parque Eólico...")
            res_eol = simular_geracao_eolica(cenario_clima, parametros_json=json_eolica)
            p_eol_w = res_eol["p_eolica_w"]
            print(f"           Eólica concluída em {time() - t_op:.2f} s "
                  f"(Total: {np.sum(p_eol_w) / 1e9:.2f} GWh)")
        else:
            p_eol_w = np.zeros(n_horas)

        # ----------------------------------------------------------------------
        # ETAPA B: Monta e avalia cada combinação de hibridização
        # ----------------------------------------------------------------------
        print("\n  -> Avaliando combinações e despacho do BESS...")
        for nome_arranjo, cfg in combinacoes.items():
            # Soma as curvas horárias das fontes geradoras habilitadas no arranjo
            p_barramento_ac_w = np.zeros(n_horas)
            if cfg.get("csp", False):
                p_barramento_ac_w = p_barramento_ac_w + p_csp_w
            if cfg.get("pv", False):
                p_barramento_ac_w = p_barramento_ac_w + p_pv_w
            if cfg.get("eolica", False):
                p_barramento_ac_w = p_barramento_ac_w + p_eol_w

            # Opera o banco de baterias (BESS) caso habilitado no arranjo
            if cfg.get("bess", False):
                res_bess = simular_despacho_bess(
                    p_entrada_ac_w=p_barramento_ac_w,
                    desejo_horario_w=demanda_horaria_w,
                )
                p_final_rede_w = res_bess["p_saida_rede_w"]
                qtd_trocas_bat = len(res_bess["trocas_bateria_anos"])
            else:
                p_final_rede_w = p_barramento_ac_w
                qtd_trocas_bat = 0

            # Energia horária limitada à meta de demanda (100 MW) vs. Energia Total
            p_firme_w = np.minimum(p_final_rede_w, demanda_horaria_w)
            p_excedente_w = np.maximum(0.0, p_final_rede_w - demanda_horaria_w)
            p_deficit_w = np.maximum(0.0, demanda_horaria_w - p_final_rede_w)

            # Agrupamento anual [GWh/ano]
            energia_anual_total_gwh = p_final_rede_w.reshape(vida_util, 8760).sum(axis=1) / 1e9
            dict_anuais_por_arranjo[nome_arranjo][f"Cenario_{cen + 1}"] = energia_anual_total_gwh

            # Métricas médias anuais do cenário
            media_anual_total_gwh = float(np.mean(energia_anual_total_gwh))
            media_anual_firme_gwh = float(np.sum(p_firme_w) / 1e9 / vida_util)
            media_anual_excedente_gwh = float(np.sum(p_excedente_w) / 1e9 / vida_util)
            media_anual_deficit_gwh = float(np.sum(p_deficit_w) / 1e9 / vida_util)

            # Fator de Atendimento da Meta de 100 MW (876 GWh/ano = 100%)
            energia_meta_anual_gwh = (demanda_horaria_w * 8760.0) / 1e9
            atendimento_meta_pct = (media_anual_firme_gwh / energia_meta_anual_gwh) * 100.0
            horas_plenas_pct = float(np.mean(p_final_rede_w >= demanda_horaria_w * 0.999) * 100.0)

            registros_resumo.append({
                "Arranjo": nome_arranjo,
                "Cenario": cen + 1,
                "CSP": "Sim" if cfg.get("csp") else "Não",
                "PV": "Sim" if cfg.get("pv") else "Não",
                "Eolica": "Sim" if cfg.get("eolica") else "Não",
                "BESS": "Sim" if cfg.get("bess") else "Não",
                "Energia_Total_Media_GWh_ano": round(media_anual_total_gwh, 2),
                "Energia_Firme_100MW_GWh_ano": round(media_anual_firme_gwh, 2),
                "Excedente_Acima_100MW_GWh_ano": round(media_anual_excedente_gwh, 2),
                "Deficit_Abaixo_100MW_GWh_ano": round(media_anual_deficit_gwh, 2),
                "Atendimento_Demanda_100MW_%": round(atendimento_meta_pct, 2),
                "Horas_Atendendo_100MW_%": round(horas_plenas_pct, 2),
                "Trocas_Bateria_Vida_Util": qtd_trocas_bat,
            })

        print(f"  -> Cenário {cen + 1} finalizado em {time() - t_cen:.2f} s.")
        gc.collect()

    df_detalhado = pd.DataFrame(registros_resumo)

    # Consolida a média entre todos os cenários simulados para cada Arranjo
    colunas_numericas = [
        "Energia_Total_Media_GWh_ano",
        "Energia_Firme_100MW_GWh_ano",
        "Excedente_Acima_100MW_GWh_ano",
        "Deficit_Abaixo_100MW_GWh_ano",
        "Atendimento_Demanda_100MW_%",
        "Horas_Atendendo_100MW_%",
        "Trocas_Bateria_Vida_Util",
    ]
    df_resumo = (
        df_detalhado.groupby(["Arranjo", "CSP", "PV", "Eolica", "BESS"], as_index=False)[colunas_numericas]
        .mean()
        .sort_values("Arranjo")
    )

    return df_resumo, dict_anuais_por_arranjo


# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    t_global = time()

    # Verifica se alguma combinação ativa precisa de DHI e Albedo (usados pelo PV)
    precisa_dhi_albedo = any(cfg.get("pv", False) for cfg in COMBINACOES_ATIVAS.values())

    # 1. Gera (ou carrega) as séries climáticas unificadas
    clima_gerado = gerar_series_climaticas(
        path_pasta=PATH_CLIMA_NSRDB,
        prefixo_arquivo=PREFIXO_ARQUIVO_CLIMA,
        vida_util=VIDA_UTIL_ANOS,
        qtd_cenarios=QTD_CENARIOS,
        modo_deterministico=MODO_DETERMINISTICO,
        ano_deterministico=ANO_DETERMINISTICO,
        incluir_dhi_albedo=precisa_dhi_albedo,
        semente_rng=SEMENTE_ALEATORIA,
    )

    # 2. Executa o estudo comparativo de todas as combinações
    df_comparativo, series_anuais = executar_estudo_integrado(
        clima=clima_gerado,
        combinacoes=COMBINACOES_ATIVAS,
        demanda_horaria_w=DEMANDA_HORARIA_W,
        json_eolica=ARQUIVO_JSON_EOLICA,
    )

    # 3. Exibe a tabela comparativa no console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 240)
    print("TABELA COMPARATIVA DE GERAÇÃO E COMPLEMENTARIDADE ENTRE TECNOLOGIAS (MÉDIA ANUAL)")
    print("=" * 240)
    print(df_comparativo.to_string(index=False))
    print("=" * 240)

    # 4. Exporta os resultados consolidados e por arranjo para um arquivo Excel
    arquivo_saida_excel = "resultados_comparativo_hibridizacao_st.xlsx"
    with pd.ExcelWriter(arquivo_saida_excel) as writer:
        df_comparativo.to_excel(writer, sheet_name="Resumo_Comparativo", index=False)
        for nome_arranjo, df_anual in series_anuais.items():
            # Limita o nome da aba a 31 caracteres (padrão do Excel)
            df_anual.to_excel(writer, sheet_name=nome_arranjo[:31], index_label="Ano_Operacao")

    print(f"\nPlanilha exportada com sucesso: '{arquivo_saida_excel}'")
    print(f"Tempo total de execução: {time() - t_global:.2f} s")