# -*- coding: utf-8 -*-
"""
Módulo Unificado de Geração de Séries Climáticas (Estocástico e Determinístico).
Atende simultaneamente os modelos de CSP, Fotovoltaico (PV), Bateria (BESS) e Eólica.
"""

from dataclasses import dataclass
import gc
from time import time
from typing import Optional

import numpy as np
import pandas as pd

from funcoes_estatisticas import (
    calibrar_parametros_mrm_a,
    calibrar_parametros_mrm_g,
    extrair_sazonalidade,
    gerar_dni_sintetica,
    simular_mrm_a,
    simular_mrm_g,
)


@dataclass
class SeriesClimaticas:
    """
    Container padronizado para as séries climáticas geradas.
    Todas as matrizes possuem formato (n_horas, qtd_cenarios).
    Para acessar o vetor 1D de um cenário 'c', basta usar: clima.dni[:, c]
    """
    dni: np.ndarray            # Irradiância Normal Direta [W/m²]
    dhi: np.ndarray            # Irradiância Horizontal Difusa [W/m²]
    albedo: np.ndarray         # Albedo de superfície [-]
    temp: np.ndarray           # Temperatura ambiente [°C]
    press: np.ndarray          # Pressão atmosférica [mbar]
    vento_vel: np.ndarray      # Velocidade do vento a 10m [m/s]
    vento_dir: np.ndarray      # Direção do vento [graus]
    vida_util: int             # Quantidade de anos simulados
    qtd_cenarios: int          # Quantidade de cenários gerados
    n_horas: int               # Total de horas (8760 * vida_util)

    def obter_cenario(self, idx_cenario: int = 0) -> dict:
        """Retorna um dicionário com arrays 1D de um único cenário para fácil consumo."""
        return {
            "dni": self.dni[:, idx_cenario],
            "dhi": self.dhi[:, idx_cenario],
            "albedo": self.albedo[:, idx_cenario],
            "temp": self.temp[:, idx_cenario],
            "press": self.press[:, idx_cenario],
            "vento_vel": self.vento_vel[:, idx_cenario],
            "vento_dir": self.vento_dir[:, idx_cenario],
        }


def carregar_historico_nsrdb(
    path_pasta: str,
    prefixo_arquivo: str = "1945275_-13.27_-43.42",
    ano_inicio: int = 1998,
    ano_fim: int = 2025,
) -> pd.DataFrame:
    """
    Lê e concatena os arquivos CSV históricos do NSRDB, configurando o índice temporal
    e removendo dias bissextos (29 de fevereiro) para garantir múltiplos exatos de 8760h.
    """
    time_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    lista_df_anos = []

    print(f"Lendo base histórica ({ano_inicio} a {ano_fim}) em '{path_pasta}'...")
    for ano in range(ano_inicio, ano_fim + 1):
        arquivo = f"{path_pasta}/{prefixo_arquivo}_{ano}.csv"
        df_ano = pd.read_csv(
            filepath_or_buffer=arquivo,
            skiprows=2,
            sep=',',
            decimal='.',
        )
        df_ano['timestamp'] = pd.to_datetime(df_ano[time_cols])
        df_ano = df_ano.set_index('timestamp').drop(columns=time_cols)

        # Garante remoção de 29 de fevereiro caso algum arquivo tenha 8784 horas
        mask_29fev = (df_ano.index.month == 2) & (df_ano.index.day == 29)
        if mask_29fev.any():
            df_ano = df_ano.loc[~mask_29fev]

        lista_df_anos.append(df_ano)

    df_clima = pd.concat(lista_df_anos)
    del lista_df_anos
    gc.collect()
    return df_clima


def gerar_series_climaticas(
    path_pasta: str = 'historico_variaveis/nsrdb_novo',
    prefixo_arquivo: str = '1945275_-13.27_-43.42',
    latitude: float = -13.27,
    longitude: float = -43.42,
    altitude: float = 438.0,
    tz: str = 'Etc/GMT+3',
    vida_util: int = 30,
    qtd_cenarios: int = 1,
    modo_deterministico: bool = False,
    ano_deterministico: int = 2021,
    incluir_dhi_albedo: bool = True,
    ano_inicio_hist: int = 1998,
    ano_fim_hist: int = 2025,
    semente_rng: Optional[int] = None,
) -> SeriesClimaticas:
    """
    Gera as séries climáticas (determinísticas ou sintéticas via MRM + Cholesky).

    Parâmetros principais:
    - incluir_dhi_albedo:
        True  -> Sorteia as 6 variáveis ['DHI', 'DNI', 'Surface Albedo', 'Temperature', 'Pressure', 'Wind Speed'] (Ideal para PV + CSP + Eólica).
        False -> Sorteia apenas as 4 variáveis ['DNI', 'Temperature', 'Pressure', 'Wind Speed'] (Modo clássico CSP).
    """
    n_horas = 8760 * vida_util
    time_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']

    # ==========================================================================
    # MODO 1: DETERMINÍSTICO (Lê apenas 1 arquivo histórico e replica por vida_util)
    # ==========================================================================
    if modo_deterministico:
        arquivo = f"{path_pasta}/{prefixo_arquivo}_{ano_deterministico}.csv"
        print(f"[Clima Determinístico] Carregando arquivo: {arquivo}...")

        df_det = pd.read_csv(filepath_or_buffer=arquivo, skiprows=2, sep=',', decimal='.')
        df_det['timestamp'] = pd.to_datetime(df_det[time_cols])
        df_det = df_det.set_index('timestamp').drop(columns=time_cols)

        mask_29fev = (df_det.index.month == 2) & (df_det.index.day == 29)
        if mask_29fev.any():
            df_det = df_det.loc[~mask_29fev]

        def expandir_para_matriz(vetor_1d: np.ndarray) -> np.ndarray:
            vetor_longo = np.tile(vetor_1d[:8760], vida_util)
            return np.repeat(vetor_longo[:, np.newaxis], qtd_cenarios, axis=1)

        dni_mat = expandir_para_matriz(df_det['DNI'].to_numpy(dtype=float))
        dhi_mat = (
            expandir_para_matriz(df_det['DHI'].to_numpy(dtype=float))
            if 'DHI' in df_det.columns else np.zeros((n_horas, qtd_cenarios))
        )
        albedo_mat = (
            expandir_para_matriz(df_det['Surface Albedo'].to_numpy(dtype=float))
            if 'Surface Albedo' in df_det.columns else np.full((n_horas, qtd_cenarios), 0.2)
        )
        temp_mat = expandir_para_matriz(df_det['Temperature'].to_numpy(dtype=float))
        press_mat = expandir_para_matriz(df_det['Pressure'].to_numpy(dtype=float))
        vento_vel_mat = expandir_para_matriz(df_det['Wind Speed'].to_numpy(dtype=float))
        vento_dir_mat = (
            expandir_para_matriz(df_det['Wind Direction'].to_numpy(dtype=float))
            if 'Wind Direction' in df_det.columns else np.zeros((n_horas, qtd_cenarios))
        )

        return SeriesClimaticas(
            dni=dni_mat,
            dhi=dhi_mat,
            albedo=albedo_mat,
            temp=temp_mat,
            press=press_mat,
            vento_vel=vento_vel_mat,
            vento_dir=vento_dir_mat,
            vida_util=vida_util,
            qtd_cenarios=qtd_cenarios,
            n_horas=n_horas,
        )

    # ==========================================================================
    # MODO 2: ESTOCÁSTICO (Sazonalidade + MRM-A / MRM-G + Decomposição de Cholesky)
    # ==========================================================================
    t0 = time()
    df_clima = carregar_historico_nsrdb(path_pasta, prefixo_arquivo, ano_inicio_hist, ano_fim_hist)

    serie_dni = df_clima['DNI']
    serie_temp = df_clima['Temperature']
    serie_press = df_clima['Pressure']
    serie_wind = df_clima['Wind Speed']

    # Direção do vento histórica para alimentar o modelo de esteira eólico
    if 'Wind Direction' in df_clima.columns:
        dir_hist_base = df_clima['Wind Direction'].to_numpy(dtype=float)
        repeticoes_dir = int(np.ceil(n_horas / len(dir_hist_base)))
        dir_longo = np.tile(dir_hist_base, repeticoes_dir)[:n_horas]
        vento_dir_sint = np.repeat(dir_longo[:, np.newaxis], qtd_cenarios, axis=1)
    else:
        vento_dir_sint = np.zeros((n_horas, qtd_cenarios))

    print("Extraindo sazonalidades e calibrando parâmetros MRM...")
    data_ref_sazon = f'{ano_inicio_hist}-01-01'

    i_temp, temp_s_sazon = extrair_sazonalidade(serie_temp, data_ref_sazon)
    i_press, press_s_sazon = extrair_sazonalidade(serie_press, data_ref_sazon)
    i_vento, vento_s_sazon = extrair_sazonalidade(serie_wind, data_ref_sazon)

    eta_temp, x_barra_temp, sigma_temp, _ = calibrar_parametros_mrm_a(temp_s_sazon, dt=1)
    eta_press, x_barra_press, sigma_press, _ = calibrar_parametros_mrm_a(press_s_sazon, dt=1)
    eta_vento, x_barra_vento, sigma_vento, _ = calibrar_parametros_mrm_g(vento_s_sazon, dt=1)

    i_temp_longo = np.tile(i_temp, vida_util)
    i_press_longo = np.tile(i_press, vida_util)
    i_vento_longo = np.tile(i_vento, vida_util)

    anos = range(ano_inicio_hist, ano_inicio_hist + vida_util)
    rng = np.random.default_rng(semente_rng)

    if incluir_dhi_albedo:
        # --- Configuração de 6 Variáveis (PV + CSP + Eólica) ---
        serie_dhi = df_clima['DHI']
        serie_albedo = df_clima['Surface Albedo']

        i_albedo, albedo_s_sazon = extrair_sazonalidade(serie_albedo, data_ref_sazon)
        eta_albedo, x_barra_albedo, sigma_albedo, _ = calibrar_parametros_mrm_a(albedo_s_sazon, dt=1)
        i_albedo_longo = np.tile(i_albedo, vida_util)

        colunas_corr = ['DHI', 'DNI', 'Surface Albedo', 'Temperature', 'Pressure', 'Wind Speed']
        m_correlacao = df_clima[colunas_corr].corr()
        Z = np.linalg.cholesky(m_correlacao)
        print("\nMatriz de Correlação (6 variáveis):")
        print(m_correlacao)

        sorteios_ind = rng.standard_normal(size=(n_horas, qtd_cenarios, 6))
        sorteios_corr = sorteios_ind @ Z.T
        del sorteios_ind

        eps_dhi = sorteios_corr[:, :, 0]
        eps_dni = sorteios_corr[:, :, 1]
        eps_albedo = sorteios_corr[:, :, 2]
        eps_temp = sorteios_corr[:, :, 3]
        eps_press = sorteios_corr[:, :, 4]
        eps_vento = sorteios_corr[:, :, 5]
        del sorteios_corr
        gc.collect()

        dhi_sint = gerar_dni_sintetica(
            latitude, longitude, altitude, tz, df_clima, anos,
            sorteios=eps_dhi, coluna_irradiancia='DHI', intervalo_confianca=0.99,
        ).reshape(n_horas, qtd_cenarios)
        del eps_dhi

        albedo_sint = simular_mrm_a(
            var_inicial_limpa=albedo_s_sazon.iloc[-1],
            var_inicial_real=serie_albedo.iloc[-1],
            x_barra=x_barra_albedo, eta=eta_albedo, sigma=sigma_albedo,
            ind_sazon_futuro=i_albedo_longo, passos_tempo=n_horas,
            cenarios=qtd_cenarios, dt=1, eps_ruido=eps_albedo,
        )[1:].reshape(n_horas, qtd_cenarios)
        del eps_albedo
        gc.collect()

    else:
        # --- Configuração de 4 Variáveis (Exclusivo CSP / Eólica) ---
        colunas_corr = ['DNI', 'Temperature', 'Pressure', 'Wind Speed']
        m_correlacao = df_clima[colunas_corr].corr()
        Z = np.linalg.cholesky(m_correlacao)
        print("\nMatriz de Correlação (4 variáveis):")
        print(m_correlacao)

        sorteios_ind = rng.standard_normal(size=(n_horas, qtd_cenarios, 4))
        sorteios_corr = sorteios_ind @ Z.T
        del sorteios_ind

        eps_dni = sorteios_corr[:, :, 0]
        eps_temp = sorteios_corr[:, :, 1]
        eps_press = sorteios_corr[:, :, 2]
        eps_vento = sorteios_corr[:, :, 3]
        del sorteios_corr
        gc.collect()

        dhi_sint = np.zeros((n_horas, qtd_cenarios))
        albedo_sint = np.zeros((n_horas, qtd_cenarios))

    # --- Simulação das 4 variáveis comuns a todas as configurações ---
    try:
        dni_sint = gerar_dni_sintetica(
            latitude, longitude, altitude, tz, df_clima, anos,
            sorteios=eps_dni, coluna_irradiancia='DNI', intervalo_confianca=0.99,
        ).reshape(n_horas, qtd_cenarios)
    except TypeError:
        # Compatibilidade caso a versão de 'gerar_dni_sintetica' não possua 'coluna_irradiancia'
        dni_sint = gerar_dni_sintetica(
            latitude, longitude, altitude, tz, df_clima, anos,
            sorteios=eps_dni, intervalo_confianca=0.99,
        ).reshape(n_horas, qtd_cenarios)

    del eps_dni
    gc.collect()

    temp_sint = simular_mrm_a(
        var_inicial_limpa=temp_s_sazon.iloc[-1],
        var_inicial_real=serie_temp.iloc[-1],
        x_barra=x_barra_temp, eta=eta_temp, sigma=sigma_temp,
        ind_sazon_futuro=i_temp_longo, passos_tempo=n_horas,
        cenarios=qtd_cenarios, dt=1, eps_ruido=eps_temp,
    )[1:].reshape(n_horas, qtd_cenarios)
    del eps_temp
    gc.collect()

    press_sint = simular_mrm_a(
        var_inicial_limpa=press_s_sazon.iloc[-1],
        var_inicial_real=serie_press.iloc[-1],
        x_barra=x_barra_press, eta=eta_press, sigma=sigma_press,
        ind_sazon_futuro=i_press_longo, passos_tempo=n_horas,
        cenarios=qtd_cenarios, dt=1, eps_ruido=eps_press,
    )[1:].reshape(n_horas, qtd_cenarios)
    del eps_press
    gc.collect()

    vento_vel_sint = simular_mrm_g(
        var_inicial_limpa=vento_s_sazon.iloc[-1],
        var_inicial_real=serie_wind.iloc[-1],
        x_barra=x_barra_vento, eta=eta_vento, sigma=sigma_vento,
        ind_sazon_futuro=i_vento_longo, passos_tempo=n_horas,
        cenarios=qtd_cenarios, dt=1, eps_ruido=eps_vento,
    )[1:].reshape(n_horas, qtd_cenarios)
    del eps_vento
    gc.collect()

    print(f"Séries sintéticas geradas com sucesso em {time() - t0:.2f} s!")

    return SeriesClimaticas(
        dni=dni_sint,
        dhi=dhi_sint,
        albedo=albedo_sint,
        temp=temp_sint,
        press=press_sint,
        vento_vel=vento_vel_sint,
        vento_dir=vento_dir_sint,
        vida_util=vida_util,
        qtd_cenarios=qtd_cenarios,
        n_horas=n_horas,
    )


# ==============================================================================
# EXEMPLO DE USO RÁPIDO DO MÓDULO
# ==============================================================================
if __name__ == "__main__":
    clima = gerar_series_climaticas(
        path_pasta='historico_variaveis/nsrdb_novo',
        vida_util=1,
        qtd_cenarios=1,
        modo_deterministico=False,
        incluir_dhi_albedo=True,
    )

    cenario_0 = clima.obter_cenario(0)
    print(f"Formato dos vetores do Cenário 0: {cenario_0['dni'].shape} horas")
    print(f"DNI médio: {cenario_0['dni'].mean():.2f} W/m² | Vento médio: {cenario_0['vento_vel'].mean():.2f} m/s")