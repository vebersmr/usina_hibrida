# Este código utiliza a biblioteca PySAM para simular a geração de energia fotovoltaica
# Autor: Veber Samir 

# import PySAM.PySSC as PySSC
# import json
import os
import pandas as pd
import time
# import random
import numpy as np
# import PySAM.TroughPhysical as TroughPhysical
import matplotlib.pyplot as plt
from scipy.stats import linregress
# import statsmodels.api as sm
# from seaborn import heatmap
from statsmodels.tsa.stattools import adfuller
import src

# =====================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO E ANÁLISE
# =====================================================================

def aplicar_testes_adf(serie, nome_da_serie="Minha Série"):
    """
    Aplica o teste ADF em três especificações (AR, ARD, ARDT) 
    e imprime os resultados de forma formatada.
    """
    # Dicionário mapeando as siglas aos parâmetros do statsmodels
    modelos = {
        'AR (Sem constante)': 'n',
        'ARD (Com constante)': 'c',
        'ARDT (Com constante e tendência)': 'ct'
        # 'ARDTQ (Com constante e tendência linear e quadrática)': 'ctt' 
    }
    
    print(f"=== Análise de Estacionariedade: {nome_da_serie} ===")
    
    for nome_modelo, parametro_reg in modelos.items():
        try:
            # Executa o teste ADF
            resultado = adfuller(serie, regression=parametro_reg)
            
            # Extraindo os valores
            estatistica_t = resultado[0]
            valor_p = resultado[1]
            lags = resultado[2]
            
            # Definindo H0 com nível de significância de 5% (0.05)
            # 1 = Rejeita H0 (Estacionária) | 0 = Não rejeita H0 (Não-estacionária)
            h0 = 1 if valor_p < 0.05 else 0
            
            print(f"Modelo: {nome_modelo}")
            print(f"  Valor p      : {valor_p:.6f}")
            print(f"  Estatística t: {estatistica_t:.6f}")
            print(f"  H0           : {h0}")
            print(f"  Lags         : {lags}")
            
        except ValueError as e:
            # Captura erros caso a série seja muito curta ou não tenha variação
            print(f"\nModelo: {nome_modelo}")
            print(f"  Erro ao calcular: {e}")
            
    print("\n==================================================")

def calcular_razao_variancia(serie, max_lag=24):
    """
    Calcula o Teste de Razão de Variância (Variance Ratio) para verificar
    a força da reversão à média na série temporal limpa.
    """
    lags = range(2, max_lag + 1)
    vrs = []
    
    # Retornos de 1 período (1 mês) e a sua variância base
    retornos_1 = np.diff(serie)
    var_1 = np.var(retornos_1, ddof=1)
    
    for q in lags:
        # Retornos sobrepostos de 'q' períodos
        retornos_q = serie[q:] - serie[:-q]
        var_q = np.var(retornos_q, ddof=1)
        
        # A Razão de Variância matemática
        vr = var_q / (q * var_1)
        vrs.append(vr)
        
    return list(lags), vrs

def extrair_sazonalidade(serie_flat, data_inicio):
    """
    Recebe uma série temporal contínua e retorna o vetor de sazonalidade 
    de 8760 horas e a série original limpa (sem sazonalidade). O formato de data_inicio é YYYY-MM-DD
    """
    df = pd.DataFrame({'Valor': serie_flat})
    
    # Remoção de Anos Bissextos
    # df = df[~((df['DataHora'].dt.month == 2) & (df['DataHora'].dt.day == 29))]
    
    medias_8760 = df.groupby([df.index.month, df.index.day, df.index.hour])['Valor'].mean()
    medias_8760 = medias_8760.rename_axis(['Mes', 'Dia', 'Hora'])
    media_global = df['Valor'].mean()
    
    I_Pm_8760 = (medias_8760 / media_global)
    
    df_mapeado = df.set_index([df.index.month, df.index.day, df.index.hour]).rename_axis(['Mes', 'Dia', 'Hora'])
    df_mapeado['I_Pm'] = medias_8760 / media_global
    # df_mapeado = df_mapeado.sort_values('DataHora')
    
    # serie_sem_sazonalidade = np.nan_to_num((df_mapeado['Valor'] / df_mapeado['I_Pm']))
    serie_sem_sazonalidade = (df_mapeado['Valor'] / df_mapeado['I_Pm']).fillna(0)

    return I_Pm_8760, serie_sem_sazonalidade

def extrair_saltos(serie, multiplicador_std=1.0):
    media = np.mean(serie)
    desvio = np.std(serie, ddof=1)
    limite = media + (multiplicador_std * desvio)
    
    serie_sem_saltos = np.clip(serie, a_min=None, a_max=limite)
    vetor_saltos = np.maximum(serie - limite, 0.0)
    
    return serie_sem_saltos, vetor_saltos, limite

def estimar_saltos(serie, serie_saltos):
    qtd_pontos = len(serie)
    aux_serie_saltos = serie_saltos[serie_saltos != 0]

    kappa = aux_serie_saltos.mean()
    gamma = np.std(aux_serie_saltos)
    phi = len(aux_serie_saltos) / qtd_pontos

    return kappa, gamma, phi

def calibrar_parametros_mrm_g(pld_historico_limpo, dt=1):
    """
    Estima os parâmetros do processo MRM-G usando Regressão Linear Simples.
    Adicionado o retorno dos resíduos para a correlação de Cholesky.
    """
    ln_P = np.log(pld_historico_limpo)
    Y = np.diff(ln_P)
    X = ln_P[:-1]
    
    slope, intercept, _, _, _ = linregress(X, Y)
    
    b = slope + 1
    a = intercept

    eta = -np.log(b) / dt
    
    residuos = Y - (intercept + slope * X)
    variancia_residuos = np.var(residuos, ddof=1)
    sigma_e = np.sqrt(variancia_residuos / dt)

    a_sobre_1_b = a / (1 - b)
    x_barra = np.exp(a_sobre_1_b + sigma_e ** 2 / (1 - b ** 2))
    sigma = sigma_e * np.sqrt(2 * np.log(b) / ((b**2 - 1) * dt))
    
    return eta, x_barra, sigma, residuos

def calibrar_parametros_mrm_a(historico_limpo, dt=1, dni=False):
    """
    Estima os parâmetros do processo MRM Aritmético usando Regressão Linear Simples.
    Adicionado o retorno dos resíduos para a correlação de Cholesky.
    """
    if dni:
        df = historico_limpo.copy()
        df = df[(df.index.get_level_values('Hora') > 5) & (df.index.get_level_values('Hora') < 19)]

        Y = np.diff(df)
        X = df[:-1]
        
        slope, intercept, _, _, _ = linregress(X, Y)
        
        a = intercept
        b = slope + 1
        
        residuos = Y - (intercept + slope * X)
        variancia_residuos = np.var(residuos, ddof=1)
        sigma_e = np.sqrt(variancia_residuos)
        
        eta = -np.log(b) / dt
        x_barra = a / (1 - b)
        termo_raiz = (-2 * np.log(b)) / ((1 - b**2) * dt)
        sigma = sigma_e * np.sqrt(termo_raiz)

    else:
        Y = np.diff(historico_limpo)
        X = historico_limpo[:-1]
        
        slope, intercept, _, _, _ = linregress(X, Y)
        
        a = intercept
        b = slope + 1
        
        residuos = Y - (intercept + slope * X)
        variancia_residuos = np.var(residuos, ddof=1)
        sigma_e = np.sqrt(variancia_residuos)
        
        eta = -np.log(b) / dt
        x_barra = a / (1 - b)
        termo_raiz = (-2 * np.log(b)) / ((1 - b**2) * dt)
        sigma = sigma_e * np.sqrt(termo_raiz)
    
    return eta, x_barra, sigma, residuos

# =====================================================================
# FUNÇÕES DE SIMULAÇÃO (Com injeção de Ruído eps_ruido)
# =====================================================================

def gerar_dni_sintetica(lat, long, alt, tz, df_historico_completo, anos_simulacao, sorteios, coluna_irradiancia='DNI', intervalo_confianca=0.95, rodadas=1):
    """
    Gera séries sintéticas de irradiância baseadas em um ano padrão de condições de céu.
    
    Parâmetros:
    - df_historico_completo (pd.DataFrame): Histórico contendo índice datetime e a coluna 'kc'.
    - anos_simulacao (list/array): Lista de anos para simular (ex: [1998, 1999, 2000]).
    - sorteios (np.ndarray): Matriz de ruídos eps, esperado shape (total_horas, n_cen)
    - coluna_irradiancia (str): Nome da coluna alvo (padrão: 'DNI').
    - intervalo_confianca (float): Intervalo do método estocástico (padrão: 0.95).
    - rodadas (int): Número de rodadas (padrão: 1).
    
    Retorna:
    - series_sinteticas (np.ndarray): Matriz final (Total de Horas, Cenários).
    """

    # Se a função clear_sky_index retornar um dataframe novo, faça a atribuição:
    # df_historico_completo = src.methods.clear_sky_index(...)
    src.methods.clear_sky_index(
        data=df_historico_completo,
        column='DNI',
        longitude=long,
        latitude=lat,
        altitude=alt,
        time_zone=tz
    )
    
    # Validação básica de segurança
    if 'kc' not in df_historico_completo.columns:
        raise ValueError("A coluna 'kc' (Clear-sky index) não foi encontrada no DataFrame histórico.")
    
    # ==============================================================================
    # 1. CÁLCULO DO ANO PADRÃO (365 DIAS) BASEADO EM TODOS OS ANOS
    # ==============================================================================
    # 1.1. Filtra apenas o período diurno (6h às 17h59)
    df_diurno = df_historico_completo.loc[(df_historico_completo.index.hour >= 6) & (df_historico_completo.index.hour < 18)]

    # 1.2. Calcula a mediana diária do kc na linha do tempo real
    kc_diario_historico = df_diurno['kc'].resample(rule='1d').median()

    # 1.3. Remove os dias 29 de fevereiro para garantir que os anos tenham 365 dias
    kc_diario_historico = kc_diario_historico[~((kc_diario_historico.index.month == 2) & (kc_diario_historico.index.day == 29))]

    # 1.4. Agrupa por Mês e Dia para encontrar a média definitiva de todos os anos
    kc_padrao_365 = kc_diario_historico.groupby([kc_diario_historico.index.month, kc_diario_historico.index.day]).mean()
    kc_padrao_365.index.names = ['Mes', 'Dia']

    # ==============================================================================
    # 2. CATEGORIZAÇÃO DAS CONDIÇÕES DE CÉU (SC)
    # ==============================================================================
    limites = [-float('inf'), 0.2, 0.4, 0.6, 0.67, float('inf')]
    categorias = ['sc1', 'sc2', 'sc3', 'sc4', 'sc5']

    # Categoriza os 365 dias do ano padrão nas condições de céu
    sc_padrao_365 = pd.cut(kc_padrao_365, bins=limites, labels=categorias)

    # ==============================================================================
    # 3. SIMULAÇÃO ESTOCÁSTICA VETORIZADA POR AGRUPAMENTO
    # ==============================================================================
    
    n_cen = len(sorteios[0])
    total_anos = len(anos_simulacao)
    
    # Alocação otimizada com float32
    series_sinteticas = np.zeros((total_anos * 365 * 24, n_cen), dtype=np.float32)

    # Transforma os ruídos de (total_horas, n_cen) para (total_dias, 24, n_cen)
    # eps_dni_diario = sorteios.reshape(-1, 24, n_cen)
    
    rng = np.random.default_rng()
    anos_hist_unicos = df_historico_completo.index.year.unique()

    # Cria uma matriz gigante sorteando os anos para TODOS os cenários e anos de uma vez
    # shape: (total_anos_simulados, n_cen)
    matriz_anos_sorteados = rng.choice(anos_hist_unicos, size=(total_anos, n_cen))
    tempo_inicio = time.time()

    # -> LOOP 1: Varre a linha do tempo (Ano a Ano da simulação)
    for ano_sim_idx, ano_simulado in enumerate(anos_simulacao):
        
        # Pega a "fatia" com o ano histórico sorteado para CADA UM dos cenários neste ano de simulação
        # Ex: [1998, 2005, 1998, 2010, ...]
        anos_hist_neste_ano = matriz_anos_sorteados[ano_sim_idx, :]
        
        # Identifica quais foram os anos históricos únicos sorteados por essa multidão (Máximo de ~27 loops)
        anos_unicos_sorteados = np.unique(anos_hist_neste_ano)
        
        # -> LOOP 2: Itera apenas pelos anos históricos que saíram no sorteio
        for ano_hist in anos_unicos_sorteados:
            
            # DESCUBRA A MÁGICA: Acha o índice exato de todos os cenários que sortearam esse mesmo ano!
            cenarios_idx = np.where(anos_hist_neste_ano == ano_hist)[0]
            runs_neste_grupo = len(cenarios_idx)
            
            # Filtra o dataframe histórico apenas UMA VEZ para atender a todos esses cenários
            df_historico_ano = df_historico_completo[df_historico_completo.index.year == ano_hist]
            
            # Zera os ponteiros do ano relativos aos 365 dias
            dia_relativo = 0
            hora_relativa = 0
            
            # Índices globais para rastrear onde estamos na matriz gigante (262800 linhas)
            dia_global_base = ano_sim_idx * 365
            hora_global_base = ano_sim_idx * 365 * 24
            
            # -> LOOP 3: Itera sobre os 365 dias do "Ano Padrão"
            for (mes, dia), sc_do_dia in sc_padrao_365.items():
                
                dia_global = dia_global_base + dia_relativo
                hora_global = hora_global_base + hora_relativa
                
                # Extrai os ruídos exatos das 24 horas apenas dos cenários pertencentes a este grupo!
                # O shape será (24, runs_neste_grupo) -> Encaixa perfeitamente na função sequential!
                ruido_do_dia = sorteios[hora_global : hora_global + 24, cenarios_idx]
                
                # Executa a geração estocástica vetorizada para TODOS esses cenários de uma vez
                resultado_diario = src.methods.sequential_otimizado(
                    data=df_historico_ano,
                    irradiance_column=coluna_irradiancia,
                    year=ano_hist,
                    month=mes,
                    sky_condition=sc_do_dia,
                    method='stochastic',
                    confidence_interval=intervalo_confianca,
                    runs=runs_neste_grupo,
                    samples=ruido_do_dia
                )
                
                # Atribui o resultado na matriz gigante, mapeando de volta para os cenários originais
                series_sinteticas[hora_global : hora_global + 24, cenarios_idx] = resultado_diario
                
                # Avança os "ponteiros"
                dia_relativo += 1
                hora_relativa += 24

    # ==============================================================================
    # 4. FINALIZAÇÃO E RETORNO
    # ==============================================================================
    tempo_fim = time.time()
    
    return series_sinteticas

def simular_mrm_g(var_inicial_limpa, var_inicial_real, x_barra, eta, sigma, ind_sazon_futuro, phi=0, kappa_barra=0, gamma=0,
                  passos_tempo=8760, cenarios=2000, dt=1, saltos=False, eps_ruido=None):
    
    # 1. Aloca com +1 (O passado no índice 0, o futuro do 1 em diante)
    P = np.zeros((passos_tempo + 1, cenarios))
    P_S = np.zeros((passos_tempo + 1, cenarios))
    
    # 2. Guarda o último dado do histórico
    P[0] = var_inicial_limpa
    P_S[0] = var_inicial_real
    
    if eps_ruido is None:
        eps_P = np.random.normal(0, 1, size=(passos_tempo, cenarios))
    else:
        eps_P = eps_ruido

    raiz_dt = np.sqrt(dt)

    if saltos:
        u = np.random.uniform(0, 1, size=(passos_tempo, cenarios))
        var_gamma = gamma ** 2
        mu_ln = np.log((kappa_barra ** 2) / np.sqrt(var_gamma + kappa_barra ** 2))
        sigma_ln = np.sqrt(np.log(1.0 + var_gamma / (kappa_barra ** 2)))
        eps_kappa = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=(passos_tempo, cenarios))

    # 3. O Loop usa os ruídos perfeitos e preenche o futuro [t+1]
    for t in range(0, passos_tempo):
        termo_drift = (eta * (np.log(x_barra) - np.log(P[t])) - (sigma ** 2) / 2) * dt
        termo_difusao = sigma * eps_P[t] * raiz_dt
        
        P[t+1] = P[t] * np.exp(termo_drift + termo_difusao)
        
        if saltos:
            salto = eps_kappa[t] * (u[t] < (phi * dt))
            P_S[t+1] = (P[t+1] * ind_sazon_futuro[t]) + salto
        else:
            P_S[t+1] = P[t+1] * ind_sazon_futuro[t]
    
    return P_S


def simular_mrm_a(var_inicial_limpa, var_inicial_real, x_barra, eta, sigma, ind_sazon_futuro,
                  passos_tempo=8760, cenarios=2000, dt=1, eps_ruido=None):
    
    # 1. Aloca com +1 (O passado no índice 0, o futuro do 1 em diante)
    X = np.zeros((passos_tempo + 1, cenarios))
    X_S = np.zeros((passos_tempo + 1, cenarios))
    
    # 2. Guarda o último dado do histórico
    X[0] = var_inicial_limpa
    X_S[0] = var_inicial_real
    
    if eps_ruido is None:
        eps_X = np.random.normal(0, 1, size=(passos_tempo, cenarios))
    else:
        eps_X = eps_ruido
        
    raiz_dt = np.sqrt(dt)
    
    # 3. O Loop usa os ruídos perfeitos e preenche o futuro [t+1]
    for t in range(0, passos_tempo):
        termo_drift = eta * (x_barra - X[t]) * dt
        termo_difusao = sigma * eps_X[t] * raiz_dt
        
        X[t+1] = X[t] + termo_drift + termo_difusao
        X_S[t+1] = X[t+1] * ind_sazon_futuro[t]
            
    # X_S = np.maximum(X_S, 0.0) # Trava de segurança para não gerar < 0
    return X_S

# =====================================================================
# FUNÇÃO DE VISUALIZAÇÃO
# =====================================================================

def plotar_figura(n_fig, var, eixo_x, eixo_y, titulo="", save=False, cenarios=False):
    plt.figure(n_fig, figsize=(15, 5))
    if cenarios:
        plt.plot(var, color='#800080', linewidth=0.5, alpha=0.05)
    else:
        plt.plot(var, color='#800080', linewidth=0.5)
    if titulo != "":
        plt.title(titulo, fontsize=14)
    plt.xlabel(eixo_x, fontsize=12)
    plt.ylabel(eixo_y, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(f'{titulo.replace(" ", "_")}.pdf', bbox_inches='tight')


def plotar_histograma(n_fig, var, n_bins, eixo_x, eixo_y, titulo="", save=False, cenarios=False):
    plt.figure(n_fig, figsize=(10, 6))
    plt.hist(var, bins=n_bins, color='#800080', edgecolor='#000000' ,linewidth=0.5, alpha=0.7)
    if titulo != "":
        plt.title(titulo, fontsize=14)
    plt.xlabel(eixo_x, fontsize=12)
    plt.ylabel(eixo_y, fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig(f'{titulo.replace(" ", "_")}.pdf', bbox_inches='tight')

def plotar_sazonalidade(n_fig):
    plt.figure(n_fig, figsize=(8, 6))

    plt.plot(i_temp, label='Temperature')
    plt.plot(i_dni, label='DNI')
    plt.plot(i_umid, label='DHI')
    plt.plot(i_press, label='Pressure')
    plt.plot(i_vento, label='Wind Speed')

    plt.legend()
    plt.xlabel('Hora', fontsize=12)
    plt.ylabel('Índice de Sazonalidade', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plotar_razao_variancia(n_fig, dict_series, max_lag, titulo):
    """
    Plota o teste de Razão de Variância. Se a curva cair abaixo de 1,
    há confirmação de reversão à média.
    """
    plt.figure(n_fig, figsize=(10, 5))
    
    # Cores fixas para manter a consistência entre os submercados
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (nome, serie), cor in zip(dict_series.items(), cores):
        lags, vrs = calcular_razao_variancia(serie, max_lag)
        plt.plot(lags, vrs, label=nome, color=cor, linewidth=2)
        
    plt.axhline(1, color='black', linestyle='--', linewidth=1.5, label='Passeio Aleatório (VR = 1)')
    
    plt.title(titulo, fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Razão de Variância', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plotar_backtesting(n_fig, historico_completo, n_treino, simulacoes, titulo, eixo_y, num_cenarios_plot=5):
    """
    Plota a validação do modelo: 80% treino, 20% teste real, cenários e média simulada.
    """
    plt.figure(n_fig, figsize=(12, 6))
    
    # 1. Histórico de Treino (80%)
    x_treino = range(n_treino)
    plt.plot(x_treino, historico_completo, color='black', label='Histórico', linewidth=1)
    # plt.plot(x_treino, historico_completo[:n_treino], color='black', label='Histórico', linewidth=1)
    
    # 2. Eixo X das previsões compartilha o ponto de união (n_treino - 1)
    x_teste = range(len(historico_completo) - 1, simulacoes.shape[0] + len(historico_completo) - 1)
    
    # 3. Cenários Simulados (limitado a num_cenarios_plot para visualização)
    qtd_total_cenarios = simulacoes.shape[1]
    cenarios_para_plotar = min(num_cenarios_plot, qtd_total_cenarios)
    
    # Aumentei um pouco o alpha para 0.3 para as poucas linhas ficarem visíveis
    plt.plot(x_teste, simulacoes[:, :cenarios_para_plotar], color='red', alpha=.3, linewidth=0.3)
    plt.plot([], [], color='red', alpha=0.5, label=f'Séries sintéticas') # Linha fantasma para legenda
    
    # 4. Média das Simulações (Calculada sobre TODOS os cenários gerados)
    media_simulacoes = np.mean(simulacoes, axis=1)
    plt.plot(x_teste, media_simulacoes, color='#800080', label=f'Média dos cenários', linewidth=1.5, linestyle='--')

    # 5. Histórico Real de Teste, também partindo do ponto de união para costura perfeita
    # plt.plot(x_teste, historico_completo[n_treino - 1:], color='black', label='Histórico (validação)', linewidth=1.5)
    
    # Divisória visual do tempo
    plt.axvline(x=n_treino - 1, color='gray', linestyle=':', linewidth=1.5)
    
    plt.title(titulo, fontsize=14)
    plt.xlabel('Meses (Índice)', fontsize=12)
    plt.ylabel(eixo_y, fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xlim((235400, 237000))


# Se o seu modelo incluir análise financeira, você também precisará importar o módulo financeiro
# import PySAM.Singleowner as Singleowner

if __name__ == '__main__':

    # ---------------------------------------------------------
    # 1. LEITURA DE DADOS
    # ---------------------------------------------------------
    print(f"Lendo dados meteorológicos...")
    
    path_ger = 'historico_variaveis/nsrdb_novo' 
    lista_df_anos_ger = []
    var_list = ['Temperature','DNI','Relative Humidity','Pressure','Wind Speed']

    for ano in range(1998, 2025):
        ano = 2020
        arquivo = f'{path_ger}/1945275_-13.27_-43.42_{ano}.csv'
        df_ano = pd.read_csv(arquivo, skiprows=2, usecols=var_list, encoding='latin1')
        print('ano / horas:', ano, len(df_ano))
        lista_df_anos_ger.append(df_ano)
        break
        
    df_total = pd.concat(lista_df_anos_ger, ignore_index=True)
    
    serie_temp  = df_total['Temperature']
    serie_dni  = df_total['DNI']
    serie_umid = df_total['Relative Humidity']
    serie_press = df_total['Pressure']
    serie_vento = df_total['Wind Speed']


    # ------
    # 1.1 TESTES ESTATÍSTICOS - DICKEY-FULLER AUMENTADO E RAZÃO DA VARIÂNCIA
    # ------

    # aplicar_testes_adf(serie_temp, "Temperatura")
    # aplicar_testes_adf(serie_dni, "DNI")
    # aplicar_testes_adf(serie_umid, "Umidade")
    # aplicar_testes_adf(serie_press, "Pressão")
    # aplicar_testes_adf(serie_vento, "Velocidade do Vento")


    # dict_series = {
    #     'Temperatura': serie_temp.values,
    #     'DNI': serie_dni.values,
    #     'Umidade': serie_umid.values,
    #     'Pressão': serie_press.values,
    #     'Velocidade do Vento': serie_vento.values
    #     }
    

    # plotar_razao_variancia(1, dict_series, max_lag=250, titulo='Variáveis Climáticas')

    # plt.show()

    # plotar_histograma(1, serie_temp.values, 20, 'Temperatura [°C]', 'Frequência')
    # plotar_histograma(2, serie_dni.values, 20, 'DNI [W/m²]', 'Frequência')
    # plotar_histograma(3, serie_umid.values, 20, 'Umidade Relativa do Ar [%]', 'Frequência')
    # plotar_histograma(4, serie_press.values, 20, 'Pressão [mbar]', 'Frequência')
    # plotar_histograma(5, serie_vento.values, 20, 'Velocidade do Vento [m/s]', 'Frequência')

    # plt.show()

    # ---------------------------------------------------------
    # 2. EXTRAÇÃO DA SAZONALIDADE
    # ---------------------------------------------------------
    print("\nExtraindo sazonalidades e saltos...")

    i_temp, temp_s_sazon = extrair_sazonalidade(df_total['Temperature'], '1998-01-01')
    i_dni, dni_s_sazon = extrair_sazonalidade(df_total['DNI'], '1998-01-01')
    i_umid, umid_s_sazon = extrair_sazonalidade(df_total['Relative Humidity'], '1998-01-01')
    i_press, press_s_sazon = extrair_sazonalidade(df_total['Pressure'], '1998-01-01')
    i_vento, vento_s_sazon = extrair_sazonalidade(df_total['Wind Speed'], '1998-01-01')

    # ---------------------------------------------------------
    # 3. DETERMINAÇÃO DOS PARÂMETROS DA SIMULAÇÃO (MRM-G e MRM-A)
    # ---------------------------------------------------------

    # Ainda é necessário verificar a adequação das variáveis aos modelos.
    eta_dni_a, x_barra_dni_a, sigma_dni_a, res_dni_a = calibrar_parametros_mrm_a(dni_s_sazon, dt=1)
    eta_dni_g, x_barra_dni_g, sigma_dni_g, res_dni_g = calibrar_parametros_mrm_g(dni_s_sazon, dt=1)


    eta_temp, x_barra_temp, sigma_temp, res_temp = calibrar_parametros_mrm_a(temp_s_sazon, dt=1)
    eta_umid, x_barra_umid, sigma_umid, res_umid = calibrar_parametros_mrm_a(umid_s_sazon, dt=1)
    eta_press, x_barra_press, sigma_press, res_press = calibrar_parametros_mrm_a(press_s_sazon, dt=1)
    eta_vento, x_barra_vento, sigma_vento, res_vento = calibrar_parametros_mrm_g(vento_s_sazon, dt=1)

    # ---------------------------------------------------------
    # 4. DECOMPOSIÇÃO DE CHOLESKY
    # ---------------------------------------------------------

    df_corr = df_total[df_total['DNI'] != 0]
    m_correlacao = df_corr.corr()
    Z = np.linalg.cholesky(m_correlacao)

    print(m_correlacao)

    # -----
    # 5. SORTEIOS CORRELACIONADOS
    # -----

    
    anos_de_simulacao = 3  # Digite aqui quantos anos você quer prever!
    passos = 20 * 24    # 8760 * anos_de_simulacao
    qtd_cenarios = 10       # Quantidade de futuros paralelos
    n_variaveis = 5

    sorteios_independentes = np.random.normal(0, 1, size=(passos, qtd_cenarios, n_variaveis))
    sorteios_correlacionados = sorteios_independentes @ Z.T

    eps_temp = sorteios_correlacionados[:, :, 0]
    eps_dni = sorteios_correlacionados[:, :, 1]
    eps_umid = sorteios_correlacionados[:, :, 2]
    eps_press = sorteios_correlacionados[:, :, 3]
    eps_vento = sorteios_correlacionados[:, :, 4]
    
    # "Estica" os vetores de sazonalidade para cobrirem todos os anos

    i_temp_longo  = np.tile(i_temp, anos_de_simulacao)
    i_dni_longo   = np.tile(i_dni, anos_de_simulacao)
    i_umid_longo  = np.tile(i_umid, anos_de_simulacao)
    i_press_longo = np.tile(i_press, anos_de_simulacao)
    i_vento_longo = np.tile(i_vento, anos_de_simulacao)

    # -----
    # 6. GERAÇÃO DOS CENÁRIOS ALEATÓRIOS
    # -----

    # temp_sint = simular_mrm_a(
    #     var_inicial_limpa=temp_s_sazon[-1],
    #     var_inicial_real=serie_temp.iloc[-1],
    #     x_barra=x_barra_temp, eta=eta_temp, sigma=sigma_temp, ind_sazon_futuro=i_temp_longo,
    #     passos_tempo=passos, cenarios=qtd_cenarios, dt=1, eps_ruido=eps_temp)  

    # dni_sint_a = simular_mrm_a(
    #     var_inicial_limpa=dni_s_sazon[-1],
    #     var_inicial_real=serie_dni.iloc[-1],
    #     x_barra=x_barra_dni_a, eta=eta_dni_a, sigma=sigma_dni_a, ind_sazon_futuro=i_dni,
    #     passos_tempo=passos, cenarios=qtd_cenarios, dt=1, eps_ruido=eps_dni)  
    
    # dni_sint_g = simular_mrm_g(
    #     var_inicial_limpa=dni_s_sazon[-1],
    #     var_inicial_real=serie_dni.iloc[-1],
    #     x_barra=x_barra_dni_g, eta=eta_dni_g, sigma=sigma_dni_g, ind_sazon_futuro=i_dni,
    #     passos_tempo=passos, cenarios=qtd_cenarios, dt=1, eps_ruido=eps_dni)

    # umid_sint = simular_mrm_a(
    #     var_inicial_limpa=umid_s_sazon[-1],
    #     var_inicial_real=serie_umid.iloc[-1],
    #     x_barra=x_barra_umid, eta=eta_umid, sigma=sigma_umid, ind_sazon_futuro=i_umid,
    #     passos_tempo=passos, cenarios=qtd_cenarios, dt=1, eps_ruido=eps_umid)  
    
    # press_sint = simular_mrm_a(
    #     var_inicial_limpa=press_s_sazon[-1],
    #     var_inicial_real=serie_press.iloc[-1],
    #     x_barra=x_barra_press, eta=eta_press, sigma=sigma_press, ind_sazon_futuro=i_press,
    #     passos_tempo=passos, cenarios=qtd_cenarios, dt=1, eps_ruido=eps_press)  
    
    # vento_sint = simular_mrm_g(
    #     var_inicial_limpa=vento_s_sazon[-1],
    #     var_inicial_real=serie_vento.iloc[-1],
    #     x_barra=x_barra_vento, eta=eta_vento, sigma=sigma_vento, ind_sazon_futuro=i_vento,
    #     passos_tempo=passos, cenarios=qtd_cenarios, dt=1, eps_ruido=eps_vento)  
    
    # print(np.mean(dni_sint_a), np.mean(dni_sint_g))
   
    # plotar_backtesting(1, serie_dni.values, len(serie_dni), dni_sint_a, "DNI - MRM-A", "irradiância [W/m²]", num_cenarios_plot=10)
    # plotar_backtesting(2, serie_dni.values, len(serie_dni), dni_sint_g, "DNI - MRM-G", "irradiância [W/m²]", num_cenarios_plot=10)
    # plotar_backtesting(1, serie_temp.values, len(serie_temp), temp_sint, "Validação do MRM-A para Temperatura", "Temperatura [°C]", num_cenarios_plot=10)
    # plotar_backtesting(2, serie_vento.values, len(serie_vento), vento_sint, 'Velocidade do Vento - MRM-G', 'velocidade [m/s]', 10)
    # plotar_backtesting(3, serie_umid.values, len(serie_umid), umid_sint, 'Umidade Relativa - MRM-A', 'umidade [%]', 10)
    # plotar_backtesting(4, serie_press.values, len(serie_press), press_sint, 'Pressão Atmosférica - MRM-A', 'pressão [mbar]', 10)
    

    # plotar_backtesting(1, df_final['PLD NE'].values, n_treino, pld_simulado, 
    #                    'Backtesting PLD NE - Modelo MRM-G', 'R$/MWh', 6)
    # plt.show()