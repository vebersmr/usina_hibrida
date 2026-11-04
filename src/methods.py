import numpy as np
import pandas as pd
import pvlib
import scipy


# =============================================================================
# Stochastic
# =============================================================================
def stochastic(dictionary:dict, year:int, month:int, sky_condition:str, runs:int) -> dict:
    '''
    '''
    # Catching exception
    if sky_condition not in ['sc1', 'sc2', 'sc3', 'sc4', 'sc5']:
        raise ValueError(f"An invalid sky condition ({sky_condition}) was selected. Select one of ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].")

    # Dictionary
    synthetic_irradiance = {}

    # Constants
    MONTHS = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
              '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}

    #Calling the data from irrad_analysis dictionary previously created
    data = dictionary[sky_condition]

    if data.empty == True:
        print(f'There is no information related to the date {MONTHS[str(month)]}-{year} {sky_condition}.')

    else:
        # DataFrame to store all the synthetic irradiance before store at dictionary
        df = pd.DataFrame()

        for i in range(runs):
            # Generation of synthetic irradiance by a gaussian (normal) distribution
            synt = np.random.normal(loc=data['mean'], scale=data['std'])
            '''
            if synt < data['min'].values:
                synt = data['min'].values

            if synt > data['max'].values:
                synt = data['max'].values
            '''
            synt[synt < data['min']] = data['min'][synt < data['min']].values
            synt[synt > data['max']] = data['max'][synt > data['max']].values

            # Synthetic data storage in dataframe
            df[f'synt{i+1}'] = synt

            if i == runs-1:
                # Change numerical index to datetime index
                df.set_index(data.index, inplace=True)
                df.index.set_names(['hour', 'minute'], inplace=True)

                # Save the synthetic data in irradsyntm1_dict
                synthetic_irradiance[f'{MONTHS[str(month)]}{year}-{sky_condition}'] = df

    return synthetic_irradiance

# =============================================================================
# Bootstrap
# =============================================================================
def bootstrap(dictionary:dict, year:int, month:int, sky_condition:str, resolution:int, runs:int) -> dict:
    '''
    '''
    # Catching exception
    if sky_condition not in ['sc1', 'sc2', 'sc3', 'sc4', 'sc5']:
        raise ValueError(f"An invalid sky condition ({sky_condition}) was selected. Select one of ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].")

    # Dictionary
    synthetic_irradiance = {}

    # Constants
    MONTHS = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
              '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}

    MULTIINDEX = [(i,j) for i in range(0, 24) for j in range(0, 60, resolution)]

    #Calling the data from irrad_analysis dictionary previously created
    data = dictionary[sky_condition]

    if data.empty == True:
        print(f'There is no information related to the date {MONTHS[str(month)]}-{year} {sky_condition}.')

    else:
        # DataFrame to store all the synthetic irradiance before store at dictionary
        df = pd.DataFrame()

        aux_data = data.to_numpy()

        synt = aux_data[np.random.choice(aux_data.shape[0], runs, replace=True)]

        for i in range(runs):
            # Synthetic data storage in dataframe
            df[f'synt{i+1}'] = synt[i]

            if i == runs-1:
                # Change numerical index to datetime index
                df.index = pd.MultiIndex.from_tuples(MULTIINDEX, names=('hour', 'minute'))

                # Save the synthetic data in irradsyntm1_dict
                synthetic_irradiance[f'{MONTHS[str(month)]}{year}-{sky_condition}'] = df

    return synthetic_irradiance

# =============================================================================
# Clear-sky index (kc)
# =============================================================================
def clear_sky_index(data:pd.DataFrame, column:str, longitude:float, latitude:float, altitude:float, time_zone:str) -> pd.DataFrame:
    '''
    '''
    # Location
    location = pvlib.location.Location(latitude, longitude, time_zone, altitude)

    # Clear-sky irradiance (Hcs)
    RESOLUTION = int(pd.Series(data.index.values).diff().median().total_seconds()/60)

    times = pd.date_range(start=data.index[0],end=data.index[-1], freq=f'{RESOLUTION}min', tz=time_zone)

    times_no_leap = times[~((times.month == 2) & (times.day == 29))]

    # Clear-sky irradiance (Hcs)
    hcs = location.get_clearsky(times=times_no_leap,
                                model='ineichen')

    # Append clear-sky irradiance to main dataframe
    data['ics_wm2'] = hcs['ghi'].values

    # Clear-sky index (kc) to main dataframe
    data['kc'] = data[column].values / data['ics_wm2'].values

    # NaN
    data['kc'] = data['kc'].fillna(1)

    # Replace kc > 1
    data.loc[data['kc'] > 1, 'kc'] = 1

    return data

# =============================================================================
# Sequential
# =============================================================================
def sequential(data:pd.DataFrame, irradiance_column:str, year:int, month:int, sky_condition:str,
               method:str, confidence_interval:float, runs:int, samples:np.ndarray) -> pd.DataFrame:
    '''
    '''
    # Catching exception
    if sky_condition not in ['sc1', 'sc2', 'sc3', 'sc4', 'sc5']:
        raise ValueError(f"An invalid sky condition ({sky_condition}) was selected. Select one of ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].")

    if method not in ['stochastic', 'bootstrap']:
        raise ValueError(f"An invalid method ({method}) for synthetic solar irradiance generation was selected. Select one of ['stochastic', 'bootstrap'].")

    # Constants
    RESOLUTION = int(pd.Series(data.index.values).diff().median().total_seconds()/60)
    TIMES = [f'{i}:0{j}' if j < 10 else f'{i}:{j}' for i in range(0, 24) for j in range(0, 60, RESOLUTION)]
    MULTIINDEX = [(i,j) for i in range(0, 24) for j in range(0, 60, RESOLUTION)]

    # DataFrame filtered by date and between 6:00 to 18:00h range
    data = data.loc[data.index.month == month] # O original utilizava dados de apenas um ano. Removi essa filtragem pois estou considerando o índice de céu claro incluindo todos os anos
    # data = data.loc[(data.index.year == year) & (data.index.month == month)] 

    # Median value of clear-sky index (kc)
    median_kc = data['kc'].loc[(data.index.hour >= 6) & (data.index.hour < 18)].resample(rule='1d').median()

    # Categorization according to clear-sky index (kc) value
    if sky_condition == 'sc1':
        days = np.array(median_kc.loc[median_kc <= 0.2].index.day)

    elif sky_condition == 'sc2':
        days = np.array(median_kc.loc[(0.2 < median_kc) & (median_kc <= 0.4)].index.day)

    elif sky_condition == 'sc3':
        days = np.array(median_kc.loc[(0.4 < median_kc) & (median_kc <= 0.6)].index.day)

    elif sky_condition == 'sc4':
        days = np.array(median_kc.loc[(0.6 < median_kc) & (median_kc <= 0.67)].index.day)

    elif sky_condition == 'sc5':
        days = np.array(median_kc.loc[median_kc > 0.67].index.day)

    else:
        raise ValueError(f'An invalid sky condition {sky_condition} was requested.')

    if len(days) == 0:
        days = np.array(median_kc.loc[median_kc > 0.67].index.day)

        
        # df = None

    aux_data = data[irradiance_column].loc[data.index.day.isin(days)]

    temp_aux_data = pd.DataFrame(aux_data.values.reshape(len(aux_data.index.day.unique()), len(TIMES)), index=list(aux_data.index.day.unique()), columns=TIMES)

    if temp_aux_data.empty != True:
        ALPHA = 1 - confidence_interval

        z = scipy.stats.norm.ppf(confidence_interval+(ALPHA/2)) # Gaussian

        # DataFrame to store all the synthetic irradiance before store at dictionary
        df = pd.DataFrame()

        for r in range(runs):
            synt = []

            for i, j in enumerate(TIMES):
                if i == 0:
                    # STEP 1. Select column from pd.DataFrame
                    temp = temp_aux_data[j]

                    # STEP 2. Generate a stochastic synthetic data
                    if method == 'stochastic':
                        # synt.append(np.random.normal(loc=np.mean(temp), scale=np.std(temp)))
                        synt.append((samples[i] * np.std(temp)) + np.mean(temp))
                        
                    else:
                        synt.append(np.random.choice(temp.values))

                else:
                    # STEP 1. Select column from pd.DataFrame
                    temp = temp_aux_data[[TIMES[i-1], TIMES[i]]]

                    # STEP 2. Confidence interval
                    N = len(temp)

                    # z = scipy.stats.norm.ppf(confidence_interval+(ALPHA/2)) # Gaussian
                    if len(temp < 2):
                        std = 0
                    else:
                        std = np.std(temp[TIMES[i-1]], ddof=1) # Sample

                    upper_bound = synt[i-1] + (z * std / np.sqrt(N))
                    lower_bound = synt[i-1] - (z * std / np.sqrt(N))

                    # STEP 3. Interval bounds
                    if upper_bound >= np.max(temp[TIMES[i-1]].values):
                        upper_bound = np.max(temp[TIMES[i-1]].values)

                    if lower_bound <= np.min(temp[TIMES[i-1]].values):
                        lower_bound = np.min(temp[TIMES[i-1]].values)

                    # STEP 3. Filter pd.DataFrame by confidence interval
                    temp = temp.loc[(lower_bound <= temp[TIMES[i-1]]) & (temp[TIMES[i-1]] <= upper_bound)]

                    # STEP 4. Generate a stochastic synthetic data
                    if len(temp[TIMES[i]]) == 0:
                        if i == 18:
                            synt.append(0)
                        else: 
                            synt.append(synt[i-1])

                    else:
                        # Stochastic
                        if method == 'stochastic':
                            s = np.random.normal(loc=np.mean(temp[TIMES[i]]), scale=np.std(temp[TIMES[i]]))
                            # s = (samples[i] * np.std(temp[TIMES[i]])) + np.mean(temp[TIMES[i]])

                            if s >= np.max(temp[TIMES[i]].values):
                                s = np.max(temp[TIMES[i]].values)

                            if s <= np.min(temp[TIMES[i]].values):
                                s = np.min(temp[TIMES[i]].values)

                            synt.append(s)

                        # Bootstrap
                        else:
                            synt.append(np.random.choice(temp[TIMES[i]].values))

            # Synthetic data storage in dataframe
            df[f'synt{r+1}'] = synt

            if r == runs-1:
                # Change numerical index to datetime index
                df.index = pd.MultiIndex.from_tuples(MULTIINDEX, names=('hour', 'minute'))

    
    teste = df['synt1'].sum()

    if teste == 0:
        print(f'Warning: The synthetic irradiance generated for {year}-{month} {sky_condition} is zero. Check the input data and parameters.')

    return df



def sequential_otimizado(data: pd.DataFrame, irradiance_column: str, year: int, month: int, sky_condition: str,
                         method: str, confidence_interval: float, runs: int, samples: np.ndarray) -> np.ndarray:
    
    if method != 'stochastic':
        raise ValueError("Esta versão otimizada suporta apenas o método 'stochastic'.")

    RESOLUTION = int((data.index[1] - data.index[0]).total_seconds() / 60)
    TIMES = [f'{i}:0{j}' if j < 10 else f'{i}:{j}' for i in range(0, 24) for j in range(0, 60, RESOLUTION)]

    # Filtra dados do mês (agora contém o mês de TODOS os anos do histórico)
    data_month = data.loc[data.index.month == month] 

    # Extrai apenas a data completa (YYYY-MM-DD) para não misturar dias de anos diferentes
    datas_hora = data_month.index.normalize() 

    mask_hours = (data_month.index.hour >= 6) & (data_month.index.hour < 18)
    median_kc = data_month['kc'].loc[mask_hours].resample(rule='1d').median().dropna()

    # Lista de filtros em ordem crescente para a busca em cascata
    filtros_ordem_crescente = [
        median_kc <= 0.2,                           # sc1
        (0.2 < median_kc) & (median_kc <= 0.4),     # sc2
        (0.4 < median_kc) & (median_kc <= 0.6),     # sc3
        (0.6 < median_kc) & (median_kc <= 0.67),    # sc4
        median_kc > 0.67                            # sc5
    ]

    # Mapeia a string 'scX' para o índice da lista (0 a 4)
    idx_condicao = int(sky_condition[-1]) - 1
    
    # Busca as datas que satisfazem a condição sorteada
    datas_validas = median_kc.loc[filtros_ordem_crescente[idx_condicao]].index

    # Nova lógica: Se não achou nenhum dia, testa as condições em ordem crescente
    if len(datas_validas) == 0:
        for filtro in filtros_ordem_crescente:
            datas_validas = median_kc.loc[filtro].index
            if len(datas_validas) > 0:
                break

    # Filtra os dados horários correspondentes às datas válidas
    mask_datas = datas_hora.isin(datas_validas)
    aux_data = data_month[irradiance_column].loc[mask_datas]
    
    num_days = len(datas_validas)

    if num_days == 0:
        return np.zeros((len(TIMES), runs)) # Retorno de segurança

    matriz_aux = aux_data.values.reshape(num_days, len(TIMES))
    
    ALPHA = 1 - confidence_interval
    z = scipy.stats.norm.ppf(confidence_interval + (ALPHA / 2)) 
    
    synt = np.zeros((len(TIMES), runs))
    
    for i in range(len(TIMES)):
        col_atual = matriz_aux[:, i] 
        
        if i == 0:
            synt[i] = (samples[i] * np.std(col_atual)) + np.mean(col_atual)
        else:
            col_anterior = matriz_aux[:, i-1]
            
            std = 0 if num_days < 2 else np.std(col_anterior, ddof=1)
            N = num_days
            intervalo = 0 if N == 0 else (z * std / np.sqrt(N))
            
            upper_bound = synt[i-1] + intervalo
            lower_bound = synt[i-1] - intervalo
            
            max_ant = np.max(col_anterior)
            min_ant = np.min(col_anterior)
            upper_bound = np.minimum(upper_bound, max_ant)
            lower_bound = np.maximum(lower_bound, min_ant)
            
            mask = (col_anterior[None, :] >= lower_bound[:, None]) & (col_anterior[None, :] <= upper_bound[:, None])
            col_atual_2d = np.broadcast_to(col_atual, (runs, len(col_atual)))
            col_mascarada = np.ma.masked_where(~mask, col_atual_2d)
            
            medias = col_mascarada.mean(axis=1).filled(0) 
            stds = col_mascarada.std(axis=1).filled(0)
            contagens = col_mascarada.count(axis=1)
            
            s = (samples[i] * stds) + medias
            
            max_atual = np.max(col_atual)
            min_atual = np.min(col_atual)
            s = np.clip(s, a_min=min_atual, a_max=max_atual)
            
            fallback = 0 if i == 18 else synt[i-1]
            synt[i] = np.where(contagens == 0, fallback, s)

    return synt


def sequential_otimizado_antigo(data: pd.DataFrame, irradiance_column: str, year: int, month: int, sky_condition: str,
                         method: str, confidence_interval: float, runs: int, samples: np.ndarray) -> pd.DataFrame:
    '''
    Versão 100% vetorizada: Processa todos os 'runs' (cenários) simultaneamente
    usando Broadcasting e Masked Arrays do NumPy, sem loops para os cenários!
    '''
    if method != 'stochastic':
        raise ValueError("Esta versão otimizada suporta apenas o método 'stochastic'.")

    # Constantes
    RESOLUTION = int((data.index[1] - data.index[0]).total_seconds() / 60)
    TIMES = [f'{i}:0{j}' if j < 10 else f'{i}:{j}' for i in range(0, 24) for j in range(0, 60, RESOLUTION)]
    MULTIINDEX = [(i,j) for i in range(0, 24) for j in range(0, 60, RESOLUTION)]

    # Filtra dados do mês
    data_month = data.loc[data.index.month == month] 

    # Valor mediano do clear-sky index (kc)
    mask_hours = (data_month.index.hour >= 6) & (data_month.index.hour < 18)
    median_kc = data_month['kc'].loc[mask_hours].resample(rule='1d').median()

    # Categorização
    if sky_condition == 'sc1': days = np.array(median_kc.loc[median_kc <= 0.2].index.day)
    elif sky_condition == 'sc2': days = np.array(median_kc.loc[(0.2 < median_kc) & (median_kc <= 0.4)].index.day)
    elif sky_condition == 'sc3': days = np.array(median_kc.loc[(0.4 < median_kc) & (median_kc <= 0.6)].index.day)
    elif sky_condition == 'sc4': days = np.array(median_kc.loc[(0.6 < median_kc) & (median_kc <= 0.67)].index.day)
    elif sky_condition == 'sc5': days = np.array(median_kc.loc[median_kc > 0.67].index.day)

    if len(days) == 0:
        days = np.array(median_kc.loc[median_kc > 0.67].index.day)

    # Coleta os dados filtrados
    aux_data = data_month[irradiance_column].loc[data_month.index.day.isin(days)]
    num_days = len(aux_data.index.day.unique())

    if num_days == 0:
        return pd.DataFrame()

    matriz_aux = aux_data.values.reshape(num_days, len(TIMES))
    
    ALPHA = 1 - confidence_interval
    z = scipy.stats.norm.ppf(confidence_interval + (ALPHA / 2)) 
    
    # =========================================================================
    # A MÁGICA DA VETORIZAÇÃO TOTAL (Loop de 'runs' FOI DELETADO)
    # synt é uma Matriz (24 horas, numero de cenarios). 
    # 'samples' é esperado com o formato (24 horas, numero de cenarios)
    # =========================================================================
    synt = np.zeros((len(TIMES), runs))
    
    for i in range(len(TIMES)):
        col_atual = matriz_aux[:, i] # shape (num_days,)
        
        if i == 0:
            # samples[i] é um vetor contendo um ruído diferente para cada cenário
            synt[i] = (samples[i] * np.std(col_atual)) + np.mean(col_atual)
        
        else:
            col_anterior = matriz_aux[:, i-1]
            
            # Tratamento de exceção se houver menos de 2 dias no histórico
            if num_days < 2:
                std = 0
            else:
                std = np.std(col_anterior, ddof=1)
                
            N = num_days
            
            # Constante do intervalo multiplicada pelo Z
            intervalo = 0 if N == 0 else (z * std / np.sqrt(N))
            
            # Limites agora são VETORES de tamanho 'runs'
            upper_bound = synt[i-1] + intervalo
            lower_bound = synt[i-1] - intervalo
            
            # Trava nos máximos e mínimos reais
            max_ant = np.max(col_anterior)
            min_ant = np.min(col_anterior)
            upper_bound = np.minimum(upper_bound, max_ant)
            lower_bound = np.maximum(lower_bound, min_ant)
            
            # -----------------------------------------------------------------
            # MASKED ARRAYS: O BroadCasting transforma col_anterior(1D) e 
            # os bounds(1D) numa matriz booleana 2D de tamanho (runs, num_days)
            # -----------------------------------------------------------------
            mask = (col_anterior[None, :] >= lower_bound[:, None]) & (col_anterior[None, :] <= upper_bound[:, None])
            
            # Expande a coluna atual para coincidir com o número de cenários
            col_atual_2d = np.broadcast_to(col_atual, (runs, len(col_atual)))
            
            # Esconde (mascara) os valores históricos que saíram fora da confiança de cada cenário (~mask)
            col_mascarada = np.ma.masked_where(~mask, col_atual_2d)
            
            # Calcula médias, desvios e conta os sobreviventes de UMA SÓ VEZ para todos os cenários!
            # .filled(0) previne erro de NaN onde o filtro foi muito rigoroso e zerou a contagem
            medias = col_mascarada.mean(axis=1).filled(0) 
            stds = col_mascarada.std(axis=1).filled(0)
            contagens = col_mascarada.count(axis=1)
            
            # Gera a simulação para todos os 'runs' simultaneamente!
            s = (samples[i] * stds) + medias
            # print(f'Hora {i}: médias={medias}, stds={stds}, contagens={contagens}, s={s}')
            
            # Trava final do resultado
            max_atual = np.max(col_atual)
            min_atual = np.min(col_atual)
            s = np.clip(s, a_min=min_atual, a_max=max_atual)
            
            # A regra do fallback agora é aplicada vetorialmente com np.where
            fallback = 0 if i == 18 else synt[i-1]
            synt[i] = np.where(contagens == 0, fallback, s)

    # Constrói o DataFrame a partir da matriz final em uma única operação

    # Verificação de erro noturno (aplicada apenas ao cenário 1 para checagem)
    # teste = df['synt1'].loc[18, 0]
    # if teste > 100:
    #     raise ValueError('Foi gerado um valor de irradiância maior que 100 para o período noturno.')

    return synt