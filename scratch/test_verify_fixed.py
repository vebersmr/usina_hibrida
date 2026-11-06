import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from geracao_series_sinteticas import gerar_series_climaticas
from otimizacao_nsga_ii_v2 import ler_metadados_nsrdb, precalcular_perfis_unitarios, exportar_auditoria_csp

pasta_clima = r'historico_variaveis\nsrdb_macau'
prefixo_loc = '2008167_-5.11_-36.62'

metadados_loc = ler_metadados_nsrdb(path_pasta=pasta_clima, prefixo_arquivo=prefixo_loc)
clima = gerar_series_climaticas(
    path_pasta=pasta_clima,
    prefixo_arquivo=prefixo_loc,
    vida_util=1,
    qtd_cenarios=1,
    modo_deterministico=True,
    ano_deterministico=2015,
    incluir_dhi_albedo=True,
)
cenario_0 = clima.obter_cenario(0)

print("\n--- Testando precalcular_perfis_unitarios com p_csp_mva_base=100.0 ---")
cache_fisico = precalcular_perfis_unitarios(
    cenario_0,
    vida_util_sim=1,
    metadados_loc=metadados_loc,
    ano_base_solar=2015,
    p_csp_mva_base=100.0,
)

print('q_inc_sf (GWht):', np.sum(cache_fisico.q_inc_1loop_mwt * 181)/1e3)
print('q_inc_costh (GWht):', np.sum(cache_fisico.q_inc_costh_1loop_mwt * 181)/1e3)
print('q_rec_inc (GWht):', np.sum(cache_fisico.q_rec_inc_1loop_mwt * 181)/1e3)
print('q_rec_abs (GWht):', np.sum(cache_fisico.q_abs_1loop_mwt * 181)/1e3)
print('q_htf_out (GWht):', np.sum(cache_fisico.p_rec_1loop_mwt * 181)/1e3)

