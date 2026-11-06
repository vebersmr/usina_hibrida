import sys, os
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
from campo_solar_sam import dimensionar_campo_tubulacoes_sam, simular_campo_solar_sam_8760
from geracao_series_sinteticas import gerar_series_climaticas

pasta_clima = r'historico_variaveis\nsrdb_macau'
clima = gerar_series_climaticas(path_pasta=pasta_clima, prefixo_arquivo='2008167_-5.11_-36.62', ano_deterministico=2015)
cenario_0 = clima.obter_cenario(0)

cfg = dimensionar_campo_tubulacoes_sam(n_loops_field=91, p_ref_mw=55.5, eta_ref=0.356)
res = simular_campo_solar_sam_8760(cfg, cenario_0, latitude=-5.11, longitude=-36.62, tz=-3)

for h in [54, 55, 56]:
    print(f"Hour {h}: q_inc={res['q_rec_inc'][h]:.2f}, q_abs={res['q_rec_abs'][h]:.2f}, q_loss={res['q_rec_loss'][h]:.2f}, q_htf={res['q_htf_out'][h]:.2f}, def={res['scas_def'][h]:.4f}")
