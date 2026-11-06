import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import math
from geracao_energia_v2 import (
    despachar_csp_tes,
    simular_receptor_csp,
    PC_EFF_DEFAULT,
    DEGRADATION_CSP_DEFAULT,
    CYCLE_THERMAL_POWER,
    TES,
    Tank,
    HX,
    eta_gross,
    cooling_tower,
    t_wetbulb,
    salt_cp_ave_k,
    vp1_cp_ave_k,
    T_HOT_OUT_DEFAULT,
    T_COLD_IN_DEFAULT,
    Q_PC_DES,
    PB_FIXED_PAR_SAM,
    PB_PUMP_COEF_SAM,
    TES_PUMP_COEF_SAM,
    SCA_DRIVES_ELEC_SAM,
)
from geracao_series_sinteticas import gerar_series_climaticas

df_sam = pd.read_excel('SAM_Macau_2017.xlsx')

clima = gerar_series_climaticas(
    path_pasta=r'historico_variaveis\nsrdb_macau',
    prefixo_arquivo='2008167_-5.11_-36.62',
    vida_util=1,
    qtd_cenarios=1,
    modo_deterministico=True,
    ano_deterministico=2017,
    incluir_dhi_albedo=True,
)
cenario_0 = clima.obter_cenario(0)
res_rec, t_in, p_loss = simular_receptor_csp(cenario_0)
diag_rec = res_rec.diagnostico
p_rec_mwt = np.array(res_rec)
scas_def_comp = diag_rec['scas_def']

# Função de teste de despacho com calibração ajustável
def despachar_calibrado(
    u_tank_val=0.40,
    ua_mult=1.0,
    bop_par=0.0,
    ct_k=0.0095,
    cutoff_threshold=0.0,
):
    cycle_power = CYCLE_THERMAL_POWER * 0.5
    fator_campo = 1.0
    n_loops_atual = 91
    tes_capacity = cycle_power * 5.0
    deg = DEGRADATION_CSP_DEFAULT
    pc_eff = PC_EFF_DEFAULT

    # TES com UA customizado
    class TankCalib(Tank):
        def __init__(self, v_tank, t_htr_k, q_htr_max_mw, v_ini, t_ini_k, t_des_k):
            super().__init__(v_tank, t_htr_k, q_htr_max_mw, v_ini, t_ini_k, t_des_k)
            self.u_tank = u_tank_val
            a_cs = v_tank / self.h_tank
            d = 2.0 * math.sqrt(a_cs / math.pi)
            self.ua = self.u_tank * (a_cs + math.pi * d * self.h_tank) * ua_mult

    class TwoTankTESCalib(TES):
        def __init__(self, q_tes_des_mwh, q_pc_des_w):
            t_hot_k = T_HOT_OUT_DEFAULT + 273.15
            t_cold_k = T_COLD_IN_DEFAULT + 273.15
            self.dt_hot = 5.0
            t_tes_hot_des = t_hot_k - self.dt_hot
            t_tes_cold_des = t_cold_k + self.dt_hot
            cp_salt = salt_cp_ave_k(t_tes_cold_des, t_tes_hot_des)
            m_avail = (q_tes_des_mwh * 1e6 * 3600.0) / (cp_salt * (t_tes_hot_des - t_tes_cold_des))
            h_tank, h_min = 12.0, 1.0
            f_active = (h_tank - h_min) / h_tank
            v_tank = (m_avail / 1850.0) / f_active
            v_cold_ini = v_tank
            v_hot_ini = v_tank * (h_min / h_tank)
            self.hot = TankCalib(v_tank, 365.0 + 273.15, 25.0, v_hot_ini, t_tes_hot_des, t_tes_hot_des)
            self.cold = TankCalib(v_tank, 250.0 + 273.15, 25.0, v_cold_ini, t_tes_cold_des, t_tes_cold_des)
            self.hx = HX(q_pc_des_w)
            cp_f = vp1_cp_ave_k(t_cold_k, t_hot_k)
            self.ratio = (t_tes_hot_des - t_tes_cold_des) * cp_salt / ((t_hot_k - t_cold_k) * cp_f)

    tes = TwoTankTESCalib(tes_capacity, cycle_power * 1e6)
    p_gross_nom_mwe = cycle_power * pc_eff
    t_hot_k = T_HOT_OUT_DEFAULT + 273.15
    t_cold_k = T_COLD_IN_DEFAULT + 273.15
    cp_f = vp1_cp_ave_k(t_cold_k, t_hot_k)
    dt_loop = t_hot_k - t_cold_k
    k_sf_pump = 2.98 / (1140.0 ** 3)

    q_pc_des = cycle_power
    q_pc_min = 0.20 * q_pc_des
    q_pc_max = 1.00 * q_pc_des
    e_su_req = 0.20 * q_pc_des
    su_time = 0.5
    q_su_max = e_su_req / su_time

    n_total = 8760
    p_in_mwt = p_rec_mwt
    temp_amb_celsius = cenario_0['temp'][:8760]
    tdew_celsius = temp_amb_celsius - 4.5
    q_fp_field_mwt = np.zeros(n_total)

    pout_bruto = np.zeros(n_total)
    pout_liquido = np.zeros(n_total)
    q_pc_out = np.zeros(n_total)
    q_ch_out = np.zeros(n_total)
    q_dc_out = np.zeros(n_total)
    q_defoc_out = np.zeros(n_total)
    w_par_total_out = np.zeros(n_total)

    pc_on = False
    e_su_acc = 0.0
    t_su_acc = 0.0

    for j in range(n_total):
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
        w_gross = eta_gross(q_pc_on, t_db, q_pc_des) * q_pc_on * frac_on if frac_on > 0.0 else 0.0

        # Parcelas Parasitas
        w_fixed = PB_FIXED_PAR_SAM * p_gross_nom_mwe
        f_load = min(w_gross / p_gross_nom_mwe, 1.2) if w_gross > 0.0 else 0.0
        w_bop = p_gross_nom_mwe * bop_par * (0.0 + 0.483 * f_load + 0.517 * (f_load ** 2)) if w_gross > 0.0 else 0.0
        w_track = SCA_DRIVES_ELEC_SAM * 1e-6 * 8.0 * n_loops_atual if q_f > 0 else 0.0

        m_sf = q_f_used * 1e3 / (cp_f * dt_loop) if q_f_used > 0 else 0.0
        w_sf = (k_sf_pump / max(fator_campo ** 2, 1e-6)) * (m_sf ** 3) if m_sf > 0 else 0.0

        m_pb = q_pc * 1e3 / (cp_f * dt_loop)
        w_pump_pb = PB_PUMP_COEF_SAM * m_pb / 1e3

        m_tank = r_tes["m_tank"]
        m_diff = abs(m_pb - m_sf)
        w_pump_tes = TES_PUMP_COEF_SAM * (m_tank + m_diff) / 1e3 if (m_tank > 0 or m_pb > 0) else 0.0

        q_rej_mw = max(q_pc_on * frac_on - w_gross, 0.0)
        w_ct = ct_k * q_rej_mw if w_gross > 0.0 else 0.0

        w_fp_field = float(q_fp_field_mwt[j])
        w_fp_tes = float(r_tes["q_htr"])

        w_par_total = w_fixed + w_bop + w_track + w_sf + w_pump_pb + w_pump_tes + w_ct + w_fp_field + w_fp_tes
        w_net = w_gross - w_par_total

        pout_bruto[j] = w_gross
        pout_liquido[j] = w_net
        q_pc_out[j] = q_pc
        q_ch_out[j] = q_ch
        q_dc_out[j] = q_dc
        w_par_total_out[j] = w_par_total

    return {
        'gross_mwh': pout_bruto.sum(),
        'net_mwh': pout_liquido.sum(),
        'par_mwh': w_par_total_out.sum(),
        'q_pc_mwt': q_pc_out.sum(),
        'q_ch_mwt': q_ch_out.sum(),
        'q_dc_mwt': q_dc_out.sum(),
        'e_96_gwh': pout_liquido.sum() * 0.96 / 1000.0,
    }

w_gr_sam = df_sam['PC electrical power output: gross | (MWe)'].sum()
w_net_sam = df_sam['System net electrical power | (MWe)'].sum()
w_par_sam = w_gr_sam - w_net_sam
e_96_sam = w_net_sam * 0.96 / 1000.0
q_pc_sam = df_sam['PC input energy | (MWt)'].sum()
q_ch_sam = df_sam['TES charge thermal power | (MWt)'].sum()
q_dc_sam = df_sam['TES discharge thermal power | (MWt)'].sum()

print('=== RESULTADOS DO SAM DE REFERÊNCIA ===')
print(f'PC In: {q_pc_sam/1000:.3f} | TES Ch: {q_ch_sam/1000:.3f} | TES Dc: {q_dc_sam/1000:.3f}')
print(f'Gross: {w_gr_sam/1000:.3f} GWh | Net: {w_net_sam/1000:.3f} GWh | Par: {w_par_sam/1000:.3f} GWh | Net 96%: {e_96_sam:.3f} GWh\n')

# Testes de calibração
casos = [
    ('1. Baseline Atual (BOP=0.0227, CT atual)', dict(bop_par=0.02273, ct_k=0.012, ua_mult=1.0)),
    ('2. Apenas Parasitas Calibrados (BOP=0, CT=0.0095)', dict(bop_par=0.0, ct_k=0.0095, ua_mult=1.0)),
    ('3. Parasitas + TES UA x 1.72 (Perdas TES calibradas)', dict(bop_par=0.0, ct_k=0.0095, ua_mult=1.72)),
    ('4. Parasitas + TES UA x 1.50', dict(bop_par=0.0, ct_k=0.0095, ua_mult=1.50)),
    ('5. Parasitas + TES UA x 1.72 + ajuste fino CT 0.0098', dict(bop_par=0.0, ct_k=0.0098, ua_mult=1.72)),
]

for nome, params in casos:
    r = despachar_calibrado(**params)
    dif_net = (r['net_mwh'] - w_net_sam) / w_net_sam * 100.0
    dif_gr = (r['gross_mwh'] - w_gr_sam) / w_gr_sam * 100.0
    dif_par = (r['par_mwh'] - w_par_sam) / w_par_sam * 100.0
    dif_pc = (r['q_pc_mwt'] - q_pc_sam) / q_pc_sam * 100.0
    dif_96 = (r['e_96_gwh'] - e_96_sam) / e_96_sam * 100.0
    q_pc_val = r['q_pc_mwt'] / 1000.0
    gr_val = r['gross_mwh'] / 1000.0
    par_val = r['par_mwh'] / 1000.0
    net_val = r['net_mwh'] / 1000.0
    e96_val = r['e_96_gwh']
    print(f'{nome}:')
    print(f'   PC In:   {q_pc_val:.3f} GWht (dif {dif_pc:+.2f}%)')
    print(f'   Gross:   {gr_val:.3f} GWh  (dif {dif_gr:+.2f}%)')
    print(f'   Par:     {par_val:.3f} GWh  (dif {dif_par:+.2f}%)')
    print(f'   Net:     {net_val:.3f} GWh  (dif {dif_net:+.2f}%)')
    print(f'   Net 96%: {e96_val:.3f} GWh   (dif {dif_96:+.2f}%)\n')