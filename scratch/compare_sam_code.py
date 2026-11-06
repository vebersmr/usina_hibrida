import json
import PySAM.TroughPhysical as tp
import numpy as np

with open('c:/Users/victo/Desktop/codigos anti/sam/csp_ssc_trough_physical.json', 'r') as f:
    cfg = json.load(f)

model = tp.new()
for k, v in cfg.items():
    try:
        model.value(k, v)
    except Exception:
        pass

model.Weather.file_name = 'c:/Users/victo/Desktop/Veber/004_mestrado/simulacao/projeto_hibrido/historico_variaveis/nsrdb_macau/2008167_-5.11_-36.62_2015.csv'
model.execute()

sam_rec_inc = np.array(model.Outputs.q_dot_rec_inc)
sam_rec_abs = np.array(model.Outputs.q_dot_rec_abs)
sam_rec_loss = np.array(model.Outputs.q_dot_rec_thermal_loss)
sam_htf_out = np.array(model.Outputs.q_dot_htf_sf_out)
sam_scas_def = np.array(model.Outputs.SCAs_def)
sam_rec_su = np.array(model.Outputs.rec_time_in_startup)
sam_qinc_costh = np.array(model.Outputs.qinc_costh)
sam_q_inc_sf_tot = np.array(model.Outputs.q_inc_sf_tot)

code = np.load('cache_campo_91loops_lat-5.11_lon-36.62_tz-3_dni2499893_t248576_n8760.npz')
code_rec_inc = code['q_rec_inc']
code_rec_abs = code['q_rec_abs']
code_rec_loss = code['q_rec_loss']
code_htf_out = code['q_htf_out']
code_scas_def = code['scas_def']
code_q_inc_costh = code['q_inc_costh']
code_q_inc_sf = code['q_inc_sf']

print('--- Comparacao das Variaveis Acumuladas (GWht) ---')
print(f'q_inc_sf_tot:      SAM = {np.sum(sam_q_inc_sf_tot)/1e3:.3f} | Code = {np.sum(code_q_inc_sf)/1e3:.3f} | Diff = {(np.sum(code_q_inc_sf)-np.sum(sam_q_inc_sf_tot))/np.sum(sam_q_inc_sf_tot)*100:+.2f}%')
print(f'qinc_costh:        SAM = {np.sum(sam_qinc_costh)/1e3:.3f} | Code = {np.sum(code_q_inc_costh)/1e3:.3f} | Diff = {(np.sum(code_q_inc_costh)-np.sum(sam_qinc_costh))/np.sum(sam_qinc_costh)*100:+.2f}%')
print(f'q_dot_rec_inc:     SAM = {np.sum(sam_rec_inc)/1e3:.3f} | Code = {np.sum(code_rec_inc)/1e3:.3f} | Diff = {(np.sum(code_rec_inc)-np.sum(sam_rec_inc))/np.sum(sam_rec_inc)*100:+.2f}%')
print(f'q_dot_rec_abs:     SAM = {np.sum(sam_rec_abs)/1e3:.3f} | Code = {np.sum(code_rec_abs)/1e3:.3f} | Diff = {(np.sum(code_rec_abs)-np.sum(sam_rec_abs))/np.sum(sam_rec_abs)*100:+.2f}%')
print(f'q_rec_loss:        SAM = {np.sum(sam_rec_loss)/1e3:.3f} | Code = {np.sum(code_rec_loss)/1e3:.3f} | Diff = {(np.sum(code_rec_loss)-np.sum(sam_rec_loss))/np.sum(sam_rec_loss)*100:+.2f}%')
print(f'q_dot_htf_sf_out:  SAM = {np.sum(sam_htf_out)/1e3:.3f} | Code = {np.sum(code_htf_out)/1e3:.3f} | Diff = {(np.sum(code_htf_out)-np.sum(sam_htf_out))/np.sum(sam_htf_out)*100:+.2f}%')

# Diagnostico das diferencas em q_dot_rec_inc
diff_rec_inc = code_rec_inc - sam_rec_inc
mask_su = sam_rec_su > 0
mask_def = (sam_scas_def < 1.0) | (code_scas_def < 1.0)
mask_normal = (sam_rec_inc > 0) & (~mask_su) & (~mask_def)

print('\n--- Decomposicao das diferencas em q_dot_rec_inc (GWht) ---')
print(f'Total diferenca q_dot_rec_inc: {np.sum(diff_rec_inc)/1e3:+.3f} GWht')
print(f'  Em horas normais (foco 100%, sem startup, n={np.sum(mask_normal)}): {np.sum(diff_rec_inc[mask_normal])/1e3:+.3f} GWht')
print(f'  Em horas com startup (n={np.sum(mask_su)}):                          {np.sum(diff_rec_inc[mask_su])/1e3:+.3f} GWht')
print(f'  Em horas com defocus (n={np.sum(mask_def)}):                          {np.sum(diff_rec_inc[mask_def])/1e3:+.3f} GWht')

# Diagnostico das diferencas em q_dot_rec_abs
diff_rec_abs = code_rec_abs - sam_rec_abs
print('\n--- Decomposicao das diferencas em q_dot_rec_abs (GWht) ---')
print(f'Total diferenca q_dot_rec_abs: {np.sum(diff_rec_abs)/1e3:+.3f} GWht')
print(f'  Em horas normais (foco 100%, sem startup, n={np.sum(mask_normal)}): {np.sum(diff_rec_abs[mask_normal])/1e3:+.3f} GWht')
print(f'  Em horas com startup (n={np.sum(mask_su)}):                          {np.sum(diff_rec_abs[mask_su])/1e3:+.3f} GWht')
print(f'  Em horas com defocus (n={np.sum(mask_def)}):                          {np.sum(diff_rec_abs[mask_def])/1e3:+.3f} GWht')

# Salva arrays de comparacao em npz para analises subsequentes rapidas
np.savez_compressed('scratch/sam_reference_2015.npz',
    sam_rec_inc=sam_rec_inc,
    sam_rec_abs=sam_rec_abs,
    sam_rec_loss=sam_rec_loss,
    sam_htf_out=sam_htf_out,
    sam_scas_def=sam_scas_def,
    sam_rec_su=sam_rec_su,
    sam_qinc_costh=sam_qinc_costh,
    sam_q_inc_sf_tot=sam_q_inc_sf_tot,
)
print('\nscratch/sam_reference_2015.npz salvo com sucesso!')

