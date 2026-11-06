import numpy as np

p = 'cache_campo_91loops_lat-5.11_lon-36.62_tz-3_dni2499893_t248576_n8760.npz'
data = np.load(p)

# Let's inspect the exact values in data
q_rec_inc = data['q_rec_inc']
q_rec_loss = data['q_rec_loss']
q_rec_abs = data['q_rec_abs']
q_htf_out = data['q_htf_out']
scas_def = data['scas_def']

print('Cache values sum:')
print(f'q_rec_inc:  {q_rec_inc.sum()/1e3:.3f} GWht')
print(f'q_rec_loss: {q_rec_loss.sum()/1e3:.3f} GWht')
print(f'q_rec_abs:  {q_rec_abs.sum()/1e3:.3f} GWht')
print(f'q_htf_out:  {q_htf_out.sum()/1e3:.3f} GWht')

