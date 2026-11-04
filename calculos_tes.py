import modelagem_csp
from math import pi

Hitec = modelagem_csp.Fluid(1)

P_gross = 55.5
eta = 0.356
TES_hours = 1

cycle_power = P_gross / eta

Q_tes_des = cycle_power * TES_hours # C na fórmula

T_cold_in = 293 # ºC 
T_hot_out = 391 # ºC 
dT_approach = 5 # °C

T_tes_hot = T_hot_out - dT_approach + 273.15
T_tes_cold = T_cold_in + dT_approach + 273.15

h_min = 1
h_tank_in = 12 
tank_pairs = 1
u_tank = 0.4

T_tes_ave = 0.5 * (T_tes_hot + T_tes_cold) - 273.15

rho_ave = Hitec.dens(T_tes_ave, 1)
cp_ave = Hitec.Cp_ave(T_tes_cold, T_tes_hot)

vol_one_temp_avail = Q_tes_des * 3600.0 / (rho_ave * cp_ave / 1000.0 * (T_tes_hot - T_tes_cold))

vol_one_temp_total = vol_one_temp_avail / (1.0 - h_min / h_tank_in)

A_cs = vol_one_temp_total / (h_tank_in * tank_pairs)


d_tank_out = pow(A_cs / pi, 0.5) * 2.0

UA_tanks_one_temp = u_tank * (A_cs + pi * d_tank_out * h_tank_in) * tank_pairs

T_amb_des = 15.0 + 273.15
q_dot_loss_cold = UA_tanks_one_temp * (T_tes_cold - T_amb_des) * 1.E-6
q_dot_loss_hot = UA_tanks_one_temp * (T_tes_hot - T_amb_des) * 1.E-6
q_dot_loss_des = q_dot_loss_cold + q_dot_loss_hot


fim = 1