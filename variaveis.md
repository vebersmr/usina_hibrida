# Variáveis do SAM para Modelagem de Receptor e Campo Solar (HTF)

Este arquivo mapeia os nomes das variáveis conforme definidos e utilizados internamente pelo SAM nos módulos C++ (`src_cpp/csp_solver_trough_collector_receiver.cpp` e `src_cpp/cmod_trough_physical.cpp`) e no arquivo JSON de parâmetros (`csp_ssc.json`). Elas representam parâmetros físicos e de controle que impactam os cálculos térmicos no receptor, nas tubulações e no balanço de energia do campo solar.

---

## 1. Parâmetros de Temperatura e Controle Térmico do Campo

| Nome no SAM (C++) | Descrição | Unidade | Valor Definido (JSON / Usuário) |
| :--- | :--- | :--- | :--- |
| `T_startup` | Temperatura mínima exigida na saída do campo antes de entregar fluido ao ciclo/TES | °C | 325,34 °C (598,49 K) |
| `T_shutdown` | Temperatura de corte abaixo da qual o campo entra em recirculação noturna | °C | 325,34 °C (598,49 K) |
| `T_loop_in_des` | Temperatura de projeto na entrada do loop solar | °C | 293,0 °C (566,15 K) |
| `T_loop_out` | Temperatura de projeto na saída do loop solar | °C | 391,0 °C (664,15 K) |
| `T_fp` | Temperatura de proteção anticongelamento (*heat trace / recirculation*) | °C | 150,0 °C (423,15 K) |

---

## 2. Parâmetros Ópticos e Geometria do Receptor (HCE)

| Nome no SAM (C++) | Descrição | Unidade | Valor Definido (JSON / Usuário) |
| :--- | :--- | :--- | :--- |
| `alpha_abs` | Absortância do tubo absorvedor | [-] | 0,963 |
| `Tau_envelope` | Transmitância do envelope de vidro | [-] | 0,964 |
| `alpha_env` | Absortância do envelope de vidro | [-] | 0,02 |
| `Dirt_HCE` | Fator de sujidade do receptor HCE | [-] | 0,98 |
| `Shadowing` | Fator de sombreamento estrutural do receptor (foles e suportes) | [-] | 0,935 (0,963 na ponta) |
| `D_2` | Diâmetro interno do tubo absorvedor metálico | m | 0,076 m |
| `D_3` | Diâmetro externo do tubo absorvedor metálico | m | 0,080 m |
| `D_4` | Diâmetro interno do envelope de vidro | m | 0,115 m |
| `D_5` | Diâmetro externo do envelope de vidro | m | 0,120 m |
| `AnnulusGas` | Gás no espaço anular do receptor (4 = Hidrogênio com vácuo) | [-] | 27 (Hidrogênio) |
| `P_a` | Pressão do gás no espaço anular | torr | 0,0001 torr |
| `epsilon_3` | Tabela de emissividade térmica do absorvedor em função da temperatura | [-] | Tabela polinomial (0,064 a 0,112) |
| `epsilon_4` | Emissividade da superfície interna do vidro | [-] | 0,86 |
| `EPSILON_5` | Emissividade da superfície externa do vidro | [-] | 0,86 |

---

## 3. Inércia Térmica e Perdas nas Tubulações (HTF)

| Nome no SAM (C++) | Descrição | Unidade | Valor Definido (JSON / Usuário) |
| :--- | :--- | :--- | :--- |
| `Pipe_hl_coef` | Coeficiente global de perda térmica nas tubulações (headers e runners) | W/(m²·K) | 0,45 W/(m²·K) |
| `mc_bal_hot` | Capacidade térmica da tubulação de saída (*hot balance of plant*) | kWht/(K·MWt) | 0,20 kWht/(K·MWt) |
| `mc_bal_cold` | Capacidade térmica da tubulação de retorno (*cold balance of plant*) | kWht/(K·MWt) | 0,20 kWht/(K·MWt) |
| `mc_bal_sca` | Inércia térmica de componentes não-fluido por SCA (suportes, flanges) | Wht/(K·m) | 4,5 Wht/(K·m) |
| `m_dot_htfmin` | Vazão mínima mássica permitida por loop | kg/s | 1,0 kg/s |
| `m_dot_htfmax` | Vazão máxima mássica permitida por loop | kg/s | 12,0 kg/s |
| `Fluid` | Fluido de transferência de calor do campo solar | [-] | 21 (Therminol VP-1) |

---

## 4. Consumo Parasitário e Bombeamento do Campo e TES

| Nome no SAM (C++) | Descrição | Unidade | Valor Definido (JSON / Usuário) |
| :--- | :--- | :--- | :--- |
| `SCA_drives_elec` | Potência elétrica consumida pelos motores de rastreamento de cada SCA | W/SCA | 125,0 W/SCA |
| `pb_fixed_par` | Fração constante de carga parasitária fixa do ciclo e campo | MWe/MWcap | 0,0055 (0,55% de P_ref) |
| `pb_pump_coef` | Coeficiente de potência da bomba de alimentação do ciclo de potência | kW/(kg/s) | 0,55 kW/(kg/s) |
| `tes_pump_coef` | Coeficiente de potência das bombas do trocador de calor do TES | kW/(kg/s) | 0,15 kW/(kg/s) |
| `wind_stow_speed` | Velocidade do vento para recolhimento e travamento (*stow*) dos coletores | m/s | 25,0 m/s |
| `bop_array` | Coeficientes do polinômio de consumo de BOP em função da carga | [-] | [0, 1, 0.483, 0.517, 0] |
| `aux_array` | Coeficientes do consumo de auxiliares em função da carga | [-] | [0.02273, 1, 0.483, 0.517, 0] |
| `dt_hot` | Delta T de aproximação no trocador de calor do TES | °C | 5,0 °C |
| `hot_tank_Thtr` | Setpoint do aquecedor anticongelamento do tanque quente | °C | 365,0 °C |
| `cold_tank_Thtr` | Setpoint do aquecedor anticongelamento do tanque frio | °C | 250,0 °C |
