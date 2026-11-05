# Checkpoint — Tradução da Física CSP/TES do SAM (C++) para Python e Validação (Macau 2017)

Data de salvamento: 2026-11-04

---

## 1. Objetivo Atual
Traduzir os métodos físicos do SAM (presentes na pasta `src_cpp/`) para o código Python do projeto (`geracao_energia_v2.py` e `otimizacao_nsga_ii_v2.py`), a partir da etapa em que a radiação solar atinge os receptores até a potência térmica entregue ao ciclo de potência, perdas térmicas e parasitárias no TES/campo/ciclo e geração elétrica líquida, simplificando apenas sub-iterações pesadas que tenham impacto desprezível na energia anual gerada.

### Caso de Referência para Validação (`SAM_Macau_2017.xlsx`)
- **Arquivo climático**: `historico_variaveis/nsrdb_macau/2008167_-5.11_-36.62_2017.csv`
- **Configuração da usina**: 50 MVA líquidos (`49,95 MWe` net / `55,5 MWe` gross, fator bruto-líquido de `0,90`), 5 horas de TES, 91 loops (`8 SCAs/loop`, `A_loop = 656 m²`, `A_total = 59.696 m²`), `eta_ref = 0,356`, `Q_pc_design = 155,898876 MWt`, `Q_tes_design = 779,49438 MWht`.

#### Somas Anuais das 14 Colunas Horárias (`SAM_Macau_2017.xlsx`):
1. `Resource Beam normal irradiance (W/m2)`: `1.963.210,000`
2. `Field thermal power incident (MWt)`: `937.557,193 MWh_t`
3. `Field collector cosine efficiency`: `4.789,716`
4. `Field thermal power incident after cosine (MWt)`: `897.921,144 MWh_t`
5. `Receiver thermal power incident (MWt)`: `605.498,431 MWh_t`
6. `Receiver thermal losses (MWt)`: `97.859,409 MWh_t`
7. `Receiver thermal power absorbed (MWt)`: `485.235,604 MWh_t`
8. `Field change in material/htf internal energy (MWt)`: `3.134,963 MWh_t`
9. `Field thermal power leaving in HTF (MWt)`: `473.318,630 MWh_t`
10. `TES charge thermal power (MWt)`: `147.042,924 MWh_t`
11. `TES discharge thermal power (MWt)`: `141.792,495 MWh_t`
12. `PC input energy (MWt)`: `468.059,796 MWh_t`
13. `PC electrical power output: gross (MWe)`: `163.353,794 MWh_e`
14. `System net electrical power (MWe)`: `150.903,654 MWh_e` (com disponibilidade de 96%: `150.903,654 * 0,96 = 144.867,508 MWh_e`)

#### Métricas Anuais de Referência do SAM:
| Métrica | Valor SAM |
| :--- | :--- |
| Annual AC energy in year 1 | `144.867.504 kWhe` |
| Annual freeze protection | `176.518 kWhe` |
| Annual TES freeze protection | `171.156 kWhe` |
| Annual field freeze protection | `5.362 kWhe` |
| Capacity factor | `33,1 %` |
| Power cycle gross electrical output | `163.353.792 kWhe` |
| First year kWh/kW | `2.900` |
| Gross-to-net conversion | `88,7 %` |
| Annual water usage | `603.634 m³` |

---

## 2. O Que Já Foi Feito

### 2.1. Leitura e Mapeamento Completo do Código C++ (`src_cpp/`) e Python
Foram lidos e analisados todos os arquivos C++ de referência do SAM e os módulos Python do projeto:
- **C++ (`src_cpp/`)**:
  - `csp_solver_trough_collector_receiver.h` / `.cpp`: Geometria do campo (`init_fieldgeom`), dimensionamento de runners e headers (`rnr_and_hdr_design`, `size_hdr_lengths`, `size_rnr_lengths`, `Pump_SGS`), balanço térmico transiente nos SCAs e tubulações (`loop_energy_balance_T_t_int`), modos operacionais (`off`, `startup`, `on`, `freeze_protection`) e defocus (`m_component_defocus` e `m_control_defocus`).
  - `sam_csp_util.cpp`: Modelo térmico 1D radial de Forristall (`Evacuated_Receiver::EvacReceiver`, `FQ_34CONV`, `FQ_34RAD`, `FQ_56CONV`, `FQ_COND_BRACKET`, `fT_2`, `FK_23`), correlações de perda de carga (`PipeFlow`, `FrictionFactor`, `pipe_sched`) e torre de resfriamento evaporativa (`C_evap_tower`).
  - `csp_solver_two_tank_tes.h` / `.cpp`: Dimensionamento e operação do TES indireto de dois tanques (`C_csp_two_tank_tes`), balanço diferencial analítico dos tanques quente e frio (`C_storage_tank::energy_balance`), aquecedores de proteção contra congelamento e trocador de calor em contracorrente (`C_hx_two_tank_tes`).
  - `csp_solver_core.h` / `.cpp` e `cmod_trough_physical.cpp`: Controlador da planta (`C_csp_solver::sim_core`), parâmetros padrão do ciclo Rankine, BOP e auxiliares.
- **Python**:
  - `geracao_energia_v2.py`, `otimizacao_nsga_ii_v2.py`, `modelagem_csp.py`, `geracao_series_sinteticas.py`.

### 2.2. Identificação das Divergências Físicas entre o Código Atual e o SAM
1. **Fluido do Campo Solar vs. Fluido do TES**:
   - O campo solar utiliza **Therminol VP-1** (`Fluid(0)`), enquanto o TES utiliza **Hitec Solar Salt** (`Fluid(1)`) separado por um trocador de calor indireto (`C_hx_two_tank_tes`, com `dt_hot = 5 °C`). No código Python anterior (`geracao_energia_v2.py`), o `Fluid(1)` estava sendo usado para o calor específico do campo solar.
2. **Unidades de Temperatura na Classe `Fluid` (`modelagem_csp.py`)**:
   - Os métodos `Fluid.Cp(T_C)`, `Fluid.dens(T_C, P)`, `Fluid.visc(T_C)` e `Fluid.cond(T_C)` em `modelagem_csp.py` esperam temperatura em **°C** (somando `273,15` internamente), ao passo que no C++ (`HTFProperties`) as funções recebem **Kelvin**.
3. **Coeficiente de Perda Térmica nas Tubulações (`Pipe_hl_coef`)**:
   - Em `geracao_energia_v2.py`, `PIPE_HL_COEF_SAM` estava definido como `10,0 W/(m²·K)`, enquanto o valor padrão do SAM (e de `modelagem_csp.py` linha 280) é **`0,45 W/(m²·K)`**.
4. **Definição de `Receiver thermal power incident` (`E_Q_DOT_REC_INC`) no SAM**:
   - No SAM (`csp_solver_trough_collector_receiver.cpp`, linhas 2243–2244 e 5100–5104), a potência térmica incidente reportada no receptor inclui a transmitância do envelope de vidro (`TAU_ENV = 0,964` para tubos intactos) e ambos os fatores de defocus (`m_component_defocus * m_control_defocus`), sendo calculada como `(q_abs + q_loss) / ALPHA_ABS`.
5. **Vazão de Recirculação durante o `startup()` do Campo Solar**:
   - No SAM (`csp_solver_trough_collector_receiver.cpp`, linhas 2458–2464), a decisão entre vazão mínima (`1 kg/s`) ou vazão proporcional à potência absorvida (`m_T_htf_out_t_end_converged[7] > 0,5*(T_fp + T_startup) = 518,15 K`) é tomada **uma única vez no início do passo horário**, antes do loop de subpassos de `600 s`.
6. **Proteção contra Congelamento do Campo (`Annual field freeze protection = 5.362 kWhe`)**:
   - No SAM (`csp_solver_trough_collector_receiver.cpp`, linhas 3033–3096), quando `T_out_SCA8 < T_fp (423,15 K)`, o método `freeze_protection` eleva as temperaturas dos nós para que a média ponderada pelo volume (`(V_hot * T_out + V_loop * 0,5*(T_in + T_out) + V_cold * T_in) / V_tot`) atinja exatamente `T_fp`, contabilizando a energia térmica/elétrica correspondente.
7. **TES Indireto de Dois Tanques e Proteção contra Congelamento do TES (`Annual TES freeze protection = 171.156 kWhe`)**:
   - Como o fluido do campo (`VP-1`) difere do fluido do TES (`Hitec Solar Salt`), o SAM ativa `m_is_hx = true` e `tanks_in_parallel = true`:
     - `T_tes_hot_des = 391 - 5 = 386 °C (659,15 K)`
     - `T_tes_cold_des = 293 + 5 = 298 °C (571,15 K)`
     - Tanque quente inicia com apenas o volume inativo (`h_min = 1 m` de `h_tank = 12 m`) a `391 °C` e, quando ocioso, resfria até o setpoint do aquecedor (`hot_tank_Thtr = 365 °C`), acionando o aquecedor de imersão (`C_storage_tank::energy_balance`), o que gera exatamente os `~171,156 MWh` anuais de `TES freeze protection`.
8. **Ciclo de Potência Rankine, Parasitárias e Consumo de Água**:
   - Eficiência bruta horária exata do ciclo Rankine em função da carga adimensional `q_norm = q_pc_on / q_dot_pc_des` e `T_dry [°C]`:
     $$\eta_{\text{gross}} = 0{,}3875325 + 0{,}0856621\,q_{\text{norm}} - 0{,}0635353\,q_{\text{norm}}^2 - 0{,}0035352\,T_{\text{dry}} + 4{,}67819\times 10^{-5}\,T_{\text{dry}}^2 + 6{,}58292\times 10^{-4}\,q_{\text{norm}}\,T_{\text{dry}}$$
   - Partida do ciclo (`startup_time = 0,5 h`, `startup_frac = 0,2`, `E_su_req = 31,180 MWht`, `q_su_max = 62,360 MWt`), corte mínimo (`cycle_cutoff_frac = 0,2 -> 31,180 MWt`) e sobrecarga máxima (`cycle_max_frac = 1,05 -> 163,694 MWt`).
   - Perdas parasitárias (`W_gross - W_net = 12.450,14 MWh_e` no ano): soma de parasitária fixa (`0,0055 * P_ref = 0,30525 MWe`), acionamento dos SCAs (`125 W/SCA * 728 SCAs = 0,091 MWe` quando rastreando), bombas de HTF do campo solar, bombas do TES e do ciclo, aquecedores de anticongelamento (campo + TES), BOP e torre de resfriamento evaporativa (`C_evap_tower`).

### 2.3. Protótipos e Scripts de Teste Criados (Apenas na Pasta `scratch/`)
Nenhum arquivo principal do projeto (`geracao_energia_v2.py` ou `otimizacao_nsga_ii_v2.py`) foi alterado ainda. Foram criados dois scripts auxiliares de verificação na pasta `scratch`:
1. `C:\Users\victo\.gemini\antigravity\brain\c345f508-e215-480e-aeec-155215a9d525\scratch\test_sam_csp.py`:
   - Implementou a geometria completa de headers/runners/SGS do SAM (`init_fieldgeom_sam`), o modelo de receptor evacuado de Forristall (`evac_receiver_sam`) e o balanço transiente de energia do campo solar (`loop_energy_balance`, `off`, `startup`, `on`, `freeze_protection`).
   - Validou `Annual field freeze protection = 5.344 kWhe` vs. `5.362 kWhe` do SAM (`0,3%` de diferença), além das somas anuais de perdas no receptor, absorção térmica, energia interna e potência térmica entregue pelo HTF.
2. `C:\Users\victo\.gemini\antigravity\brain\c345f508-e215-480e-aeec-155215a9d525\scratch\analyze_sam_excel.py`:
   - Analisou e regrediu o comportamento horário exato de `SAM_Macau_2017.xlsx` para o despacho do TES, eficiência bruta do ciclo Rankine e decomposição das potências parasitárias (`W_par = W_gross - W_net`) em todos os regimes operacionais (noite desligado, partida, descarga pura do TES e operação diurna com carga do TES).

---

## 3. O Que Falta Fazer (Passo a Passo)

### Passo 1: Finalizar e Validar o Protótipo Completo em `scratch/test_sam_csp.py`
1. Aplicar os dois ajustes finais no módulo do campo solar dentro de `scratch/test_sam_csp.py`:
   - Incluir `* TAU_ENV[0]` em `q_rec_inc` reportado e escalonar pelo defocus total (`df_comp * df_ctrl`) para igualar a coluna `Receiver thermal power incident (MWt)` (`605.498,431 MWh_t`).
   - Mover a verificação `t_end_prev[7] > 0.5 * (T_fp + T_su)` para fora do loop de subpassos de `600 s` em `run_off_or_su_substeps` (idêntico ao `startup()` do SAM).
   - Otimizar o loop interno de `evac_receiver_sam` (pré-calculando constantes geométricas dos 4 estados de HCE e reduzindo iterações redundantes) para que a simulação anual de 8.760 h rode em ~1 segundo.
2. Adicionar ao `scratch/test_sam_csp.py` a classe fiel do TES indireto de dois tanques (`C_storage_tank`, `C_hx_two_tank_tes`, `C_csp_two_tank_tes`) e o despacho acoplado Campo Solar + TES + Ciclo Rankine + Torre de Resfriamento (`C_evap_tower`).
3. Executar `scratch/test_sam_csp.py` e conferir lado a lado todas as **14 colunas horárias** de `SAM_Macau_2017.xlsx` e as **9 métricas anuais** da tabela do SAM (energia líquida com 96% de disponibilidade, proteções anticongelamento do campo e do TES, fator de capacidade, geração bruta, conversão bruto-líquido e consumo de água).

### Passo 2: Integrar a Nova Física em `geracao_energia_v2.py`
1. Atualizar `simular_receptor_csp` em `geracao_energia_v2.py`:
   - Incorporar o dimensionamento geométrico do campo (`init_fieldgeom_sam` para qualquer `n_loops_real`), o modelo `evac_receiver_sam` acelerado, o balanço transiente `loop_energy_balance` com `Therminol VP-1` (`Fluid(0)`), `Pipe_hl_coef = 0,45 W/(m²·K)` e a contabilização de `freeze_protection` do campo e potência de bombeamento/rastreamento.
2. Atualizar `calcular_perda_tes_w` e `despachar_csp_tes` em `geracao_energia_v2.py`:
   - Substituir o modelo simplificado de tanque único pelo modelo de dois tanques indiretos do SAM (`C_storage_tank` + `C_hx_two_tank_tes` com `Hitec Solar Salt` `Fluid(1)` no TES e `Therminol VP-1` `Fluid(0)` no campo/ciclo), incluindo o controle acoplado de defocus quando o tanque quente atinge a capacidade máxima, aquecedores de anticongelamento dos tanques (`hot_tank_Thtr = 365 °C`, `cold_tank_Thtr = 250 °C`), curva de eficiência bruta do ciclo Rankine em função da carga e `T_dry`, perdas parasitárias completas e consumo de água.
   - Garantir que `despachar_csp_tes` mantenha assinatura e retorno compatíveis com o restante do projeto (retornando os vetores em Watts `[W]` e dicionário/atributos extras de auditoria quando solicitado).

### Passo 3: Atualizar `otimizacao_nsga_ii_v2.py` e Validar Auditoria
1. Ajustar as chamadas em `precalcular_perfis_unitarios`, `avaliar_configuracao_hibrida`, `executar_validacao_sam` e `exportar_auditoria_csp_sam` em `otimizacao_nsga_ii_v2.py` para refletir com fidelidade todas as grandezas calculadas pelo novo modelo físico.
2. Garantir que o cache físico (`CacheFisicoUnitario`) e a avaliação no loop do NSGA-II permaneçam rápidos e vetorizados/eficientes.
3. Executar `py -3.11 otimizacao_nsga_ii_v2.py` para gerar a tabela comparativa no terminal e a planilha `auditoria_csp_50mva.xlsx`, confirmando a aderência aos resultados de `SAM_Macau_2017.xlsx`.

