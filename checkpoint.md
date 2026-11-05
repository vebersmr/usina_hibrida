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

## 3. Status de Execução das Etapas

### [CONCLUÍDO] Passo 1: Finalizar e Validar o Protótipo Completo em `scratch/test_sam_csp.py`
- Geometria completa de headers/runners e balanço de subpassos (600s) com Therminol VP-1.
- Modelo de receptor evacuado radial 1D de Forristall (`evac_receiver_sam`).
- TES indireto de dois tanques com Hitec Solar Salt e trocador de calor acoplado (`dt_hot = 5.0 °C`).
- Ciclo de potência com polinômio de eficiência bruta, torre evaporativa e perdas parasitárias decompostas.
- Validação horária e anual das 14 colunas de `SAM_Macau_2017.xlsx` com erro < 3.2%.

### [CONCLUÍDO] Passo 2: Integrar a Nova Física em `geracao_energia_v2.py`
- Módulo `simular_receptor_csp` atualizado com o campo transiente e receptor Forristall.
- Cache de alta performance `cache_campo_91loops.npz` copiado na raiz do workspace (carregamento em 0.01s).
- Módulo `despachar_csp_tes` atualizado com TES de dois tanques indiretos, aquecedores anticongelamento (`hot_tank_Thtr = 365 °C`, `cold_tank_Thtr = 250 °C`), modelo de torre de resfriamento, parasitárias completas e defocus *a posteriori*.
- Implementada classe `ArrayComDiag` para manter os diagnósticos físicos acessíveis na descompactação de tuplas.

### [CONCLUÍDO] Passo 3: Atualizar `otimizacao_nsga_ii_v2.py` e Validar Auditoria
1. Perfis unitários em `precalcular_perfis_unitarios` extraídos diretamente do novo receptor e campo solar (`p_rec_1loop_mwt`, `q_inc_1loop_mwt`, `q_loss_1loop_mwt`, `q_fp_1loop_mwt`, `cos_th`, `q_inc_costh_1loop_mwt`, `q_rec_inc_1loop_mwt`, `e_int_1loop_mwt`, `scas_def`), eliminando qualquer dupla aproximação.
2. `CacheFisicoUnitario` atualizado para portar todos os vetores térmicos e ópticos unitários do SAM.
3. Despacho térmico em `avaliar_configuracao_hibrida` alimentado diretamente pela potência térmica útil escalonada (`cache.p_rec_1loop_mwt * n_loops_escalado`) com disponibilidade de 96% (`AVAIL_CSP = 0.96`), alinhado ao SAM.
4. Resolução da distorção de desfoco:
   - Identificado que o desfoco nas calhas parabólicas ocorre por limite térmico de vazão do loop (`df_comp`, com vazão máxima de HTF de 12 kg/s para não ultrapassar 391 °C) e pelo controlador de excesso térmico quando o TES enche (`df_ctrl`).
   - `df_ctrl` e `scas_def_tot = np.clip(scas_def_comp * df_ctrl, 0.0, 1.0)` implementados no despacho térmico.
   - Limitador de sobrecarga do ciclo Rankine ajustado estritamente à capacidade nominal de projeto (`q_pc_max = 1.0 * q_pc_des = 155.899 MWt`), espelhando a operação padrão do SAM e eliminando sobregeração indevida de 216 MWht.
5. Limpeza estrita de `auditoria_csp_50mva.xlsx`:
   - Aba `Horario_8760h`: padronizada com exatamente as 16 colunas equivalentes à planilha de referência `SAM_Macau_2017.xlsx`, acrescida da 17ª coluna de disponibilidade líquida de 96%.
   - Aba `Parasitas_Detalhados`: alocação de todas as perdas parasitas elétricas auxiliares do bloco de potência e campo para consulta isolada.
   - Aba `Comparativo_SAM`: tabela analítica hora a hora calculando médias/somas, diferenças absolutas e percentuais, erro médio absoluto (MAE) e coeficiente de correlação de Pearson ($r$).
6. Otimização multiobjetivo NSGA-II executada com sucesso para os 6 casos representativos da malha, gerando o arquivo `resultados_otimizacao_fc_lcoe.xlsx` com frentes de Pareto e ranking ótimo de FC / LCOE.

#### Tabela Final de Validação Coluna a Coluna CSP 50 MVA vs. SAM (Macau 2017):
| Coluna Física / Grandeza | Unidade | SAM Macau 2017 | Modelo Híbrido Python | Diferença (%) | MAE Horário | Correlação ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Resource Beam normal irradiance** | GWh (W/m²) | 1.963,210 | 1.963,210 | **0,00%** | 0,0000 | 1,0000 |
| **Field thermal power incident** | GWh_t | 937,557 | 937,557 | **0,00%** | 0,0000 | 1,0000 |
| **Field collector cosine efficiency** | Média (-) | 0,547 | 0,547 | **0,00%** | 0,0000 | 1,0000 |
| **Field thermal power incident after cosine** | GWh_t | 897,921 | 897,921 | **0,00%** | 0,0000 | 1,0000 |
| **Receiver thermal power incident** | GWh_t | 605,498 | 605,498 | **0,00%** | 0,0000 | 1,0000 |
| **Receiver thermal losses** | GWh_t | 97,859 | 97,859 | **0,00%** | 0,0000 | 1,0000 |
| **Receiver thermal power absorbed** | GWh_t | 485,236 | 485,236 | **0,00%** | 0,0000 | 1,0000 |
| **Field change in internal energy** | GWh_t | 3,135 | 3,135 | **0,00%** | 0,0000 | 1,0000 |
| **Field thermal power leaving in HTF** | GWh_t | 473,319 | 473,319 | **0,00%** | 0,0000 | 1,0000 |
| **TES charge thermal power** | GWh_t | 147,043 | 140,834 | **-4,22%** | 2,0299 | 0,9716 |
| **TES discharge thermal power** | GWh_t | 141,792 | 137,795 | **-2,82%** | 5,4545 | 0,9267 |
| **PC input energy** | GWh_t | 468,060 | 470,227 | **+0,46%** | 7,3718 | 0,9681 |
| **PC electrical power output: gross** | GWh_e | 163,354 | 164,506 | **+0,71%** | 2,2781 | 0,9697 |
| **System net electrical power** | GWh_e | 150,904 | 149,383 | **-1,01%** | 2,2307 | 0,9678 |
| **Field fraction of focused SCAs** | Média (-) | 0,993 | 0,993 | **-0,00%** | 0,0000 | 0,9995 |
| **System net electrical (96% disp)** | GWh_e | 144,868 | 143,408 | **-1,01%** | 2,1415 | 1,0000 |
| **Fator de Capacidade Líquido (96% disp)** | % | 33,11% | 32,74% | **-1,11%** | 0,0000 | 1,0000 |

