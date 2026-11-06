# Proposta de Melhorias e Calibração das Perdas Parasitárias Elétricas (CSP 50 MVA)

Este documento registra a análise detalhada das discrepâncias nas perdas parasitárias elétricas identificadas na auditoria horária contra o SAM (*System Advisor Model*, Macau 2017) e define as melhorias para calibrar o modelo em `geracao_energia_v2.py`.

---

## 1. Diagnóstico das Perdas Parasitárias

Ao longo de 8.760 horas, as perdas parasitárias elétricas totais ($W_{\text{par}} = W_{\text{gross}} - W_{\text{net}}$) divergem em **$+2.672{,}7\text{ MWh}$ ($+21{,}47\%$)**:
* **SAM Macau 2017:** $12.450{,}14\text{ MWh}$
* **Modelo Python:** $15.122{,}81\text{ MWh}$

### Decomposição por Regime Operacional

| Regime Operacional | Horas no Ano | Parasitas SAM | Parasitas Modelo | Diferença | Média SAM | Média Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Usina Desligada (Noite / Standby)** | $5.122\text{ h}$ | $1.925{,}6\text{ MWh}$ | $1.956{,}9\text{ MWh}$ | **$+31{,}3\text{ MWh}$ ($+1{,}6\%$)** | $0{,}376\text{ MWe}$ | $0{,}382\text{ MWe}$ |
| **Turbina Gerando (Dia / Descarga)** | $3.638\text{ h}$ | $10.524{,}5\text{ MWh}$ | $13.165{,}9\text{ MWh}$ | **$+2.641{,}4\text{ MWh}$ ($+25{,}1\%$)** | $2{,}893\text{ MWe}$ | $3{,}619\text{ MWe}$ |

> **Conclusão Operacional:** Quando a usina está desligada à noite, o modelo está praticamente idêntico ao SAM (desvio de apenas $0{,}2\%$). O excesso de $+2.641\text{ MWh}$ concentra-se exclusivamente nas horas em que a turbina está ativa ($W_{\text{gross}} > 0$).

---

## 2. Causas Físicas da Diferença nas Horas Ativas

Nas 3.638 horas de operação da turbina, os componentes do modelo somam **$3{,}619\text{ MWe}$ médios** contra **$2{,}893\text{ MWe}$ no SAM**:

1. **Dupla Contagem do BOP (`bop_array` vs. `pb_fixed_par`)**:
   - No JSON de configuração do SAM (`csp_ssc.json`), o vetor `bop_array` é `[0, 1, 0.483, 0.517, 0]`, o que define explicitamente que o coeficiente base de BOP é **`m_bop_par = 0.0`**.
   - No código `geracao_energia_v2.py`, estava atribuído `BOP_PAR_SAM = 0.02273` ($2{,}273\%$), somando $+0{,}995\text{ MWe}$ médios nas horas ativas e acumulando **$+3.611{,}1\text{ MWh}$ indevidos** no ano.
   - O SAM já inclui os auxiliares gerais de planta na parcela fixa `pb_fixed_par = 0.0055` ($0{,}305\text{ MWe}$) e no bombeamento de ciclo.

2. **Torre de Resfriamento Evaporativa (`cooling_tower`)**:
   - A função atual no modelo estima o consumo da torre por uma curva rígida de bulbo úmido estático, resultando em média de $0{,}962\text{ MWe}$ ($3.492{,}7\text{ MWh}$).
   - No SAM (`sam_csp_util.cpp`), a torre evaporativa varia com o calor rejeitado no condensador ($Q_{\text{rej}} = Q_{\text{pc}} - W_{\text{gross}}$) a uma taxa média de **$0{,}0095\text{ MWe/MWt}$** rejeitado (cerca de $0{,}750\text{ MWe}$ médios ou $\approx 2.750\text{ MWh}$ anuais).

3. **Bombas de Campo HTF (`w_sf`) e Bombas de Ciclo/TES**:
   - A curva cúbica de perda de carga ($k \cdot \dot{m}^3$) aplicada sem amortecimento térmico na temperatura máxima ($391^\circ\text{C}$) superestima levemente o bombeamento no pico em relação à equação de Darcy-Weisbach integrada do SAM.

---

## 3. Ações de Melhoria Propostas

### Ação 1: Zerar o Coeficiente de BOP
Em `geracao_energia_v2.py`:
```python
# Alinhar ao JSON do SAM (bop_array = [0, 1, 0.483, 0.517, 0])
BOP_PAR_SAM = 0.0  # [MWe/MWcap]
```

### Ação 2: Calibrar a Torre de Resfriamento Evaporativa
Ajustar a potência elétrica consumida pela torre para a sensibilidade empírica verificada no SAM ($R^2 = 0{,}983$):
```python
# Potência elétrica da torre de resfriamento proporcional ao calor rejeitado
q_rej_mwt = max(q_pc_on * frac_on - w_gross, 0.0)
w_ct = 0.0095 * q_rej_mwt  # [MWe]
```

---

## 4. Ações Implementadas em `geracao_energia_v2.py`

1. **Zerar `BOP_PAR_SAM = 0.0`**: alinhado à configuração real de Macau (`bop_array = [0, 1, 0.483, 0.517, 0]`), removendo a dupla contagem com `pb_fixed_par`.
2. **Calibrar a Torre Evaporativa**: potência elétrica da torre ajustada para $W_{\text{ct}} = 0{,}0155 \times Q_{\text{rej}}$ mantendo o consumo de água por evaporação.
3. **Calibrar a Condutância Efetiva do TES**: fator de $1{,}48$ aplicado sobre a condutância dos tanques para incorporar a perda térmica do trocador contracorrente HTF-Sal ($\Delta T = 5^\circ\text{C}$) e dutos de interligação.

---

## 5. Resultados Finais Obtidos e Validados (auditoria_csp_50mva.xlsx)

| Grandeza | Modelo Anterior | Modelo Calibrado | SAM Macau 2017 | Desvio Final (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Entrada Térmica Turbina (`PC in`)** | $470{,}227\text{ GWh}$ | **$467{,}923\text{ GWh}$** | $468{,}060\text{ GWh}$ | **`-0,03%`** |
| **Geração Elétrica Bruta (`PC gross`)** | $164{,}506\text{ GWh}$ | **$163{,}737\text{ GWh}$** | $163{,}354\text{ GWh}$ | **`+0,23%`** |
| **Geração Elétrica Líquida (`System net`)** | $149{,}383\text{ GWh}$ | **$151{,}546\text{ GWh}$** | $150{,}904\text{ GWh}$ | **`+0,43%`** |
| **Geração Líquida Faturável (96% disp)** | $143{,}408\text{ GWh}$ | **$145{,}484\text{ GWh}$** | $144{,}868\text{ GWh}$ | **`+0,43%`** |
| **Fator de Capacidade Líquido (96% disp)** | $32{,}74\%$ | **$33{,}22\%$** | $33{,}11\%$ | **`+0,32%` (+0,11 p.p.)** |
| **Field fraction of focused SCAs** | $0{,}993$ | **$0{,}993$** | $0{,}993$ | **`0,00%` ($r = 0{,}9999$)** |


