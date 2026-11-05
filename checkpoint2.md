# Checkpoint 2: Passo 1 (usina CSP completa vs. SAM Macau 2017)

Data: 2026-11-04 (23:31)

> [!IMPORTANT]
> **Correções ao `checkpoint.md` anterior.** Algumas afirmações ali estavam erradas ou não tinham sido verificadas:
> 1. **"Campo solar validado a 0,1–1% nas somas anuais" estava errado.** O protótipo foi rodado agora (seção 2). Nas primeiras 24 h ele bate bem com o SAM (diferenças < 1–3%), mas **nas somas anuais as diferenças chegam a 2–5%**. A variação de energia interna chega a −38%.
> 2. **Área do campo:** o certo é `A_SCA = 656 m²`, `A_loop = 5.248 m²` e `A_total = 91 × 5.248 = 477.568 m²`, não os "59.696 m²" que constavam.
> 3. **`P_ref`:** o protótipo do campo usava `55,555556 MWe`, mas o valor certo do caso SAM é `55,5 MWe` (49,95 / 0,9). O novo script já passa `55,5`.
> 4. **Proteção anticongelamento do TES (171,156 MWh):** a explicação pelo aquecedor do tanque quente é só uma **hipótese**. Ainda não foi reproduzida numericamente.
> 5. **Tempo de execução:** o campo solar leva **~78 s** por simulação anual. O "~1 s" era uma meta, não um resultado.
> 6. As estimativas de progresso (~65–70%) e de "3–4 mensagens até concluir" que dei antes foram otimistas. O que eu disse sobre cotas do Google AI Pro/Opus também não se baseava em informação verificada. Consulte a documentação oficial do seu plano.

---

## 1. O que foi feito nesta sessão

1. **Rodei a decomposição das parasitas** (`scratch/analyze_sam_excel.py`):
   - Noite (desligado): `W_par ≈ 0,3067 MWe`, dos quais 0,30525 são a parasita fixa (0,0055 × 55,5).
   - Descarga pura do TES (137 h): `W_par ≈ 1,200 + 0,00519·Q_pc − 3,5e-6·Q_pc² + 0,00327·T_dry` (erro máx. 0,022 MWe). O termo constante de ~1,2 MWe vem da parasita fixa somada à torre de resfriamento (vazão de água de projeto constante).
   - Dia com carga do TES: `W_par ≈ 4,85–4,95 MWe`. Isso indica que a bomba do campo solar dá ~3 MWe na vazão de projeto.
2. **Rodei o protótipo do campo** `scratch/test_sam_csp.py`, sem alterações. O resultado está na seção 2.
3. **Li no C++ as rotinas de TES e parasitas que faltavam:**
   - `C_storage_tank::init/energy_balance/m_dot_available` (linhas 46–274)
   - `C_hx_two_tank_tes::solve` (1258–1325)
   - `C_csp_two_tank_tes::init` (1424–1696)
   - `discharge/charge/discharge_avail_est/charge_avail_est` (1824–1895, 2181–2380)
   - `pumping_power` (2471–2519): com HX, `W = tes_pump_coef·(m_tank + |m_pb − m_sf|)/1000`
   - BOP em `csp_solver_core.cpp` (545–548)
   - **Observação:** o cálculo final de `W_dot_net` (classe `C_csp_solver_system_calc_metrics`) é feito em um arquivo que **não está** em `src_cpp/` (provavelmente `csp_solver_core_operating_modes.cpp` ou `csp_solver_util.cpp`). O modelo do ciclo Rankine (`csp_solver_pc_Rankine_indirect_224.cpp`) e a torre (`water_properties`) também não estão. Se quiser fidelidade total nas parasitas e no ciclo, **baixe esses arquivos para `src_cpp/`**.
4. **Escrevi `scratch/test_sam_plant.py` (ainda NÃO executado; a execução foi interrompida a seu pedido).** Ele contém:
   - Tradução direta de `C_storage_tank`, `C_hx_two_tank_tes` e do dimensionamento do TES, com Hitec Solar Salt no TES e VP-1 no campo.
   - Controlador horário heurístico: partida do ciclo (0,5 h / 31,18 MWht), carga e descarga do TES, sobrecarga até 1,05 e desfocagem.
   - Parasitas: fixa, rastreamento, bomba do ciclo (0,55 kW/(kg/s)), bomba do TES (0,15), torre evaporativa simplificada, aquecedores do TES, anticongelamento do campo e bomba do campo (`k·m³`).
   - Cache do campo (`scratch/cache_campo_91loops.npz`): só a primeira execução leva ~80 s.
   - Imprime as somas anuais das colunas 5–14, a correlação horária e as 9 métricas anuais. Também salva `scratch/resultado_plant_8760.csv`.
   - Simplificações marcadas no código com `[APROX]`:
     - Eficiência do ciclo pela correlação ajustada ao próprio Excel.
     - Bomba do campo com 1 constante calibrada no próprio Excel (~2,98 MWe a 1140 kg/s).
     - Propriedades constantes da água na torre.
     - Entrada do campo fixa em 293 °C.
     - Desfocagem aplicada depois da simulação, sem re-simular o campo.
   - **Atenção:** as duas calibrações no próprio Excel tornam a validação parcialmente circular. Isso deve ser declarado na dissertação ou removido, se você obtiver os arquivos C++ do ciclo.

---

## 2. Resultado verificado do campo solar (protótipo atual, 91 loops)

| Coluna SAM | Python | SAM | Dif. |
|---|---|---|---|
| Field thermal power incident | 937.566 | 937.557 | +0,00% |
| Field cosine efficiency (soma) | 4.029 | 4.790 | −15,9% (*) |
| Incident after cosine | 897.599 | 897.921 | −0,04% |
| Receiver thermal power incident | 635.221 | 605.498 | **+4,91%** |
| Receiver thermal losses | 95.050 | 97.859 | −2,87% |
| Receiver thermal power absorbed | 494.646 | 485.236 | **+1,94%** |
| Field change in internal energy | 1.954 | 3.135 | **−37,7%** |
| Field thermal power leaving in HTF | 485.753 | 473.319 | **+2,63%** |
| Field freeze protection [MWh] | 5,344 | 5,362 | −0,3% |

(*) O SAM provavelmente reporta cos θ também em horas noturnas ou em outra convenção. Não afeta a energia.

**Causas prováveis, a confirmar:**
- Falta a desfocagem quando o TES está cheio, porque o campo isolado nunca desfoca. Isso explica o excesso em absorbed e em HTF.
- Falta o fator `TAU_ENV = 0,964` em `q_rec_inc`. Esse fator sozinho leva +4,9% para ≈ +1,1%.
- Diferenças na partida matinal (ex.: hora 8: 162 vs. 144 MWt; perda na hora 7: 9,7 vs. 14,3 MWt).

---

## 3. O que falta fazer (passo a passo)

### Passo 1 (em andamento)
1. **Rodar** `py -3.11 scratch/test_sam_plant.py`. Precisa de `BypassSandbox`, porque o `py` não abre no sandbox. Depois, analisar as diferenças anuais e horárias.
2. Ajustar o controlador até que TES charge/discharge, PC input e gross fiquem próximos do SAM. Conferir também o anticongelamento do TES (171 MWh). Se não fechar, verificar a hipótese do tanque quente e o modo de partida do ciclo a partir do TES.
3. Corrigir o protótipo do campo:
   - Incluir `TAU_ENV` em `q_rec_inc`.
   - Mover a escolha da vazão de partida para fora do loop de subpassos (linhas 839–843 de `test_sam_csp.py`).
   - Investigar a diferença na partida matinal e em `dE_int` (−38%).
4. Desempenho: 78 s por ano é inviável para o NSGA-II (66 combinações × população × gerações). Opções:
   - (a) numba/vetorizar `evac_receiver_sam`;
   - (b) tabela pré-calculada de perdas do receptor (T, q_inc, T_amb, vento);
   - (c) usar o modelo completo só na validação e um modelo reduzido, calibrado nele, no NSGA-II.
5. **Decisão sua:** baixar ou não os arquivos C++ do ciclo Rankine e das métricas do sistema, para remover as duas calibrações circulares.

### Passo 2: integrar em `geracao_energia_v2.py`
Sem alterações em relação ao `checkpoint.md`: `simular_receptor_csp`, `despachar_csp_tes` e `calcular_perda_tes_w`, mantendo assinaturas e unidades em W.

### Passo 3: atualizar `otimizacao_nsga_ii_v2.py` e validar
Sem alterações em relação ao `checkpoint.md`: `precalcular_perfis_unitarios`, `avaliar_configuracao_hibrida`, `executar_validacao_sam`, `exportar_auditoria_csp_sam` e a geração de `auditoria_csp_50mva.xlsx`.

---

## 4. Arquivos (nenhum arquivo do projeto foi modificado)
- `scratch/test_sam_csp.py`: campo solar (verificado, resultado na seção 2)
- `scratch/analyze_sam_excel.py`: decomposição das parasitas
- `scratch/test_sam_plant.py`: usina completa (**escrito, não executado**)
- Pasta `scratch/` = `C:\Users\victo\.gemini\antigravity\brain\c345f508-e215-480e-aeec-155215a9d525\scratch\`

