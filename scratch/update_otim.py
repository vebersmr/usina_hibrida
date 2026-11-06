with open('otimizacao_nsga_ii_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. CacheFisicoUnitario
target1 = "    scas_def: Optional[np.ndarray] = None\n"
rep1 = "    scas_def: Optional[np.ndarray] = None\n    q_piping_loss_1loop_mwt: Optional[np.ndarray] = None\n"
if target1 in content:
    content = content.replace(target1, rep1, 1)
    print("Replaced target1")
else:
    target1_crlf = "    scas_def: Optional[np.ndarray] = None\r\n"
    rep1_crlf = "    scas_def: Optional[np.ndarray] = None\r\n    q_piping_loss_1loop_mwt: Optional[np.ndarray] = None\r\n"
    content = content.replace(target1_crlf, rep1_crlf, 1)
    print("Replaced target1 CRLF")

# 2. precalcular_perfis_unitarios
target2 = '    scas_def_comp = diag_rec.get("scas_def", np.ones_like(p_rec_1loop))\n'
rep2 = '    scas_def_comp = diag_rec.get("scas_def", np.ones_like(p_rec_1loop))\n    q_piping_loss_1loop = (diag_rec["q_piping_loss"] / n_loops_base) if (diag_rec and "q_piping_loss" in diag_rec) else np.zeros_like(p_rec_1loop)\n'
if target2 in content:
    content = content.replace(target2, rep2, 1)
    print("Replaced target2")
else:
    target2_crlf = '    scas_def_comp = diag_rec.get("scas_def", np.ones_like(p_rec_1loop))\r\n'
    rep2_crlf = '    scas_def_comp = diag_rec.get("scas_def", np.ones_like(p_rec_1loop))\r\n    q_piping_loss_1loop = (diag_rec["q_piping_loss"] / n_loops_base) if (diag_rec and "q_piping_loss" in diag_rec) else np.zeros_like(p_rec_1loop)\r\n'
    content = content.replace(target2_crlf, rep2_crlf, 1)
    print("Replaced target2 CRLF")

target3 = '    scas_def_rep = np.tile(scas_def_comp, n_anos)[:n_horas]\n'
rep3 = '    scas_def_rep = np.tile(scas_def_comp, n_anos)[:n_horas]\n    q_piping_loss_1loop_mwt = np.tile(q_piping_loss_1loop, n_anos)[:n_horas]\n'
if target3 in content:
    content = content.replace(target3, rep3, 1)
    print("Replaced target3")
else:
    target3_crlf = '    scas_def_rep = np.tile(scas_def_comp, n_anos)[:n_horas]\r\n'
    rep3_crlf = '    scas_def_rep = np.tile(scas_def_comp, n_anos)[:n_horas]\r\n    q_piping_loss_1loop_mwt = np.tile(q_piping_loss_1loop, n_anos)[:n_horas]\r\n'
    content = content.replace(target3_crlf, rep3_crlf, 1)
    print("Replaced target3 CRLF")

target4 = '        scas_def=scas_def_rep,\n'
rep4 = '        scas_def=scas_def_rep,\n        q_piping_loss_1loop_mwt=q_piping_loss_1loop_mwt,\n'
if target4 in content:
    content = content.replace(target4, rep4, 1)
    print("Replaced target4")
else:
    target4_crlf = '        scas_def=scas_def_rep,\r\n'
    rep4_crlf = '        scas_def=scas_def_rep,\r\n        q_piping_loss_1loop_mwt=q_piping_loss_1loop_mwt,\r\n'
    content = content.replace(target4_crlf, rep4_crlf, 1)
    print("Replaced target4 CRLF")

# 3. exportar_auditoria_csp
target5 = '    q_rec_loss = cache.q_loss_1loop_mwt[:8760] * n_loops_escalado\n'
rep5 = '    q_rec_loss = cache.q_loss_1loop_mwt[:8760] * n_loops_escalado\n    q_piping_loss = cache.q_piping_loss_1loop_mwt[:8760] * n_loops_escalado if (hasattr(cache, "q_piping_loss_1loop_mwt") and cache.q_piping_loss_1loop_mwt is not None) else np.zeros(8760)\n'
if target5 in content:
    content = content.replace(target5, rep5, 1)
    print("Replaced target5")
else:
    target5_crlf = '    q_rec_loss = cache.q_loss_1loop_mwt[:8760] * n_loops_escalado\r\n'
    rep5_crlf = '    q_rec_loss = cache.q_loss_1loop_mwt[:8760] * n_loops_escalado\r\n    q_piping_loss = cache.q_piping_loss_1loop_mwt[:8760] * n_loops_escalado if (hasattr(cache, "q_piping_loss_1loop_mwt") and cache.q_piping_loss_1loop_mwt is not None) else np.zeros(8760)\r\n'
    content = content.replace(target5_crlf, rep5_crlf, 1)
    print("Replaced target5 CRLF")

target6 = '        "Field change in material/htf internal energy | (MWt)": e_int_energy,\n'
rep6 = '        "Field change in material/htf internal energy | (MWt)": e_int_energy,\n        "Field piping thermal losses | (MWt)": q_piping_loss,\n'
if target6 in content:
    content = content.replace(target6, rep6, 1)
    print("Replaced target6")
else:
    target6_crlf = '        "Field change in material/htf internal energy | (MWt)": e_int_energy,\r\n'
    rep6_crlf = '        "Field change in material/htf internal energy | (MWt)": e_int_energy,\r\n        "Field piping thermal losses | (MWt)": q_piping_loss,\r\n'
    content = content.replace(target6_crlf, rep6_crlf, 1)
    print("Replaced target6 CRLF")

target7 = '    e_int_campo_gwh = float(np.sum(e_int_energy) / 1000.0)\n'
rep7 = '    e_int_campo_gwh = float(np.sum(e_int_energy) / 1000.0)\n    e_pipe_loss_gwh = float(np.sum(q_piping_loss) / 1000.0)\n'
if target7 in content:
    content = content.replace(target7, rep7, 1)
    print("Replaced target7")
else:
    target7_crlf = '    e_int_campo_gwh = float(np.sum(e_int_energy) / 1000.0)\r\n'
    rep7_crlf = '    e_int_campo_gwh = float(np.sum(e_int_energy) / 1000.0)\r\n    e_pipe_loss_gwh = float(np.sum(q_piping_loss) / 1000.0)\r\n'
    content = content.replace(target7_crlf, rep7_crlf, 1)
    print("Replaced target7 CRLF")

target8 = '        {"Categoria": "Campo Solar",   "Grandeza": "Variação de Energia Interna no Campo", "Unidade": "GWht", "Valor": round(e_int_campo_gwh, 3)},\n'
rep8 = '        {"Categoria": "Campo Solar",   "Grandeza": "Variação de Energia Interna no Campo", "Unidade": "GWht", "Valor": round(e_int_campo_gwh, 3)},\n        {"Categoria": "Campo Solar",   "Grandeza": "Perdas Térmicas nas Tubulações do Campo", "Unidade": "GWht", "Valor": round(e_pipe_loss_gwh, 3)},\n'
if target8 in content:
    content = content.replace(target8, rep8, 1)
    print("Replaced target8")
else:
    target8_crlf = '        {"Categoria": "Campo Solar",   "Grandeza": "Variação de Energia Interna no Campo", "Unidade": "GWht", "Valor": round(e_int_campo_gwh, 3)},\r\n'
    rep8_crlf = '        {"Categoria": "Campo Solar",   "Grandeza": "Variação de Energia Interna no Campo", "Unidade": "GWht", "Valor": round(e_int_campo_gwh, 3)},\r\n        {"Categoria": "Campo Solar",   "Grandeza": "Perdas Térmicas nas Tubulações do Campo", "Unidade": "GWht", "Valor": round(e_pipe_loss_gwh, 3)},\r\n'
    content = content.replace(target8_crlf, rep8_crlf, 1)
    print("Replaced target8 CRLF")

with open('otimizacao_nsga_ii_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("otimizacao_nsga_ii_v2.py updated successfully!")

