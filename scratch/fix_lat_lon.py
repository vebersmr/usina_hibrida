with open('otimizacao_nsga_ii_v2.py', 'rb') as f:
    raw = f.read().decode('utf-8')

is_crlf = '\r\n' in raw
content = raw.replace('\r\n', '\n')

# 1. Add latitude, longitude, tz to CacheFisicoUnitario
old_dc = """    n_loops_base: int = 91
    p_csp_mva_base: float = 50.0"""

new_dc = """    n_loops_base: int = 91
    p_csp_mva_base: float = 50.0
    latitude: float = -13.27
    longitude: float = -43.42
    tz: int = -3"""

assert old_dc in content, "old_dc not found"
content = content.replace(old_dc, new_dc, 1)

# 2. Pass latitude, longitude, tz to CacheFisicoUnitario in precalcular_perfis_unitarios
old_ret = """        n_loops_base=n_base,
        p_csp_mva_base=p_base_mva,
    )"""

new_ret = """        n_loops_base=n_base,
        p_csp_mva_base=p_base_mva,
        latitude=lat,
        longitude=lon,
        tz=tz,
    )"""

assert old_ret in content, "old_ret not found"
content = content.replace(old_ret, new_ret, 1)

# 3. Pass latitude, longitude, tz to simular_receptor_csp in exportar_auditoria_csp
old_sim_call = """        res_exato, _, _ = simular_receptor_csp(
            cenario_clima,
            n_loops_sim=n_loops_escalado,
            p_ref_mw=p_ref_gross,
        )"""

new_sim_call = """        lat_c = getattr(cache, "latitude", -13.27)
        lon_c = getattr(cache, "longitude", -43.42)
        tz_c = getattr(cache, "tz", -3)
        res_exato, _, _ = simular_receptor_csp(
            cenario_clima,
            n_loops_sim=n_loops_escalado,
            p_ref_mw=p_ref_gross,
            latitude=lat_c,
            longitude=lon_c,
            tz=tz_c,
        )"""

assert old_sim_call in content, "old_sim_call not found"
content = content.replace(old_sim_call, new_sim_call, 1)

if is_crlf:
    content = content.replace('\n', '\r\n')

with open('otimizacao_nsga_ii_v2.py', 'wb') as f:
    f.write(content.encode('utf-8'))

print("otimizacao_nsga_ii_v2.py atualizado com sucesso!")

