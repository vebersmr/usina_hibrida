"""
Módulo para declarar as classes de modelagem da planta solar de CSP.

Este módulo contém as classes que representam os componentes principais do sistema de energia solar concentrada (CSP), 
incluindo coletores, receptores, fluidos de transferência de calor e o design do sistema.

:autor: Veber Samir Moreira Ribeiro
:data: 23-07-2025
:versão: 1.0.0
"""

import math

# Parâmetros dos coletores
class Collector:
	def __init__(self):
		# SkyFuel SkyTrough
		# Geometria dos coletores
		self.reflective_aperture_area = 656  # [m²]
		self.aperture_width = 6  # [m]
		self.length_collector_assembly = 115  # [m]
		self.modules_per_assembly = 8
		self.average_surface_to_focus_path_length = 2.15  # [m]
		self.piping_distance_between_assemblies = 1  # [m]
		self.L_tot = self.length_collector_assembly * self.modules_per_assembly # [m]

		# Parâmetros ópticos
		self.tracking_error = 0.988
		self.general_optical_error = 1
		self.geometry_effects = 0.952
		self.mirror_reflectance = 0.93
		self.dirt_on_mirror = 0.97
		# self.Col_opt_eff = self.tracking_error * self.general_optical_error * self.geometry_effects *self.mirror_reflectance * self.dirt_on_mirror

	@property
	def Col_opt_eff(self):
		# Agora, toda vez que você chamar Col_opt_eff, ele fará essa conta na hora,
		# pegando o valor mais recente de mirror_reflectance.
		return (self.tracking_error * 
				self.general_optical_error * 
				self.geometry_effects * 
				self.mirror_reflectance * 
				self.dirt_on_mirror)

# Parâmetros dos receptores
class Receiver:
	def __init__(self):
		# Schott PTR80
		# Geometria do receptor
		self.absorber_tube_inner_diameter = 0.076 # [m]
		self.absorber_tube_outer_diameter = 0.08 # [m]
		self.glass_envelope_inner_diameter = 0.115 # [m]
		self.glass_envelope_outer_diameter = 0.12 # [m]
		self.absorber_flow_plug_diameter = 0 # [m]
		self.internal_surface_roughness = 4.5e-05 # [-]
		self.absorber_flow_pattern = str = "Tube flow"
		self.absorber_material_type = str = "304L"

		# Parâmetros e Variações
		self.rec_type = 0 # Tipo de receptor utilizado nos SCAs
		self.field_frac = [1,0,0,0]
		# self.field_frac = [[0.985, 0.01, 0.005, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
		# Absorvedor:
		self.alpha_abs = [0.963, 0.963, 0.8, 0]
		# self.alpha_abs = [[0.963, 0.963, 0.8, 0], [0.963, 0.963, 0.8, 0], [0.963, 0.963, 0.8, 0], [0.963, 0.963, 0.8, 0]]
		self.epsilon_3_11 = [[0, 0.061], [50,0.062],[100, 0.064], [150, 0.0665], [200, 0.07], [250, 0.0745], [ 300, 0.08], [350, 0.0865], [400, 0.094], [450, 0.1025], [500, 0.112]]
		self.epsilon_3_12 = [[0.65]]
		self.epsilon_3_13 = [[0.65]]
		self.epsilon_3_14 = [[0]]
		self.epsilon_3_21 = [[100, 0.064], [150, 0.0665], [200, 0.07], [250, 0.0745], [ 300, 0.08], [350, 0.0865], [400, 0.094], [450, 0.1025], [500, 0.112]]
		self.epsilon_3_22 = [[0.65]]
		self.epsilon_3_23 = [[0.65]]
		self.epsilon_3_24 = [[0]]
		self.epsilon_3_31 = [[100, 0.064], [150, 0.0665], [200, 0.07], [250, 0.0745], [ 300, 0.08], [350, 0.0865], [400, 0.094], [450, 0.1025], [500, 0.112]]
		self.epsilon_3_32 = [[0.65]]
		self.epsilon_3_33 = [[0.65]]
		self.epsilon_3_34 = [[0]]
		self.epsilon_3_41 = [[100, 0.064], [150, 0.0665], [200, 0.07], [250, 0.0745], [ 300, 0.08], [350, 0.0865], [400, 0.094], [450, 0.1025], [500, 0.112]]
		self.epsilon_3_42 = [[0.65]]
		self.epsilon_3_43 = [[0.65]]
		self.epsilon_3_44 = [[0]]
		# Envelope:
		self.alpha_env = [0.02, 0.02, 0, 0]
		self.epsilon_env = [0.86, 0.86, 1, 0]
		self.tau_env = [0.964, 0.964, 1, 0]
		self.GlazingIntact = [1, 1, 0, 1]
		# Parâmetros do gás: (1 = ar; 26 = argônio; 27 = hidrogênio)
		self.AnnulusGas = [27, 1, 1, 27]
		self.P_a = [0.0001, 750, 750, 0]
		self.Design_loss = [190, 1270, 1500, 0]
		self.Shadowing = [0.935, 0.935, 0.935, 0.963]
		self.Dirt_HCE = [0.98, 0.98, 1, 0.98]


	def HCE_Heat_loss(self):
		HCE_heat_loss_des = 0
		# for i in range(len(self.field_frac[self.rec_type])):
		# 		HCE_heat_loss_des += self.field_frac[self.rec_type][i] * self.Design_loss[self.rec_type][i]
		for i in range(len(self.field_frac)):
			HCE_heat_loss_des += self.field_frac[i] * self.Design_loss[i]
		return HCE_heat_loss_des

	def HCE_optical_derate(self):
		HCE_optical_derate_des = 0
		
		# for i in range(len(self.field_frac[self.rec_type])):
		# 	HCE_optical_derate_des += self.field_frac[self.rec_type][i] * self.alpha_abs[self.rec_type][i] * self.Shadowing[self.rec_type][i] * self.Dirt_HCE[self.rec_type][i] * self.tau_env[self.rec_type][i]
		for i in range(len(self.field_frac)):
			HCE_optical_derate_des += self.field_frac[i] * self.alpha_abs[i] * self.Shadowing[i] * self.Dirt_HCE[i] * self.tau_env[i]
		return HCE_optical_derate_des
	

class Fluid:
	"""
	Fluido utilizado no Campo Solar: Therminol VP1

	"""
	def __init__(self,fluido):
		self.fluido = fluido # 0 = Therminol VP1; 1 = Hitec Solar Salt; 2 = Ar; 3 = Argônio (não utilizado); 4 = Hidrogênio
		
	def min_temp(self):  # [°C]
		if self.fluido == 0:
			return 12
		elif self.fluido == 1:
			return 238
		else: return
	
	def max_temp(self): # [°C]
		if self.fluido == 0:
			return 400
		elif self.fluido == 1:
			return 593
		else: return
		
	def Cp_ave(self,T_cold_K,T_hot_K):
		if T_cold_K > 0 and T_hot_K > 0:
			cp_sum = 0
			dT = (T_hot_K - T_cold_K) / 100
			T_i = T_cold_K + dT / 2 # meio-passo
			for i in range(100):
				cp_sum += self.Cp(T_i - 273.15)
				T_i += dT
			return cp_sum / 100
		else: return 0

	def Cp(self,T_C): # Calor específico [kJ/kg-K]
		T_K = T_C + 273.15
		if self.fluido == 0:
			return (1.509 + 0.002496 * T_C + 0.0000007888 * T_C*T_C)
		elif self.fluido == 1:
			return (1443. + 0.172 * T_C)/1000.0
		elif self.fluido == 2:
			return 1.03749 - 0.000305497*T_K + 7.49335E-07*T_K*T_K - 3.39363E-10*T_K*T_K*T_K
		elif self.fluido == 3:
			return 0.5203
		elif self.fluido == 4:
			return min(max(-45.4022 + 0.690156*T_K - 0.00327354*T_K*T_K + 0.00000817326*T_K*T_K*T_K - 1.13234e-08*T_K*T_K*T_K*T_K + 8.24995e-12*T_K*T_K*T_K*T_K*T_K - 2.46804e-15*T_K*T_K*T_K*T_K*T_K*T_K,11.3),14.7)
	
	def dens(self,T_C,P): # Densidade [kg/m³]
		T_K = T_C + 273.15
		if self.fluido == 0:
			return max(1074.0 - 0.6367 * T_C - 0.0007762 * T_C*T_C,400.0)
		elif self.fluido == 1:
			return max(2090.0 - 0.636 * T_C,1000.0)
		elif self.fluido == 2:
			return P/(287.0*T_K)
		elif self.fluido == 3:
			return max(P/(208.13*T_K),1e-10)
		elif self.fluido == 4:
			return max(P/(4124.0*T_K),1e-10)
	
	def visc(self,T_C): # Viscosidade [kg/m-s] ou [Pa-s]
		T_K = T_C + 273.15
		if self.fluido == 0:
			return 0.001 * (pow(10.,0.8703)*pow(max(T_C,20.),(0.2877 + math.log10(pow(max(T_C,20.),-0.3638)))))
		elif self.fluido == 1:
			return max((22.714 - 0.12 * T_C + 0.0002281 *T_C*T_C - 0.0000001474 * pow(T_C,3)) / 1000.0,1e-6)
		elif self.fluido == 2:
			return max(0.0000010765 + 7.15173e-08*T_K - 5.03525e-11*T_K*T_K + 2.02799e-14*T_K*T_K*T_K,1e-6)
		elif self.fluido == 3:
			return 4.4997e-6 + 6.38920e-08*T_K - 1.24550e-11*T_K*T_K
		elif self.fluido == 4:
			return 0.00000231 + 2.37842e-08*T_K - 5.73624e-12*T_K*T_K
	
	def cond(self,T_C): # Condutividade [W/m-K]
		T_K = T_C + 273.15
		if self.fluido == 0:
			return max(0.1381 - 0.00008708 * T_C - 0.0000001729 * pow(T_C,2),.001)
		elif self.fluido == 1:
			return 0.443 + 0.00019 * T_C
		elif self.fluido == 2:
			return max(0.00145453 + 0.0000872152*T_K - 2.20614e-08*T_K*T_K,1e-4)
		elif self.fluido == 3:
			return 0.00548 + 0.0000438969*T_K - 6.81410e-09*T_K*T_K
		elif self.fluido == 4:
			return max(0.0302888 + 0.00053634*T_K - 1.59604e-07*T_K*T_K,.01)
	
	def enth(self,T_C): # Entalpia [J/kg]
		if self.fluido == 0:
			return 1000 * (-18.34 + 1.498*T_C + 0.001377*T_C*T_C)
		elif self.fluido == 1:
			return 1443*T_C + 0.086*T_C*T_C
		else: return

	def temp(self,H): # Temperatura [K]
		if self.fluido == 0:
			return 7.4333E-17*pow(H,3) - 0.00000000024625*pow(H,2) + 0.00063282 * H + 12.403
		elif self.fluido == 1:
			return -0.0000000000262*H*H + 0.0006923 * H + 0.03058
		else: return
		
	def Cv(self,T_K):
		if self.fluido == 2:
			return 0.750466 - 0.000305497*T_K + 7.49335E-07*T_K*T_K - 3.39363E-10*pow(T_K,3)
		elif self.fluido == 3:
			return 0.3122
		elif self.fluido == 4:
			return min(max(-49.5264 + 0.690156*T_K - 0.00327354*T_K*T_K + 0.00000817326*pow(T_K,3) - 1.13234e-08*pow(T_K,4) + 8.24995e-12*pow(T_K,5) - 2.46804e-15*pow(T_K,6),7.20),10.60)
		else: return

class HTFState:
	def __init__(self, temp, pres, m_dot):
		self.temp = temp
		self.pres = pres
		self.m_dot = m_dot

# Tela Design do Sistema
class SystemDesign:
	  
	def __init__(self):
		self.option = 0 # 0 = escolhe o múltiplo solar; 1 = escolhe a área do campo solar
		# Campo solar
		self.solar_multiple = 1.98 # [-]
		self.field_aperture = 949888 # [m²]
		self.design_point_dni = 950 # [W/m²]
		self.loop_inlet_htf_temperature = 293 # [°C]
		self.loop_outlet_htf_temperature = 391 # [°C]
		
		self.rec = Receiver()
		self.col = Collector()
		# self.field_htf = Fluid()
		
		# Ciclo de potência
		self.design_turbine_gross_output = 111 # [MWe]
		self.estimated_gross_to_net_conversion_factor = 0.9 # [-]
		self.cycle_thermal_efficiency = 0.356 # [-]
		# Armazenamento térmico
		self.hours_of_storage_at_design_point = 6 # [h]
		
		# Cálculo do fluxo de massa do campo solar
		# Calor específico [J/kg-K]:
		# self.c_htf_ave = self.field_htf.Cp_ave(self.loop_inlet_htf_temperature,self.loop_inlet_htf_temperature) * 1e3

	def __str__(self):
		return self.__dict__
	
	# def q_loop_ideal(self):
	# 	return self.col.single_loop_aperture * self.design_point_dni # [W]

	def cycle_thermal_power(self):
		return self.design_turbine_gross_output / self.cycle_thermal_efficiency  # [MWt]
	
	# def number_of_loops(self) -> int:
	# 	return 181 # era 168

	def field_thermal_power(self):
		return self.solar_multiple * self.cycle_thermal_power()  # [MWt]

	def estimated_net_output_at_design(self):
		return self.design_turbine_gross_output * self.estimated_gross_to_net_conversion_factor  # [MWe]

# Tela Campo Solar
class SolarField:

	def __init__(self):
		# Parâmetros do campo solar
		self.row_spacing = 15 # [m]
		self.header_pipe_roughness = 4.57e-05 # [m]
		self.HTF_pump_efficiency = 0.85
		self.piping_thermal_loss_coefficient = 0.45 # [W/m²-K]
		self.wind_stow_speed = 25 # [m/s]
		self.receiver_startup_delay_time = 0.2 # [h]
		self.receiver_startup_delay_energy_fraction = 0.25
		self.collector_startup_energy = 0.021 # [kWhe/sca]
		self.tracking_power_per_SCA = 125 # [W/sca]
		self.number_of_field_subsections = 2
		self.allow_partial_defocusing = str = "Simultaneous"
		# HTF
		self.field_htf_fluid = str = "Therminol VP-1"
		# self.field_htf_min_op_temp = 12 # [°C]
		# self.field_htf_max_op_temp = 400 # [°C]
		self.freeze_protection_temp = 150 # [°C]
		self.min_single_loop_flow_rate = 1 # [kg/s]
		self.max_single_loop_flow_rate = 12 # [kg/s]
		# self.min_field_flow_vel = 0.3 # [m/s]
		# self.max_field_flow_vel = 3.7 # [m/s]
		self.cold_hdr_des_min_flow_vel = 2 # [m/s]
		self.cold_hdr_des_max_flow_vel = 2 # [m/s]
		self.hot_hdr_des_min_flow_vel = 3 # [m/s]
		self.hot_hdr_des_max_flow_vel = 3 # [m/s]
		# Orientação dos coletores
		self.col_tilt = 0 # [°]
		self.col_azi = 0 # [°]
		self.stow_angle = 170 # [°]
		self.deploy_angle = 10 # [°]
		# Lavagem dos espelhos
		self.water_usage = 0.7 # [L/m²,aper]
		self.washes_per_year = 63
		# Capacidade de calor 
		self.hot_piping_thermal_inertia = 0.2 # [kWht/K-MWt]
		self.cold_piping_thermal_inertia = 0.2 # [kWht/K-MWt]
		self.field_loop_piping_thermal_inertia = 4.5 # [kWht/K-MWt]
		self.nSCA = 8
		self.col = Collector()
		self.rec = Receiver()
		self.system = SystemDesign()
		self.field_htf = Fluid(0)
		self.annulus_gas = Fluid(4)


	def single_loop_aperture(self): # [m²]
		return self.col.reflective_aperture_area * self.col.modules_per_assembly

	def loop_optical_eta(self):
		return self.col.Col_opt_eff * self.rec.HCE_optical_derate()
	
	def loss_tot(self):
		loss_col = self.col.length_collector_assembly * self.rec.HCE_Heat_loss()
		return loss_col * self.nSCA
		
	def total_loop_conversion_eta(self):
		q_loop_actual = self.system.design_point_dni * self.single_loop_aperture() * self.loop_optical_eta() - self.loss_tot()
		return q_loop_actual / self.system.design_point_dni / self.single_loop_aperture()
	
	def required_total_aperture_SM1(self):
		q_pb_design = 1e6 * self.system.design_turbine_gross_output / self.system.cycle_thermal_efficiency
		return q_pb_design / self.system.design_point_dni / self.total_loop_conversion_eta()
	
	def required_nLoops_SM1(self):
		return math.ceil(self.required_total_aperture_SM1 / self.single_loop_aperture)
	
	def actual_nLoops(self):
		if (self.system.option == 0):
			ap = self.system.solar_multiple * self.required_total_aperture_SM1()
			return math.ceil(ap / self.single_loop_aperture())
		else:
			ap = self.system.field_aperture
			return math.ceil(ap / self.single_loop_aperture())
		
	def actual_aperture(self):
		return self.actual_nLoops() * self.single_loop_aperture()
	
	def actual_SM(self):
		return self.actual_aperture() / self.required_total_aperture_SM1()
	
	def actual_field_thermal_output(self):
		return 0

	def total_tracking_power(self):
		return self.tracking_power_per_SCA * self.nSCA * self.actual_nLoops()

	def ideal_field_thermal_power(self):
		return (self.actual_aperture() *
		  		self.system.design_point_dni *
				self.loop_optical_eta() -
				self.loss_tot() * self.actual_nLoops())
  
	def m_dot_design(self):
		self.Tin = self.system.loop_inlet_htf_temperature
		self.Tout = self.system.loop_outlet_htf_temperature
		self.c_htf_ave = self.field_htf.Cp_ave(self.Tin,self.Tout) * 1e3
		self.m_dot_min = self.min_single_loop_flow_rate * self.actual_nLoops()
		self.m_dot_max = self.max_single_loop_flow_rate * self.actual_nLoops()
		self.m_dot = self.ideal_field_thermal_power() / (self.c_htf_ave * (self.Tout - self.Tin))
		return max(min(self.m_dot,self.m_dot_max),self.m_dot_min)
				
	def actual_field_thermal_power(self):
		return self.m_dot_design() * self.c_htf_ave * (self.Tout - self.Tin)
	

