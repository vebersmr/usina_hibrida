# -*- coding: utf-8 -*-
"""
Simulação Isolada de Geração de Energia Eólica (1 Ano de Operação).
Código unificado fiel à lógica original de geracao_eolica.py, lib_windwatts.py e lib_windwakemodel.py.
"""

from abc import ABC, abstractmethod
import json
import math
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.special as sps


# ==============================================================================
# 1. CLASSES FÍSICAS E MODELOS DE ESTEIRA (lib_windwakemodel.py)
# ==============================================================================

class Physics:
    PI = math.pi
    AIR_DENSITY_SEA_LEVEL = 1.225
    R_GAS_DRY_AIR = 287.058
    Pa_PER_Atm = 101325.0

    @staticmethod
    def celcius_to_kelvin(temp_c: float) -> float:
        return temp_c + 273.15


class WakeModelOptions:
    SIMPLE = 0
    PARK = 1
    EDDYVISCOSITY = 2
    CONSTANTVALUE = 3


class WindTurbine:
    """Armazena as características da turbina e a curva de potência."""

    def __init__(self):
        self.shear_exponent: float = -999
        self.measurement_height: float = -999
        self.hub_height: float = -999
        self.rotor_diameter: float = -999

        self.ct_curve = np.array([])
        self.power_curve_ws = np.array([])
        self.power_curve_kw = np.array([])
        self.density_corrected_ws = np.array([])
        self.power_curve_rpm = np.array([])

        self.cut_in_speed: float = 0.0
        self.previous_air_density: float = Physics.AIR_DENSITY_SEA_LEVEL
        self.err_details: str = ""

    @property
    def power_curve_array_length(self) -> int:
        return len(self.power_curve_ws)

    def is_initialized(self) -> bool:
        return all([
            self.shear_exponent != -999,
            self.measurement_height != -999,
            self.hub_height != -999,
            self.rotor_diameter != -999,
            self.power_curve_array_length > 0,
        ])

    def set_power_curve(self, wind_speeds: list, power_output: list) -> bool:
        if len(wind_speeds) != len(power_output):
            self.err_details = "Os tamanhos dos arrays da curva de potência da turbina são diferentes."
            return False

        self.power_curve_ws = np.array(wind_speeds)
        self.power_curve_kw = np.array(power_output)
        self.density_corrected_ws = np.copy(self.power_curve_ws)
        self.power_curve_rpm = np.full(self.power_curve_array_length, -1.0)
        return True

    def set_ct_curve(self, ct_curve: list) -> bool:
        if len(ct_curve) != self.power_curve_array_length:
            self.err_details = "O comprimento da curva Ct deve ser igual ao da curva de potência."
            return False
        self.ct_curve = np.array(ct_curve)
        return True

    def calculate_eff(self, reduced_power: float, original_power: float) -> float:
        if original_power < 0.0:
            return 0.0
        return 100.0 * (reduced_power + 0.0001) / (original_power + 0.0001)

    def turbine_power(self, wind_velocity: float, air_density: float) -> Tuple[float, float, float]:
        if not self.is_initialized():
            self.err_details = "WindTurbine não inicializada com os dados necessários."
            return 0.0, 0.0, 0.0

        if abs(air_density - self.previous_air_density) > 0.001:
            correction = (Physics.AIR_DENSITY_SEA_LEVEL / air_density) ** (1.0 / 3.0)
            self.density_corrected_ws = self.power_curve_ws * correction
            self.previous_air_density = air_density

        try:
            first_power_idx = np.where(self.power_curve_kw > 0)[0][0]
            cut_in_idx = max(0, first_power_idx - 1)
            self.cut_in_speed = self.density_corrected_ws[cut_in_idx]
        except IndexError:
            self.cut_in_speed = float('inf')

        out_pwr = 0.0
        if self.density_corrected_ws[0] < wind_velocity < self.density_corrected_ws[-1]:
            out_pwr = np.interp(wind_velocity, self.density_corrected_ws, self.power_curve_kw)
        elif wind_velocity == self.density_corrected_ws[-1]:
            out_pwr = self.power_curve_kw[-1]

        if wind_velocity < self.cut_in_speed:
            out_pwr = 0.0

        if out_pwr > 0:
            turbine_output = out_pwr
            turbine_gross = out_pwr

            pden = 0.5 * air_density * (wind_velocity ** 3.0)
            area = Physics.PI / 4.0 * self.rotor_diameter**2
            power_coeff = max(0.0, 1000.0 * out_pwr / (pden * area))

            if power_coeff >= 0.0:
                thrust_coeff = max(
                    0.0,
                    -1.453989e-2
                    + 1.473506 * power_coeff
                    - 2.330823 * power_coeff**2
                    + 3.885123 * power_coeff**3,
                )
            else:
                thrust_coeff = 0.0

            return turbine_output, turbine_gross, thrust_coeff

        return 0.0, 0.0, 0.0


class WakeModelBase(ABC):
    def __init__(self):
        self.n_turbines: int = 0
        self.w_turbine: WindTurbine = None
        self.err_details: str = ""

    def get_model_name(self) -> str:
        return ""

    @abstractmethod
    def wake_calculations(
        self, air_density: float, distance_downwind: np.ndarray,
        distance_crosswind: np.ndarray, power: np.ndarray,
        eff: np.ndarray, thrust: np.ndarray, wind_speed: np.ndarray,
        turbulence_intensity: np.ndarray
    ) -> None:
        pass


class SimpleWakeModel(WakeModelBase):
    """Modelo de esteira simples de Pat Quinlan (PQ)."""

    def __init__(self, num_turbines: int, wt: WindTurbine):
        super().__init__()
        self.n_turbines = num_turbines
        self.w_turbine = wt

    def get_model_name(self) -> str:
        return "PQ"

    def _vel_delta_pq(
        self, radii_crosswind: float, axial_dist_in_radii: float,
        thrust_coeff: float, current_ti: float
    ) -> Tuple[float, float]:
        if (
            radii_crosswind > 20.0
            or current_ti <= 0.0
            or axial_dist_in_radii <= 0.0
            or thrust_coeff <= 0.0
        ):
            return 0.0, current_ti

        added_turbulence = (thrust_coeff / 7.0) * (1.0 - (2.0 / 5.0) * math.log(2.0 * axial_dist_in_radii))
        new_ti = math.sqrt(added_turbulence**2 + current_ti**2)

        aa = (new_ti**2) * (axial_dist_in_radii**2)
        f_exp = max(-99.0, -(radii_crosswind**2) / (2.0 * aa))
        velocity_deficit = (thrust_coeff / (4.0 * aa)) * math.exp(f_exp)

        final_deficit = max(0.0, min(1.0, velocity_deficit))
        return final_deficit, new_ti

    def wake_calculations(
        self, air_density, distance_downwind, distance_crosswind,
        power, eff, thrust, wind_speed, turbulence_intensity
    ):
        for i in range(1, self.n_turbines):
            total_deficit_factor = 1.0
            for j in range(i):
                f_distance_downwind = abs(distance_downwind[j] - distance_downwind[i])
                f_distance_crosswind = abs(distance_crosswind[j] - distance_crosswind[i])

                vdef, new_ti = self._vel_delta_pq(
                    f_distance_crosswind, f_distance_downwind, thrust[j], turbulence_intensity[i]
                )
                turbulence_intensity[i] = new_ti
                total_deficit_factor *= (1.0 - vdef)

            wind_speed[i] *= total_deficit_factor
            power[i], _, thrust[i] = self.w_turbine.turbine_power(wind_speed[i], air_density)

            if self.w_turbine.err_details:
                self.err_details = self.w_turbine.err_details
                return

            eff[i] = self.w_turbine.calculate_eff(power[i], power[0])

        eff[0] = 100.0


class ParkWakeModel(WakeModelBase):
    """Modelo de esteira de Park/WAsP."""

    def __init__(self, num_turbines: int, wt: WindTurbine):
        super().__init__()
        self.n_turbines = num_turbines
        self.w_turbine = wt
        self.wake_decay_coefficient = 0.07
        self.min_thrust_coeff = 0.02

    def get_model_name(self) -> str:
        return "Park"

    def _circle_overlap(self, dist_center_to_center: float, rad1: float, rad2: float) -> float:
        d = dist_center_to_center
        if d < 0 or rad1 < 0 or rad2 < 0 or d >= rad1 + rad2:
            return 0.0
        if rad1 >= d + rad2:
            return Physics.PI * rad2**2
        if rad2 >= d + rad1:
            return Physics.PI * rad1**2

        t1 = rad1**2 * math.acos((d**2 + rad1**2 - rad2**2) / (2 * d * rad1))
        t2 = rad2**2 * math.acos((d**2 + rad2**2 - rad1**2) / (2 * d * rad2))
        t3 = 0.5 * math.sqrt((-d + rad1 + rad2) * (d + rad1 - rad2) * (d - rad1 + rad2) * (d + rad1 + rad2))
        return t1 + t2 - t3

    def _delta_v_park(
        self, u_freestream: float, u_upwind: float, dist_crosswind: float,
        dist_downwind: float, radius_upstream: float, radius_downstream: float,
        thrust_coeff: float
    ) -> float:
        ct = max(self.min_thrust_coeff, min(0.999, thrust_coeff))
        k = self.wake_decay_coefficient

        radius_of_wake = radius_upstream + (k * dist_downwind)
        area_overlap = self._circle_overlap(dist_crosswind, radius_downstream, radius_of_wake)

        if area_overlap <= 0.0:
            return u_freestream

        d_def = (
            (1 - math.sqrt(1 - ct))
            * (radius_upstream / radius_of_wake) ** 2
            * (area_overlap / (Physics.PI * radius_downstream**2))
        )
        return u_upwind * (1.0 - d_def)

    def wake_calculations(
        self, air_density, distance_downwind, distance_crosswind,
        power, eff, thrust, wind_speed, turbulence_intensity
    ):
        turbine_radius = self.w_turbine.rotor_diameter / 2.0

        for i in range(1, self.n_turbines):
            new_speed = wind_speed[0]
            for j in range(i):
                dist_downwind_m = turbine_radius * abs(distance_downwind[i] - distance_downwind[j])
                dist_crosswind_m = turbine_radius * abs(distance_crosswind[i] - distance_crosswind[j])

                speed_from_j_wake = self._delta_v_park(
                    wind_speed[0], wind_speed[j], dist_crosswind_m, dist_downwind_m,
                    turbine_radius, turbine_radius, thrust[j]
                )
                new_speed = min(new_speed, speed_from_j_wake)

            wind_speed[i] = new_speed
            power[i], _, thrust[i] = self.w_turbine.turbine_power(wind_speed[i], air_density)

            if self.w_turbine.err_details:
                self.err_details = self.w_turbine.err_details
                return

            eff[i] = self.w_turbine.calculate_eff(power[i], power[0])

        eff[0] = 100.0


class ConstantWakeModel(WakeModelBase):
    """Modelo que aplica um derate constante na potência."""

    def __init__(self, num_turbines: int, wt: WindTurbine, derate_multiplier: float):
        super().__init__()
        self.n_turbines = num_turbines
        self.w_turbine = wt
        self.derate = derate_multiplier

    def get_model_name(self) -> str:
        return "Constant"

    def wake_calculations(
        self, air_density, distance_downwind, distance_crosswind,
        power, eff, thrust, wind_speed, turbulence_intensity
    ):
        turb_power, _, turb_thrust = self.w_turbine.turbine_power(wind_speed[0], air_density)

        if self.w_turbine.err_details:
            self.err_details = self.w_turbine.err_details
            return

        turb_power *= self.derate
        for i in range(self.n_turbines):
            power[i] = turb_power
            thrust[i] = turb_thrust
            eff[i] = 100.0


# ==============================================================================
# 2. CALCULADORA DE POTÊNCIA (lib_windwatts.py)
# ==============================================================================

class WindPowerCalculator:
    def __init__(self):
        self.wind_turb: WindTurbine = None
        self.n_turbines: int = 0
        self.turbulence_intensity: float = 0.0
        self.x_coords: np.ndarray = np.array([])
        self.y_coords: np.ndarray = np.array([])

        self.wake_model: WakeModelBase = None
        self.err_details: str = ""
        self.MAX_WIND_TURBINES = 300

    def initialize_model(self, selected_wake_model: WakeModelBase) -> bool:
        if selected_wake_model:
            self.wake_model = selected_wake_model
            return True
        return False

    def _gammaln(self, x: float) -> float:
        return sps.gammaln(x)

    def _coord_trans(
        self, meters_north: float, meters_east: float, wind_dir_degrees: float
    ) -> Tuple[float, float]:
        wind_dir_adjusted = wind_dir_degrees + 90.0
        wind_dir_rad = math.radians(wind_dir_adjusted)

        cos_rad = math.cos(wind_dir_rad)
        sin_rad = math.sin(wind_dir_rad)

        meters_downwind = meters_east * cos_rad - meters_north * sin_rad
        meters_crosswind = meters_east * sin_rad + meters_north * cos_rad

        return meters_downwind, meters_crosswind

    def wind_power_using_resource(
        self, wind_speed: float, wind_dir_deg: float,
        air_pressure_atm: float, temp_dry_c: float,
        power: np.ndarray, thrust: np.ndarray, eff: np.ndarray,
        ad_wind_speed: np.ndarray, ti: np.ndarray,
        distance_downwind: np.ndarray, distance_crosswind: np.ndarray
    ) -> Tuple[float, float]:
        if not self.wake_model:
            self.err_details = "Modelo de esteira não inicializado."
            return 0.0, 0.0

        if not (1 <= self.n_turbines <= self.MAX_WIND_TURBINES):
            self.err_details = "O número de turbinas está fora do intervalo permitido."
            return 0.0, 0.0

        air_pressure_pa = (
            air_pressure_atm * Physics.Pa_PER_Atm
            if 0.5 < air_pressure_atm < 1.1
            else air_pressure_atm
        )
        air_density = air_pressure_pa / (Physics.R_GAS_DRY_AIR * Physics.celcius_to_kelvin(temp_dry_c))

        f_turbine_output, f_turbine_gross, f_thrust_coeff = self.wind_turb.turbine_power(
            wind_speed, air_density
        )

        if self.wind_turb.err_details:
            self.err_details = self.wind_turb.err_details
            return 0.0, 0.0

        farm_power_gross = f_turbine_gross * self.n_turbines

        power.fill(0.0)
        thrust.fill(0.0)
        eff.fill(0.0)
        ad_wind_speed.fill(wind_speed)
        ti.fill(self.turbulence_intensity)

        if self.n_turbines < 2:
            return f_turbine_output, farm_power_gross

        if f_turbine_output <= 0.0:
            return 0.0, farm_power_gross

        if self.wake_model.get_model_name() == "Constant":
            self.wake_model.wake_calculations(
                air_density, None, None, power, eff, thrust, ad_wind_speed, ti
            )
            return power[0] * self.n_turbines, farm_power_gross

        for i in range(self.n_turbines):
            distance_downwind[i], distance_crosswind[i] = self._coord_trans(
                self.y_coords[i], self.x_coords[i], wind_dir_deg
            )

        distance_downwind -= np.min(distance_downwind)
        distance_crosswind -= np.min(distance_crosswind)

        rotor_radii = self.wind_turb.rotor_diameter / 2.0
        distance_downwind /= rotor_radii
        distance_crosswind /= rotor_radii

        wt_id = np.arange(self.n_turbines)
        sort_indices = np.argsort(distance_downwind)

        sorted_downwind = distance_downwind[sort_indices]
        sorted_crosswind = distance_crosswind[sort_indices]
        sorted_wt_id = wt_id[sort_indices]

        sorted_power = np.zeros(self.n_turbines)
        sorted_thrust = np.zeros(self.n_turbines)
        sorted_eff = np.zeros(self.n_turbines)
        sorted_ws = np.full(self.n_turbines, wind_speed)
        sorted_ti = np.full(self.n_turbines, self.turbulence_intensity)

        sorted_power[0] = f_turbine_output
        sorted_thrust[0] = f_thrust_coeff
        sorted_eff[0] = 100.0 if f_turbine_output >= 1.0 else 0.0

        self.wake_model.wake_calculations(
            air_density, sorted_downwind, sorted_crosswind,
            sorted_power, sorted_eff, sorted_thrust, sorted_ws, sorted_ti
        )
        if self.wake_model.err_details:
            self.err_details = self.wake_model.err_details
            return 0.0, 0.0

        farm_power = np.sum(sorted_power)

        unsort_indices = np.argsort(sorted_wt_id)
        power[:] = sorted_power[unsort_indices]
        thrust[:] = sorted_thrust[unsort_indices]
        eff[:] = sorted_eff[unsort_indices]
        ad_wind_speed[:] = sorted_ws[unsort_indices]
        ti[:] = sorted_ti[unsort_indices]
        distance_downwind[:] = sorted_downwind[unsort_indices] * rotor_radii
        distance_crosswind[:] = sorted_crosswind[unsort_indices] * rotor_radii

        return farm_power, farm_power_gross

    def wind_power_using_weibull(
        self, weibull_k: float, avg_speed: float, ref_height: float
    ) -> Tuple[float, np.ndarray]:
        hub_ht_windspeed = avg_speed * (
            (self.wind_turb.hub_height / ref_height) ** self.wind_turb.shear_exponent
        )
        denom = math.exp(self._gammaln(1 + (1 / weibull_k)))
        lambda_ = hub_ht_windspeed / denom

        ws_bins = self.wind_turb.power_curve_ws
        ws_shifted = ws_bins + 0.125
        weibull_cumulative = 1.0 - np.exp(-((ws_shifted / lambda_) ** weibull_k))
        weibull_bin_prob = np.diff(weibull_cumulative, prepend=0)

        energy_turbine_by_bin = (8760.0 * weibull_bin_prob) * self.wind_turb.power_curve_kw
        total_energy_turbine = np.sum(energy_turbine_by_bin)
        return total_energy_turbine, energy_turbine_by_bin


# ==============================================================================
# 3. LEITURA E SIMULAÇÃO DO PARQUE EÓLICO (geracao_eolica.py)
# ==============================================================================

class WindData:
    """Lê dados de vento de uma estrutura de dicionário em memória."""

    def __init__(self, data_table: dict):
        self._irecord = 0
        self._heights = np.array(data_table.get("heights", []))
        self._fields = np.array(data_table.get("fields", []))
        self._data = np.array(data_table.get("data", [[]]))

    def n_records(self) -> int:
        return self._data.shape[0]

    def read(
        self, hub_height, wind_speed, wind_dir, temp, pres,
        meas_height, closest_dir_meas_ht, interpolate
    ) -> bool:
        if self._irecord >= self.n_records():
            return False

        idx_speed = np.argmin(np.abs(self._heights - hub_height))
        idx_dir = idx_speed

        row_data = self._data[self._irecord]
        wind_speed[0] = row_data[self._fields == 3][0]
        wind_dir[0] = row_data[self._fields == 4][0]
        temp[0] = row_data[self._fields == 1][0]
        pres[0] = row_data[self._fields == 2][0]
        meas_height[0] = self._heights[idx_speed]
        closest_dir_meas_ht[0] = self._heights[idx_dir]

        self._irecord += 1
        return True


class AdjustmentFactors:
    def __init__(self, inputs, prefix):
        self.factors = np.ones(8760)

    def setup(self, n_steps=8760) -> bool:
        if len(self.factors) != n_steps:
            self.factors = np.ones(n_steps)
        return True

    def __call__(self, timestep: int) -> float:
        return self.factors[timestep % len(self.factors)]


def _get_fixed_losses(inputs: dict) -> float:
    loss_multiplier = 1.0
    loss_names = [
        "avail_bop_loss", "avail_grid_loss", "avail_turb_loss", "elec_eff_loss",
        "elec_parasitic_loss", "env_degrad_loss", "env_exposure_loss", "env_env_loss",
        "env_icing_loss", "ops_env_loss", "ops_grid_loss", "ops_load_loss",
        "ops_strategies_loss", "turb_generic_loss", "turb_hysteresis_loss",
        "turb_perf_loss", "turb_specific_loss", "wake_ext_loss", "wake_future_loss",
    ]
    for loss in loss_names:
        if loss in inputs:
            loss_multiplier *= (1.0 - inputs[loss] / 100.0)
    return loss_multiplier


def _calculate_losses(outputs: dict, annual_wake_int_loss_percent: float, inputs: dict):
    avail_loss = 1.0 - (
        (100.0 - inputs.get("avail_bop_loss", 0)) / 100.0
        * (100.0 - inputs.get("avail_grid_loss", 0)) / 100.0
        * (100.0 - inputs.get("avail_turb_loss", 0)) / 100.0
    )
    total_wake_loss = 1.0 - (
        (100.0 - inputs.get("wake_ext_loss", 0)) / 100.0
        * (100.0 - inputs.get("wake_future_loss", 0)) / 100.0
        * (100.0 - annual_wake_int_loss_percent) / 100.0
    )
    outputs["avail_losses"] = avail_loss * 100.0
    outputs["annual_wake_loss_total_percent"] = total_wake_loss * 100.0


def tratar_dados_eolica(weather_file: str, hub_height: float) -> dict:
    """
    Executa exatamente a mesma transformação que o bloco 'else' original
    de tratar_dados_eolica em geracao_eolica.py.
    """
    df = pd.read_csv(weather_file, skiprows=2)

    temp = df['Temperature'].values                              # [°C]
    wind_speed10 = df['Wind Speed'].values                       # [m/s] a 10m
    wind_speed_hub = wind_speed10 * (hub_height / 10) ** 0.14    # [m/s] Wang et al. (2022)
    wind_dir = df['Wind Direction'].values                       # [°]
    pressure = df['Pressure'].values / 1013.25                   # [mbar] para [atm]

    data = np.column_stack((wind_speed_hub, wind_dir, temp, pressure))

    return {
        "heights": [hub_height],
        "fields": [3, 4, 1, 2],
        "data": data,
    }


def simular_eolica(weather_input: str, parametros_json: str) -> dict:
    """Simula a geração anual do parque eólico a partir do CSV climático e do JSON do SAM."""
    outputs = {}

    with open(parametros_json, "r", encoding="utf-8") as file:
        inputs = json.load(file)

    weather_inputs = tratar_dados_eolica(weather_input, inputs["wind_turbine_hub_ht"])
    inputs["wind_resource_data"] = weather_inputs

    # 1. Configurar WindTurbine
    wt = WindTurbine()
    wt.shear_exponent = inputs["wind_resource_shear"]
    wt.hub_height = inputs["wind_turbine_hub_ht"]
    wt.measurement_height = wt.hub_height
    wt.rotor_diameter = inputs["wind_turbine_rotor_diameter"]

    pc_w = inputs["wind_turbine_powercurve_windspeeds"]
    pc_p = inputs["wind_turbine_powercurve_powerout"]
    wt.set_power_curve(list(pc_w), list(pc_p))

    if "wind_turbine_ct_curve" in inputs:
        wt.set_ct_curve(list(inputs["wind_turbine_ct_curve"]))

    # 2. Configurar WindPowerCalculator
    wpc = WindPowerCalculator()
    wpc.wind_turb = wt
    ti_coeff = inputs["wind_resource_turbulence_coeff"]
    # Garante que se o JSON tiver TI em % (ex: 10.0), seja convertido para fração (0.10)
    wpc.turbulence_intensity = ti_coeff / 100.0 if ti_coeff > 1.0 else ti_coeff
    wpc.x_coords = np.array(inputs["wind_farm_xCoordinates"])
    wpc.y_coords = np.array(inputs["wind_farm_yCoordinates"])
    wpc.n_turbines = len(wpc.x_coords)

    if wpc.n_turbines < 1:
        raise ValueError("O número de turbinas deve ser pelo menos 1.")

    # 3. Configurar Perdas
    haf = AdjustmentFactors(inputs, "adjust")
    loss_multiplier = _get_fixed_losses(inputs)
    annual_wake_int_loss_percent = 0.0

    # 4. Executar o Modelo de Recurso Eólico Escolhido
    model_choice = inputs["wind_resource_model_choice"]

    if model_choice == 1:
        total_e, by_bin = wpc.wind_power_using_weibull(
            inputs["weibull_k_factor"],
            inputs["weibull_wind_speed"],
            inputs["weibull_reference_height"],
        )
        outputs["turbine_output_by_windspeed_bin"] = by_bin
        gross_energy = total_e * wpc.n_turbines

        annual_wake_int_loss_percent = inputs.get("wake_int_loss", 0.0)
        wake_mult = inputs.get("wake_loss_multiplier", 1.0)
        annual_wake_int_loss_percent *= wake_mult

        net_energy_single_turbine = total_e * (1.0 - annual_wake_int_loss_percent / 100.0)

        n_steps = 8760
        haf.setup(n_steps)
        gen_profile = np.full(n_steps, (net_energy_single_turbine * wpc.n_turbines) / n_steps)

        for i in range(n_steps):
            gen_profile[i] *= loss_multiplier * haf(i)

        outputs["gen"] = gen_profile
        outputs["annual_energy"] = np.sum(gen_profile)
        outputs["annual_gross_energy"] = gross_energy
        outputs["wind_speed_average"] = inputs["weibull_wind_speed"]

    else:
        wake_model_choice = inputs["wind_farm_wake_model"]
        wake_model = None
        if wake_model_choice == WakeModelOptions.SIMPLE:
            wake_model = SimpleWakeModel(wpc.n_turbines, wt)
        elif wake_model_choice == WakeModelOptions.PARK:
            wake_model = ParkWakeModel(wpc.n_turbines, wt)
        elif wake_model_choice == WakeModelOptions.CONSTANTVALUE:
            annual_wake_int_loss_percent = inputs.get("wake_int_loss", 0.0)
            derate = (100.0 - annual_wake_int_loss_percent) / 100.0
            wake_model = ConstantWakeModel(wpc.n_turbines, wt, derate)

        if not wpc.initialize_model(wake_model):
            raise RuntimeError("Falha ao inicializar o modelo de esteira.")

        if model_choice == 0:
            wdprov = WindData(inputs["wind_resource_data"])
            n_steps = wdprov.n_records()
            haf.setup(n_steps)

            gen = np.zeros(n_steps)
            annual_energy = 0.0
            annual_gross = 0.0
            annual_after_wake_loss = 0.0

            for i in range(n_steps):
                wind, wdir, temp, pres, meas_ht, closest_dir_meas_ht = [0], [0], [0], [0], [0], [0]
                wdprov.read(wt.hub_height, wind, wdir, temp, pres, meas_ht, closest_dir_meas_ht, True)

                power = np.zeros(wpc.n_turbines)
                farm_p, gross_p = wpc.wind_power_using_resource(
                    wind[0], wdir[0], pres[0], temp[0],
                    power, np.zeros(wpc.n_turbines), np.zeros(wpc.n_turbines),
                    np.zeros(wpc.n_turbines), np.zeros(wpc.n_turbines),
                    np.zeros(wpc.n_turbines), np.zeros(wpc.n_turbines),
                )

                annual_gross += gross_p / (n_steps / 8760.0)
                annual_after_wake_loss += farm_p / (n_steps / 8760.0)

                farm_p *= loss_multiplier * haf(i)
                gen[i] = farm_p
                annual_energy += gen[i] / (n_steps / 8760.0)

            outputs["gen"] = gen
            outputs["annual_energy"] = annual_energy
            outputs["annual_gross_energy"] = annual_gross
            outputs["wind_speed_average"] = float(np.mean(weather_inputs["data"][:, 0]))
            if annual_gross > 0:
                annual_wake_int_loss_percent = (1.0 - annual_after_wake_loss / annual_gross) * 100.0

    # 5. Calcular Saídas Finais e Métricas
    nameplate = inputs["system_capacity"]
    outputs["system_capacity"] = nameplate
    outputs["n_turbines"] = wpc.n_turbines
    if nameplate > 0:
        kwh_per_kw = outputs["annual_energy"] / nameplate
        outputs["kwh_per_kw"] = kwh_per_kw
        outputs["capacity_factor"] = kwh_per_kw / 87.6

    _calculate_losses(outputs, annual_wake_int_loss_percent, inputs)
    return outputs


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL (1 ANO DE OPERAÇÃO DA USINA EÓLICA)
# ==============================================================================

if __name__ == '__main__':
    # 1. Caminho dos arquivos de entrada (mesma estrutura do seu geracao_hibrida_2.py)
    weather_path = r'hibrido\historico_clima_macau'
    parametros_eolica = r'hibrido\eolica\sam\eolica_windpower.json'

    # Seleciona o primeiro ano (2000) da pasta de Macau ou defina um arquivo específico abaixo:
    weather_files = sorted([
        f for f in os.listdir(weather_path)
        if f.endswith('.csv') and int(f[:-4][-4:]) >= 2000
    ])
    arquivo_clima_ano1 = os.path.join(weather_path, weather_files[0])

    print("=" * 60)
    print(f"Simulando 1 ano de geração eólica com o arquivo:\n  {arquivo_clima_ano1}")
    print("=" * 60)

    st = time.time()
    resultado_eolica = simular_eolica(arquivo_clima_ano1, parametros_eolica)
    tempo_simulacao = time.time() - st

    energia_horaria_eolica = np.array(resultado_eolica.get('gen', [0] * 8760))  # [kWh]
    cap_instalada_kw = resultado_eolica.get('system_capacity', 100000.0)
    n_turbinas = resultado_eolica.get('n_turbines', 0)
    vel_media_cubo = resultado_eolica.get('wind_speed_average', 0.0)

    energia_anual_kwh = float(np.sum(energia_horaria_eolica))
    energia_anual_mwh = energia_anual_kwh / 1e3
    energia_anual_gwh = energia_anual_kwh / 1e6
    potencia_maxima_kw = float(np.max(energia_horaria_eolica))
    fator_capacidade = resultado_eolica.get('capacity_factor', 0.0)

    print(f"\nSimulação Eólica concluída em {tempo_simulacao:.2f} segundos")
    print("-" * 60)
    print(f"  Turbinas no Parque          : {n_turbinas} unidades")
    print(f"  Capacidade Instalada (JSON) : {cap_instalada_kw:,.2f} kW ({cap_instalada_kw / 1000:,.2f} MW)")
    print(f"  Vel. Média na Altura do Cubo: {vel_media_cubo:.2f} m/s")
    print(f"  Potência Máxima Atingida    : {potencia_maxima_kw:,.2f} kW ({potencia_maxima_kw / 1000:,.2f} MW)")
    print(f"  Energia Anual Bruta         : {resultado_eolica['annual_gross_energy'] / 1e6:,.2f} GWh")
    print(f"  Energia Anual Eólica (Líq.) : {energia_anual_gwh:,.2f} GWh ({energia_anual_mwh:,.2f} MWh | {energia_anual_kwh:,.2f} kWh)")
    print(f"  Fator de Capacidade         : {fator_capacidade:.2f} %")
    print(f"  Perda Total por Esteira     : {resultado_eolica['annual_wake_loss_total_percent']:.2f} %")
    print("=" * 60)