"""
Basalt-PCM Thermal Buffer Sizing Calculator - Professional Edition v3.0

A Streamlit app for calculating PCM requirements for thermal buffering in walls.

Features:
- Core PCM sizing calculations
- Realistic Time-of-Use (TOU) economic model with peak/off-peak rates
- Year-round savings: cooling + heating season buffering
- HVAC downsizing capital offset
- 25-Year and 50-Year Lifecycle ROI (appropriate for passive materials)
- Peak demand charge savings
- Climate presets
- Material comparison mode
- Feasibility warnings
"""

import csv
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================================
# UNIT CONVERSION FUNCTIONS
# ============================================================================

def ft2_to_m2(area_ft2: float) -> float:
    """Convert square feet to square meters."""
    return area_ft2 * 0.09290304


def m2_to_ft2(area_m2: float) -> float:
    """Convert square meters to square feet."""
    return area_m2 / 0.09290304


def kg_to_lb(mass_kg: float) -> float:
    """Convert kilograms to pounds."""
    return mass_kg * 2.20462262


def lb_to_kg(mass_lb: float) -> float:
    """Convert pounds to kilograms."""
    return mass_lb / 2.20462262


def btu_hr_ft2_f_to_w_m2_k(u_imperial: float) -> float:
    """Convert U-value from BTU/(hr·ft²·°F) to W/(m²·K)."""
    return u_imperial * 5.678263


def w_m2_k_to_btu_hr_ft2_f(u_metric: float) -> float:
    """Convert U-value from W/(m²·K) to BTU/(hr·ft²·°F)."""
    return u_metric / 5.678263


def celsius_to_fahrenheit_delta(delta_c: float) -> float:
    """Convert temperature difference from Celsius to Fahrenheit."""
    return delta_c * 1.8


def fahrenheit_to_celsius_delta(delta_f: float) -> float:
    """Convert temperature difference from Fahrenheit to Celsius."""
    return delta_f / 1.8


def liters_to_cubic_inches(liters: float) -> float:
    """Convert liters to cubic inches."""
    return liters * 61.0237


def m3_to_liters(volume_m3: float) -> float:
    """Convert cubic meters to liters."""
    return volume_m3 * 1000


def mm_to_inches(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm / 25.4


# ============================================================================
# CLIMATE PRESETS
# ============================================================================

CLIMATE_PRESETS = {
    'Hot-Dry': {
        'delta_t_c': 18.0,
        'delta_t_f': 32.4,
        'duration_hours': 10.0,
        'cooling_days': 180,
        'heating_days': 60,
        'description': 'Desert climates with high diurnal temperature swings'
    },
    'Tropical-Humid': {
        'delta_t_c': 8.0,
        'delta_t_f': 14.4,
        'duration_hours': 6.0,
        'cooling_days': 300,
        'heating_days': 0,
        'description': 'Hot and humid with moderate daily temperature variation'
    },
    'Temperate': {
        'delta_t_c': 12.0,
        'delta_t_f': 21.6,
        'duration_hours': 8.0,
        'cooling_days': 120,
        'heating_days': 120,
        'description': 'Moderate climate with seasonal variations'
    },
    'Cold': {
        'delta_t_c': 15.0,
        'delta_t_f': 27.0,
        'duration_hours': 12.0,
        'cooling_days': 60,
        'heating_days': 200,
        'description': 'Cold climates with heating focus'
    }
}


# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================

def compute_energy_per_area(
    u_value_w_m2_k: float,
    delta_t_k: float,
    duration_hours: float
) -> float:
    """
    Calculate the energy per unit area that needs to be buffered.

    Parameters
    ----------
    u_value_w_m2_k : float
        Overall heat transfer coefficient in W/(m²·K)
    delta_t_k : float
        Temperature difference in Kelvin (or Celsius difference)
    duration_hours : float
        Duration of buffering in hours

    Returns
    -------
    float
        Energy per unit area in J/m²
    """
    if u_value_w_m2_k <= 0 or delta_t_k <= 0 or duration_hours <= 0:
        raise ValueError("All parameters must be positive values")

    duration_seconds = duration_hours * 3600
    heat_flux = u_value_w_m2_k * delta_t_k          # W/m²
    energy_per_area = heat_flux * duration_seconds    # J/m²
    return energy_per_area


def compute_effective_energy_per_kg(
    latent_heat_kj_kg: float,
    cp_kj_kg_k: float,
    usable_temp_swing_k: float
) -> float:
    """
    Calculate effective energy storage capacity of PCM per kilogram.

    Parameters
    ----------
    latent_heat_kj_kg : float
        Latent heat of fusion in kJ/kg
    cp_kj_kg_k : float
        Specific heat capacity of PCM in kJ/(kg·K)
    usable_temp_swing_k : float
        Usable temperature swing around melting point in K

    Returns
    -------
    float
        Effective energy per kg in kJ/kg
    """
    if latent_heat_kj_kg <= 0 or cp_kj_kg_k <= 0 or usable_temp_swing_k <= 0:
        raise ValueError("All parameters must be positive values")

    effective_energy = latent_heat_kj_kg + (cp_kj_kg_k * usable_temp_swing_k)
    return effective_energy


def compute_pcm_mass_per_area(
    energy_per_area_j_m2: float,
    effective_energy_kj_kg: float,
    safety_factor: float
) -> float:
    """
    Calculate the required PCM mass per unit area.

    Parameters
    ----------
    energy_per_area_j_m2 : float
        Energy to be buffered per unit area in J/m²
    effective_energy_kj_kg : float
        Effective energy storage capacity in kJ/kg
    safety_factor : float
        Safety factor multiplier (typically 1.0–2.0)

    Returns
    -------
    float
        PCM mass per unit area in kg/m²
    """
    if energy_per_area_j_m2 <= 0 or effective_energy_kj_kg <= 0 or safety_factor <= 0:
        raise ValueError("All parameters must be positive values")

    energy_per_area_kj_m2 = energy_per_area_j_m2 / 1000
    pcm_mass_per_area = (energy_per_area_kj_m2 / effective_energy_kj_kg) * safety_factor
    return pcm_mass_per_area


def compute_pcm_volume_and_thickness(
    pcm_mass_kg: float,
    pcm_density_kg_m3: float,
    wall_area_m2: float
) -> tuple:
    """
    Calculate PCM volume and equivalent uniform thickness.

    Returns
    -------
    tuple
        (volume_m3, thickness_m, volume_liters, thickness_mm)
    """
    if pcm_mass_kg <= 0 or pcm_density_kg_m3 <= 0 or wall_area_m2 <= 0:
        raise ValueError("All parameters must be positive values")

    volume_m3 = pcm_mass_kg / pcm_density_kg_m3
    thickness_m = volume_m3 / wall_area_m2
    volume_liters = m3_to_liters(volume_m3)
    thickness_mm = thickness_m * 1000
    return volume_m3, thickness_m, volume_liters, thickness_mm


def compute_composite_requirements(
    pcm_mass_kg: float,
    pcm_mass_fraction_percent: float,
    composite_density_kg_m3: float = None
) -> dict:
    """
    Calculate composite material requirements.

    Returns
    -------
    dict
        Dictionary containing composite mass and volume (if density provided)
    """
    if pcm_mass_kg <= 0 or pcm_mass_fraction_percent <= 0 or pcm_mass_fraction_percent > 100:
        raise ValueError("Invalid parameters for composite calculation")

    pcm_fraction = pcm_mass_fraction_percent / 100
    total_composite_mass_kg = pcm_mass_kg / pcm_fraction

    result = {
        'composite_mass_kg': total_composite_mass_kg,
        'composite_mass_lb': kg_to_lb(total_composite_mass_kg)
    }

    if composite_density_kg_m3 and composite_density_kg_m3 > 0:
        composite_volume_m3 = total_composite_mass_kg / composite_density_kg_m3
        result['composite_volume_m3'] = composite_volume_m3
        result['composite_volume_liters'] = m3_to_liters(composite_volume_m3)

    return result


# ============================================================================
# COST CALCULATION FUNCTIONS
# ============================================================================

def compute_cost_analysis(
    pcm_mass_kg: float,
    wall_area_m2: float,
    pcm_cost_per_kg: float,
    installation_cost_per_m2: float,
    labor_markup_percent: float,
    transport_cost: float,
    waste_factor_percent: float,
    use_composite: bool = False,
    composite_mass_kg: float = None,
    composite_cost_per_kg: float = None
) -> dict | None:
    """
    Calculate comprehensive cost analysis for PCM installation.

    Returns
    -------
    dict or None
        Dictionary containing all cost components, or None on error
    """
    try:
        if use_composite and composite_mass_kg and composite_cost_per_kg:
            raw_material_cost = composite_mass_kg * composite_cost_per_kg
        else:
            raw_material_cost = pcm_mass_kg * pcm_cost_per_kg

        waste_multiplier = 1 + (waste_factor_percent / 100)
        adjusted_material_cost = raw_material_cost * waste_multiplier

        base_installation_cost = wall_area_m2 * installation_cost_per_m2
        labor_cost = base_installation_cost * (labor_markup_percent / 100)

        total_cost = (
            adjusted_material_cost + base_installation_cost + labor_cost + transport_cost
        )

        cost_per_m2 = total_cost / wall_area_m2 if wall_area_m2 > 0 else 0
        cost_per_ft2 = cost_per_m2 / m2_to_ft2(1)

        return {
            'raw_material_cost': raw_material_cost,
            'adjusted_material_cost': adjusted_material_cost,
            'installation_cost': base_installation_cost,
            'labor_cost': labor_cost,
            'transport_cost': transport_cost,
            'total_cost': total_cost,
            'cost_per_m2': cost_per_m2,
            'cost_per_ft2': cost_per_ft2
        }
    except Exception as e:
        st.error(f"Error in cost calculation: {str(e)}")
        return None


def compute_tou_energy_savings(
    energy_per_area_j_m2: float,
    wall_area_m2: float,
    # Cooling season
    annual_cooling_days: int,
    cooling_cop: float,
    load_reduction_percent: float,
    # TOU rates
    peak_rate_per_kwh: float,
    offpeak_rate_per_kwh: float,
    pct_load_shifted_to_offpeak: float,
    # Demand charge savings
    peak_demand_reduction_kw: float,
    demand_charge_per_kw_month: float,
    # Heating season
    annual_heating_days: int,
    heating_efficiency: float,
    heating_energy_cost_per_kwh: float
) -> dict | None:
    """
    Calculate annual energy savings using a Time-of-Use (TOU) model that captures:
      - Load shifting value  (peak kWh shifted to cheaper off-peak hours)
      - Peak demand charge avoidance ($/kW/month)
      - Year-round utilisation (cooling + heating season buffering)

    The PCM does not reduce total energy consumption — it shifts *when* that
    energy is consumed.  The savings come from paying off-peak rates instead
    of peak rates for the buffered load, plus avoiding peak-kW demand charges.

    Parameters
    ----------
    energy_per_area_j_m2 : float
        Daily thermal energy buffered per m² (J/m²)
    wall_area_m2 : float
        Wall area (m²)
    annual_cooling_days : int
        Days per year the buffer operates in cooling mode
    cooling_cop : float
        COP of the cooling system
    load_reduction_percent : float
        Fraction of total wall-buffered load actively shifted (%)
    peak_rate_per_kwh : float
        On-peak electricity tariff ($/kWh)
    offpeak_rate_per_kwh : float
        Off-peak electricity tariff ($/kWh)
    pct_load_shifted_to_offpeak : float
        Percentage of buffered load successfully shifted to off-peak (%)
    peak_demand_reduction_kw : float
        Estimated peak kW reduction due to PCM (user-supplied or 0)
    demand_charge_per_kw_month : float
        Monthly demand charge ($/kW/month); 0 if flat-rate billing
    annual_heating_days : int
        Days per year the buffer operates in heating mode
    heating_efficiency : float
        Efficiency of the heating system (COP for heat-pump, or η for boiler)
    heating_energy_cost_per_kwh : float
        Cost of heating fuel converted to $/kWh-equivalent

    Returns
    -------
    dict or None
    """
    try:
        # --- Daily thermal energy buffered (kWh) ---
        daily_thermal_energy_j = energy_per_area_j_m2 * wall_area_m2
        daily_thermal_energy_kwh = daily_thermal_energy_j / 3.6e6

        rate_delta = max(0.0, peak_rate_per_kwh - offpeak_rate_per_kwh)
        fraction_shifted = pct_load_shifted_to_offpeak / 100.0
        effective_fraction = load_reduction_percent / 100.0

        # ---- COOLING SEASON ------------------------------------------------
        annual_cooling_thermal_kwh = daily_thermal_energy_kwh * annual_cooling_days
        # Electrical energy that can be shifted (thermal ÷ COP)
        cooling_shifted_elec_kwh = (
            annual_cooling_thermal_kwh * effective_fraction * fraction_shifted / cooling_cop
        )
        # Value: paying off-peak rate instead of peak rate
        cooling_tou_savings = cooling_shifted_elec_kwh * rate_delta

        # ---- HEATING SEASON ------------------------------------------------
        annual_heating_thermal_kwh = daily_thermal_energy_kwh * annual_heating_days
        # Heating equivalent electrical savings (thermal ÷ efficiency)
        heating_shifted_elec_kwh = (
            annual_heating_thermal_kwh * effective_fraction * fraction_shifted
            / max(heating_efficiency, 0.01)
        )
        heating_tou_savings = heating_shifted_elec_kwh * rate_delta

        # ---- DEMAND CHARGE AVOIDANCE ---------------------------------------
        # Active cooling months only (rough: cooling_days / 30)
        cooling_months = annual_cooling_days / 30.0
        annual_demand_savings = (
            peak_demand_reduction_kw * demand_charge_per_kw_month * cooling_months
        )

        # ---- TOTALS --------------------------------------------------------
        annual_tou_savings = cooling_tou_savings + heating_tou_savings
        annual_total_savings = annual_tou_savings + annual_demand_savings

        return {
            'daily_thermal_energy_kwh': daily_thermal_energy_kwh,
            'annual_cooling_thermal_kwh': annual_cooling_thermal_kwh,
            'annual_heating_thermal_kwh': annual_heating_thermal_kwh,
            'cooling_shifted_elec_kwh': cooling_shifted_elec_kwh,
            'heating_shifted_elec_kwh': heating_shifted_elec_kwh,
            'cooling_tou_savings': cooling_tou_savings,
            'heating_tou_savings': heating_tou_savings,
            'annual_demand_savings': annual_demand_savings,
            'annual_tou_savings': annual_tou_savings,
            'annual_total_savings': annual_total_savings,
        }
    except Exception as e:
        st.error(f"Error in TOU energy savings calculation: {str(e)}")
        return None


def compute_lifecycle_payback(
    net_project_cost: float,
    annual_savings: float
) -> dict:
    """
    Calculate simple payback period and multi-decade lifecycle ROI.

    Passive building materials (PCM, insulation, concrete) have service lives
    of 50+ years; a 10-year window severely understates their value.

    Parameters
    ----------
    net_project_cost : float
        Total project cost minus HVAC downsizing savings ($)
    annual_savings : float
        Annual cost savings ($/year)

    Returns
    -------
    dict
        payback_years, 25-year ROI, 50-year ROI
    """
    try:
        if annual_savings > 0:
            payback_years = net_project_cost / annual_savings
            roi_25 = ((annual_savings * 25) - net_project_cost) / net_project_cost * 100
            roi_50 = ((annual_savings * 50) - net_project_cost) / net_project_cost * 100
        else:
            payback_years = float('inf')
            roi_25 = -100.0
            roi_50 = -100.0

        return {
            'payback_years': payback_years,
            'roi_25yr': roi_25,
            'roi_50yr': roi_50
        }
    except Exception as e:
        st.error(f"Error in lifecycle payback calculation: {str(e)}")
        return {
            'payback_years': float('inf'),
            'roi_25yr': -100.0,
            'roi_50yr': -100.0
        }


# ============================================================================
# MATERIAL COMPARISON FUNCTIONS
# ============================================================================

def compute_alternative_material_comparison(
    energy_per_area_j_m2: float,
    wall_area_m2: float
) -> dict | None:
    """
    Compare PCM with EPS insulation and concrete thermal mass.
    """
    try:
        # EPS Insulation
        eps_r_per_mm = 0.037           # m²·K/W per mm
        eps_cost_per_m2_per_mm = 0.315  # $/m² per mm
        target_r_value = 2.0            # m²·K/W
        eps_thickness_mm = target_r_value / eps_r_per_mm
        eps_cost_per_m2 = eps_thickness_mm * eps_cost_per_m2_per_mm
        eps_total_cost = eps_cost_per_m2 * wall_area_m2

        # Concrete Thermal Mass
        concrete_cp = 0.88          # kJ/(kg·K)
        concrete_density = 2400     # kg/m³
        concrete_cost_per_m3 = 100  # $
        delta_t_concrete = 10       # K
        energy_per_area_kj_m2 = energy_per_area_j_m2 / 1000
        concrete_mass_per_m2 = energy_per_area_kj_m2 / (concrete_cp * delta_t_concrete)
        concrete_thickness_m = concrete_mass_per_m2 / concrete_density
        concrete_thickness_mm = concrete_thickness_m * 1000
        concrete_cost_per_m2 = concrete_thickness_m * concrete_cost_per_m3
        concrete_total_cost = concrete_cost_per_m2 * wall_area_m2

        return {
            'eps': {
                'thickness_mm': eps_thickness_mm,
                'cost_per_m2': eps_cost_per_m2,
                'total_cost': eps_total_cost,
                'performance_note': 'Reduces heat transfer (steady-state)'
            },
            'concrete': {
                'thickness_mm': concrete_thickness_mm,
                'cost_per_m2': concrete_cost_per_m2,
                'total_cost': concrete_total_cost,
                'performance_note': 'Thermal mass (time-lag)'
            }
        }
    except Exception as e:
        st.error(f"Error in material comparison: {str(e)}")
        return None


# ============================================================================
# FEASIBILITY CHECK FUNCTIONS
# ============================================================================

def check_feasibility(
    thickness_mm: float,
    cost_per_m2: float,
    payback_years: float,
    pcm_mass_fraction_percent: float = None
) -> list:
    """
    Check project feasibility and generate warnings.

    Returns
    -------
    list
        List of warning dicts with 'level' and 'message'
    """
    warnings = []

    if thickness_mm > 50:
        warnings.append({
            'level': 'warning',
            'message': (
                f"⚠️ Required thickness ({thickness_mm:.1f} mm) exceeds 50 mm. "
                "This may present installation challenges and require structural modifications."
            )
        })

    typical_insulation_cost = 40  # $/m²
    if cost_per_m2 > typical_insulation_cost:
        warnings.append({
            'level': 'warning',
            'message': (
                f"⚠️ Cost per m² (${cost_per_m2:.2f}) exceeds typical insulation cost "
                f"(${typical_insulation_cost}/m²). Consider cost-effectiveness."
            )
        })

    # For passive materials, the relevant threshold is service life, not 15 yrs
    if payback_years > 50:
        warnings.append({
            'level': 'error',
            'message': (
                f"🚨 Payback period ({payback_years:.1f} years) exceeds 50-year "
                "material service life. Project is unlikely to be economically viable."
            )
        })
    elif payback_years > 25:
        warnings.append({
            'level': 'warning',
            'message': (
                f"⚠️ Payback period ({payback_years:.1f} years) is long. "
                "Verify TOU rate spread and demand-charge assumptions."
            )
        })

    if pcm_mass_fraction_percent and pcm_mass_fraction_percent > 70:
        warnings.append({
            'level': 'warning',
            'message': (
                f"⚠️ PCM mass fraction ({pcm_mass_fraction_percent:.0f}%) exceeds 70%. "
                "This may affect structural integrity and workability of the composite."
            )
        })

    return warnings


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_inputs(
    u_value: float,
    delta_t: float,
    duration: float,
    latent_heat: float,
    cp: float,
    usable_swing: float,
    density: float,
    safety_factor: float,
    # --- NEW TOU / heating parameters ---
    peak_rate: float = None,
    offpeak_rate: float = None,
    pct_shifted: float = None,
    demand_charge: float = None,
    cooling_cop: float = None,
    heating_efficiency: float = None,
    annual_cooling_days: int = None,
    annual_heating_days: int = None,
    hvac_downsizing_savings: float = None
) -> list:
    """
    Validate user inputs and return a list of warning strings.

    New validation rules added in v3.0
    ------------------------------------
    - peak_rate must be >= offpeak_rate (you don't save money otherwise)
    - pct_shifted must be 0–100 %
    - demand_charge must be >= 0
    - cooling_cop must be >= 1
    - heating_efficiency must be > 0 and <= 6
    - cooling_days + heating_days must be <= 365
    - hvac_downsizing_savings must be >= 0
    """
    warnings = []

    # Core physics inputs
    if u_value <= 0:
        warnings.append("U-value must be positive.")
    if u_value > 10:
        warnings.append("U-value > 10 W/(m²·K) is unusually high. Check your input.")
    if delta_t <= 0.1:
        warnings.append("Temperature difference is very small. Results may be near zero.")
    if delta_t > 50:
        warnings.append("Temperature difference > 50 K is very large. Verify your input.")
    if duration <= 0:
        warnings.append("Duration must be positive.")
    if duration > 168:
        warnings.append("Duration > 1 week (168 hours). Long-term assumptions may not hold.")
    if latent_heat <= 0:
        warnings.append("Latent heat must be positive.")
    if latent_heat < 50:
        warnings.append("Latent heat < 50 kJ/kg is low for typical PCMs.")
    if latent_heat > 400:
        warnings.append("Latent heat > 400 kJ/kg is unusually high. Verify your input.")
    if cp <= 0:
        warnings.append("Specific heat capacity must be positive.")
    if cp > 10:
        warnings.append("Specific heat > 10 kJ/(kg·K) is unusually high.")
    if usable_swing <= 0:
        warnings.append("Usable temperature swing must be positive.")
    if usable_swing > 20:
        warnings.append("Usable temperature swing > 20 K is unusually large for typical PCMs.")
    if density <= 0:
        warnings.append("PCM density must be positive.")
    if density < 100:
        warnings.append("Density < 100 kg/m³ is very low. Verify your input.")
    if density > 3000:
        warnings.append("Density > 3000 kg/m³ is very high. Verify your input.")
    if safety_factor < 1.0:
        warnings.append("Safety factor should typically be >= 1.0.")
    if safety_factor > 3.0:
        warnings.append("Safety factor > 3.0 is very conservative. Results will be oversized.")

    # --- NEW: TOU & economic inputs ---
    if peak_rate is not None and offpeak_rate is not None:
        if offpeak_rate < 0:
            warnings.append("Off-peak rate cannot be negative.")
        if peak_rate < offpeak_rate:
            warnings.append(
                "Peak rate is less than off-peak rate. TOU savings will be zero or negative; "
                "the rate differential drives load-shifting value."
            )
        if peak_rate > 1.0:
            warnings.append(
                "Peak rate > $1.00/kWh is unusually high. Verify your utility tariff."
            )

    if pct_shifted is not None:
        if not (0 <= pct_shifted <= 100):
            warnings.append("% Load Shifted to Off-Peak must be between 0 and 100.")

    if demand_charge is not None and demand_charge < 0:
        warnings.append("Demand charge ($/kW/month) cannot be negative.")

    if cooling_cop is not None and cooling_cop < 1.0:
        warnings.append("Cooling COP < 1.0 violates thermodynamic limits.")

    if heating_efficiency is not None:
        if heating_efficiency <= 0:
            warnings.append("Heating efficiency must be positive.")
        if heating_efficiency > 6.0:
            warnings.append(
                "Heating efficiency > 6.0 is unusually high "
                "(only advanced heat-pumps reach COP ≈ 5–6)."
            )

    if annual_cooling_days is not None and annual_heating_days is not None:
        if annual_cooling_days + annual_heating_days > 365:
            warnings.append(
                "Cooling days + heating days exceed 365. "
                "Ensure the two seasons do not overlap."
            )

    if hvac_downsizing_savings is not None and hvac_downsizing_savings < 0:
        warnings.append("HVAC downsizing savings cannot be negative.")

    return warnings


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_sensitivity_plot_duration(
    u_value_w_m2_k, delta_t_k, latent_heat, cp, usable_swing,
    safety_factor, wall_area_ft2, duration_range=None
):
    """Create sensitivity plot of PCM requirement vs. buffering duration."""
    try:
        if duration_range is None:
            duration_range = np.linspace(1, 24, 50)

        pcm_lb_ft2_values = []
        for duration in duration_range:
            try:
                energy_pa = compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration)
                eff_energy = compute_effective_energy_per_kg(latent_heat, cp, usable_swing)
                pcm_kg_m2 = compute_pcm_mass_per_area(energy_pa, eff_energy, safety_factor)
                pcm_lb_ft2_values.append(kg_to_lb(pcm_kg_m2) / m2_to_ft2(1))
            except Exception:
                pcm_lb_ft2_values.append(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(duration_range, pcm_lb_ft2_values, linewidth=2, color='#1f77b4')
        ax.set_xlabel('Duration (hours)', fontsize=12)
        ax.set_ylabel('PCM Required (lb/ft²)', fontsize=12)
        ax.set_title('PCM Requirement vs. Buffering Duration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(duration_range[0], duration_range[-1])
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig


def create_sensitivity_plot_delta_t(
    u_value_w_m2_k, duration_hours, latent_heat, cp, usable_swing,
    safety_factor, wall_area_ft2, delta_t_range=None
):
    """Create sensitivity plot of PCM requirement vs. temperature difference."""
    try:
        if delta_t_range is None:
            delta_t_range = np.linspace(1, 30, 50)

        pcm_lb_ft2_values = []
        for delta_t in delta_t_range:
            try:
                energy_pa = compute_energy_per_area(u_value_w_m2_k, delta_t, duration_hours)
                eff_energy = compute_effective_energy_per_kg(latent_heat, cp, usable_swing)
                pcm_kg_m2 = compute_pcm_mass_per_area(energy_pa, eff_energy, safety_factor)
                pcm_lb_ft2_values.append(kg_to_lb(pcm_kg_m2) / m2_to_ft2(1))
            except Exception:
                pcm_lb_ft2_values.append(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(delta_t_range, pcm_lb_ft2_values, linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Temperature Difference (°C or K)', fontsize=12)
        ax.set_ylabel('PCM Required (lb/ft²)', fontsize=12)
        ax.set_title(
            'PCM Requirement vs. Temperature Difference', fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(delta_t_range[0], delta_t_range[-1])
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig


def create_savings_waterfall_chart(savings_results: dict) -> plt.Figure:
    """
    Create a waterfall chart breaking down the sources of annual savings.
    """
    labels = [
        'Cooling TOU\nSavings',
        'Heating TOU\nSavings',
        'Demand Charge\nSavings',
        'Total Annual\nSavings'
    ]
    values = [
        savings_results['cooling_tou_savings'],
        savings_results['heating_tou_savings'],
        savings_results['annual_demand_savings'],
        savings_results['annual_total_savings']
    ]
    colours = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"${val:,.0f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    ax.set_ylabel('Annual Savings ($)', fontsize=12)
    ax.set_title('Annual Savings Breakdown by Source', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    return fig


# ============================================================================
# CSV EXPORT FUNCTION
# ============================================================================

def generate_csv_summary(inputs: dict, outputs: dict) -> str:
    """
    Generate a CSV summary of inputs and outputs using proper CSV formatting.
    """
    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Basalt-PCM Thermal Buffer Sizing Calculator - Professional Edition v3.0'])
    writer.writerow([])
    writer.writerow(['INPUTS'])
    for key, value in inputs.items():
        writer.writerow([key, value])
    writer.writerow([])
    writer.writerow(['OUTPUTS'])
    for key, value in outputs.items():
        writer.writerow([key, value])

    return output.getvalue()


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main function to run the Streamlit app."""

    st.set_page_config(
        page_title="PCM Thermal Buffer Calculator - Professional Edition",
        page_icon="🔥",
        layout="wide"
    )

    st.title("🔥 Basalt-PCM Thermal Buffer Sizing Calculator — Professional Edition v3.0")
    st.markdown(
        "Calculates the required Phase Change Material (PCM) mass for thermal buffering in wall "
        "assemblies, with a **realistic TOU economic model**, year-round savings, HVAC downsizing "
        "offsets, and 25/50-year lifecycle ROI appropriate for passive building materials."
    )

    st.sidebar.header("📊 Input Parameters")

    # ========================================================================
    # CLIMATE PRESETS
    # ========================================================================
    st.sidebar.subheader("🌍 Climate Type")

    climate_type = st.sidebar.selectbox(
        "Select Climate Preset",
        options=['Custom'] + list(CLIMATE_PRESETS.keys()),
        help="Select a climate type to auto-populate typical values"
    )

    if climate_type != 'Custom':
        preset = CLIMATE_PRESETS[climate_type]
        st.sidebar.info(f"📍 {preset['description']}")
        preset_delta_t_c = preset['delta_t_c']
        preset_delta_t_f = preset['delta_t_f']
        preset_duration = preset['duration_hours']
        preset_cooling_days = preset['cooling_days']
        preset_heating_days = preset['heating_days']
    else:
        preset_delta_t_c = None
        preset_delta_t_f = None
        preset_duration = None
        preset_cooling_days = None
        preset_heating_days = None

    # ========================================================================
    # SECTION A: Geometry and Design Target
    # ========================================================================
    st.sidebar.subheader("A. Geometry and Design Target")

    wall_area_ft2 = st.sidebar.number_input(
        "Wall Area (ft²)",
        min_value=0.1, value=100.0, step=1.0,
        help="Total wall area to be analysed"
    )

    units_system = st.sidebar.radio(
        "Preferred Units",
        options=["Imperial", "Metric"],
        help="Primary unit system for inputs (outputs always include both)"
    )

    duration_hours = st.sidebar.number_input(
        "Duration of Buffering (hours)",
        min_value=0.1,
        value=float(preset_duration) if preset_duration else 8.0,
        step=0.5,
        help=(
            f"Climate preset suggests {preset_duration} hours"
            if preset_duration else "Time period requiring thermal buffering"
        )
    )

    if units_system == "Imperial":
        delta_t_input = st.sidebar.number_input(
            "Temperature Difference (°F)",
            min_value=0.1,
            value=float(preset_delta_t_f) if preset_delta_t_f else 20.0,
            step=1.0,
            help="Temperature difference between outside and inside"
        )
        delta_t_k = fahrenheit_to_celsius_delta(delta_t_input)
    else:
        delta_t_input = st.sidebar.number_input(
            "Temperature Difference (°C)",
            min_value=0.1,
            value=float(preset_delta_t_c) if preset_delta_t_c else 11.1,
            step=0.5,
            help="Temperature difference between outside and inside"
        )
        delta_t_k = delta_t_input

    u_value_units = st.sidebar.selectbox(
        "U-value Units",
        options=["W/(m²·K)", "BTU/(hr·ft²·°F)"],
        help="Select units for thermal transmittance input"
    )

    if u_value_units == "W/(m²·K)":
        u_value_input = st.sidebar.number_input(
            "U-value (W/(m²·K))", min_value=0.01, value=0.5, step=0.1,
            help="Overall heat transfer coefficient"
        )
        u_value_w_m2_k = u_value_input
    else:
        u_value_input = st.sidebar.number_input(
            "U-value (BTU/(hr·ft²·°F))", min_value=0.01, value=0.088, step=0.01,
            help="Overall heat transfer coefficient"
        )
        u_value_w_m2_k = btu_hr_ft2_f_to_w_m2_k(u_value_input)

    safety_factor = st.sidebar.slider(
        "Safety Factor", min_value=1.0, max_value=2.0, value=1.2, step=0.1,
        help="Multiplier to account for uncertainties"
    )

    # ========================================================================
    # SECTION B: PCM Properties
    # ========================================================================
    st.sidebar.subheader("B. PCM Properties")

    latent_heat_kj_kg = st.sidebar.number_input(
        "Latent Heat of Fusion (kJ/kg)", min_value=1.0, value=200.0, step=10.0,
        help="Latent heat of phase change"
    )
    cp_pcm_kj_kg_k = st.sidebar.number_input(
        "Specific Heat Capacity (kJ/(kg·K))", min_value=0.1, value=2.0, step=0.1
    )
    usable_temp_swing_k = st.sidebar.number_input(
        "Usable Temperature Swing (K)", min_value=0.1, value=6.0, step=0.5,
        help="Effective temperature range around melting point"
    )
    pcm_density_kg_m3 = st.sidebar.number_input(
        "PCM Density (kg/m³)", min_value=1.0, value=800.0, step=50.0
    )

    # ========================================================================
    # SECTION C: Optional Composite Constraints
    # ========================================================================
    st.sidebar.subheader("C. Optional: Composite Constraints")

    use_composite = st.sidebar.checkbox(
        "Calculate Composite Requirements", value=False
    )

    composite_density_kg_m3 = None
    pcm_mass_fraction = None
    composite_cost_per_kg = None

    if use_composite:
        pcm_mass_fraction = st.sidebar.number_input(
            "PCM Mass Fraction in Composite (%)",
            min_value=1.0, max_value=100.0, value=20.0, step=1.0
        )
        composite_density_kg_m3 = st.sidebar.number_input(
            "Composite Density (kg/m³)", min_value=1.0, value=1200.0, step=50.0
        )

    # ========================================================================
    # SECTION D: Economic Analysis (material + installation costs)
    # ========================================================================
    st.sidebar.subheader("D. 💰 Economic Analysis")

    enable_economic = st.sidebar.checkbox(
        "Enable Cost Estimation", value=True
    )

    # Defaults so variables are always defined
    pcm_cost_per_kg = 5.0
    installation_cost_per_m2 = 15.0
    labor_markup_percent = 30.0
    transport_cost = 200.0
    waste_factor_percent = 10.0

    if enable_economic:
        with st.sidebar.expander("📦 Material & Installation Costs", expanded=True):
            pcm_cost_per_kg = st.number_input(
                "PCM Cost ($/kg)", min_value=0.01, value=5.0, step=0.5,
                help="Cost per kilogram of PCM"
            )
            installation_cost_per_m2 = st.number_input(
                "Installation Cost ($/m²)", min_value=0.0, value=15.0, step=1.0
            )
            labor_markup_percent = st.number_input(
                "Labor Markup (%)", min_value=0.0, value=30.0, step=5.0,
                help="Labor cost as percentage of installation cost"
            )
            transport_cost = st.number_input(
                "Transport Cost ($)", min_value=0.0, value=200.0, step=50.0
            )
            waste_factor_percent = st.number_input(
                "Waste Factor (%)", min_value=0.0, value=10.0, step=1.0
            )
            if use_composite:
                composite_cost_per_kg = st.number_input(
                    "Composite Cost ($/kg)", min_value=0.01, value=3.0, step=0.5
                )

        with st.sidebar.expander("🏗️ HVAC Downsizing Savings", expanded=False):
            st.markdown(
                "PCM thermal mass allows HVAC equipment to be downsized at design time, "
                "creating an upfront capital saving that offsets the PCM installation cost."
            )
            hvac_load_reduction_tons = st.number_input(
                "Peak Load Reduction (tons of cooling)",
                min_value=0.0, value=0.0, step=0.5,
                help="Estimated reduction in design cooling capacity due to PCM"
            )
            hvac_cost_per_ton = st.number_input(
                "HVAC Equipment Cost per Ton ($)",
                min_value=0.0, value=1200.0, step=100.0,
                help="Installed cost per ton of HVAC capacity (typical: $800–$2,000/ton)"
            )
            hvac_downsizing_savings = hvac_load_reduction_tons * hvac_cost_per_ton
            if hvac_downsizing_savings > 0:
                st.info(
                    f"💡 Estimated HVAC downsizing saving: **${hvac_downsizing_savings:,.0f}**"
                )
    else:
        hvac_downsizing_savings = 0.0

    # ========================================================================
    # SECTION E: Energy Savings — TOU Model + Heating Season
    # ========================================================================
    st.sidebar.subheader("E. ⚡ Energy Savings (TOU Model)")

    enable_savings = st.sidebar.checkbox("Enable Savings Analysis", value=True)

    # Defaults so variables are always defined
    peak_rate_per_kwh = 0.20
    offpeak_rate_per_kwh = 0.08
    pct_load_shifted = 70.0
    peak_demand_reduction_kw = 0.0
    demand_charge_per_kw_month = 0.0
    cooling_cop = 3.0
    load_reduction_percent = 15.0
    annual_cooling_days = 120
    annual_heating_days = 90
    heating_efficiency = 0.90
    heating_energy_cost_per_kwh = 0.10

    if enable_savings:
        with st.sidebar.expander("🌡️ Cooling Season", expanded=True):
            annual_cooling_days = st.number_input(
                "Annual Cooling Days",
                min_value=0, max_value=365,
                value=int(preset_cooling_days) if preset_cooling_days else 120,
                step=10,
                help="Days per year requiring active cooling"
            )
            cooling_cop = st.number_input(
                "Cooling System COP", min_value=1.0, value=3.0, step=0.5,
                help="Coefficient of Performance of the cooling system"
            )
            load_reduction_percent = st.number_input(
                "Effective Load Reduction (%)", min_value=0.0, max_value=100.0,
                value=15.0, step=1.0,
                help="Fraction of the buffered thermal load that is actively shifted"
            )

        with st.sidebar.expander("🔥 Heating Season", expanded=False):
            annual_heating_days = st.number_input(
                "Annual Heating Days",
                min_value=0, max_value=365,
                value=int(preset_heating_days) if preset_heating_days else 90,
                step=10,
                help="Days per year the PCM buffer assists the heating system"
            )
            heating_efficiency = st.number_input(
                "Heating System Efficiency (COP or η)",
                min_value=0.1, value=0.90, step=0.05,
                help=(
                    "COP for heat pumps (typical 2.5–4.5); "
                    "combustion efficiency η for boilers (typical 0.80–0.97)"
                )
            )
            heating_energy_cost_per_kwh = st.number_input(
                "Heating Energy Cost ($/kWh equivalent)",
                min_value=0.001, value=0.10, step=0.01,
                help=(
                    "Use electricity rate for heat pumps. "
                    "For gas boilers: gas price ÷ boiler efficiency, in $/kWh-thermal."
                )
            )

        with st.sidebar.expander("⚡ Time-of-Use (TOU) Rates", expanded=True):
            st.markdown(
                "The PCM *shifts* load from expensive peak hours to cheap off-peak hours. "
                "The savings come from the **rate differential**, not just consumption reduction."
            )
            peak_rate_per_kwh = st.number_input(
                "On-Peak Rate ($/kWh)", min_value=0.01, value=0.20, step=0.01,
                help="Electricity tariff during peak demand hours"
            )
            offpeak_rate_per_kwh = st.number_input(
                "Off-Peak Rate ($/kWh)", min_value=0.001, value=0.08, step=0.01,
                help="Electricity tariff during off-peak / overnight hours"
            )
            pct_load_shifted = st.number_input(
                "% of Buffered Load Shifted to Off-Peak",
                min_value=0.0, max_value=100.0, value=70.0, step=5.0,
                help=(
                    "What fraction of the buffered thermal load is successfully discharged "
                    "during off-peak hours (i.e., HVAC runs overnight instead of midday)"
                )
            )

        with st.sidebar.expander("📉 Demand Charge Savings (optional)", expanded=False):
            st.markdown(
                "Many commercial tariffs include a separate $/kW/month 'demand charge' "
                "based on your highest 15-min peak draw. PCM peak shaving can reduce this charge."
            )
            peak_demand_reduction_kw = st.number_input(
                "Estimated Peak Demand Reduction (kW)",
                min_value=0.0, value=0.0, step=0.5,
                help="Reduction in coincident peak kW draw due to PCM load shifting"
            )
            demand_charge_per_kw_month = st.number_input(
                "Demand Charge ($/kW/month)",
                min_value=0.0, value=0.0, step=0.5,
                help="Enter 0 if your tariff is a flat kWh rate with no demand charge"
            )

    # ========================================================================
    # SECTION F: Material Comparison
    # ========================================================================
    st.sidebar.subheader("F. 🧱 Material Comparison")

    enable_comparison = st.sidebar.checkbox(
        "Enable Alternative Material Comparison", value=False
    )

    # ========================================================================
    # PERFORM CALCULATIONS
    # ========================================================================

    try:
        wall_area_m2 = ft2_to_m2(wall_area_ft2)

        # Validate all inputs including new economic parameters
        warnings = validate_inputs(
            u_value_w_m2_k, delta_t_k, duration_hours,
            latent_heat_kj_kg, cp_pcm_kj_kg_k, usable_temp_swing_k,
            pcm_density_kg_m3, safety_factor,
            peak_rate=peak_rate_per_kwh if enable_savings else None,
            offpeak_rate=offpeak_rate_per_kwh if enable_savings else None,
            pct_shifted=pct_load_shifted if enable_savings else None,
            demand_charge=demand_charge_per_kw_month if enable_savings else None,
            cooling_cop=cooling_cop if enable_savings else None,
            heating_efficiency=heating_efficiency if enable_savings else None,
            annual_cooling_days=annual_cooling_days if enable_savings else None,
            annual_heating_days=annual_heating_days if enable_savings else None,
            hvac_downsizing_savings=hvac_downsizing_savings if enable_economic else None
        )

        if warnings:
            st.sidebar.warning("⚠️ Input Validation Warnings:")
            for w in warnings:
                st.sidebar.warning(f"• {w}")

        # Core PCM sizing
        energy_per_area_j_m2 = compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration_hours)
        effective_energy_kj_kg = compute_effective_energy_per_kg(
            latent_heat_kj_kg, cp_pcm_kj_kg_k, usable_temp_swing_k
        )
        pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(
            energy_per_area_j_m2, effective_energy_kj_kg, safety_factor
        )

        pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
        total_pcm_kg = pcm_mass_per_area_kg_m2 * wall_area_m2
        total_pcm_lb = kg_to_lb(total_pcm_kg)

        volume_m3, thickness_m, volume_liters, thickness_mm = compute_pcm_volume_and_thickness(
            total_pcm_kg, pcm_density_kg_m3, wall_area_m2
        )
        volume_cubic_inches = liters_to_cubic_inches(volume_liters)
        thickness_inches = mm_to_inches(thickness_mm)

        # Composite requirements (optional)
        composite_results = None
        if use_composite and pcm_mass_fraction and composite_density_kg_m3:
            composite_results = compute_composite_requirements(
                total_pcm_kg, pcm_mass_fraction, composite_density_kg_m3
            )

        # ====================================================================
        # DISPLAY: MAIN SIZING OUTPUTS
        # ====================================================================
        st.header("📈 Calculation Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "PCM Required",
                f"{pcm_mass_per_area_lb_ft2:.3f} lb/ft²",
                help="PCM mass per unit area (imperial)"
            )
            st.caption(f"({pcm_mass_per_area_kg_m2:.3f} kg/m²)")

        with col2:
            st.metric(
                "Total PCM Mass",
                f"{total_pcm_lb:.1f} lb",
                help="Total PCM mass for the entire wall"
            )
            st.caption(f"({total_pcm_kg:.1f} kg)")

        with col3:
            st.metric(
                "PCM Volume",
                f"{volume_liters:.2f} L",
                help="Total volume of PCM required"
            )
            st.caption(f"({volume_cubic_inches:.1f} in³)")

        with col4:
            st.metric(
                "Equivalent Thickness",
                f"{thickness_mm:.2f} mm",
                help="Thickness if PCM is spread uniformly over wall area"
            )
            st.caption(f"({thickness_inches:.3f} in)")

        # Composite display
        if composite_results:
            st.subheader("🧱 Composite Material Requirements")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Composite Mass",
                    f"{composite_results['composite_mass_lb']:.1f} lb",
                    help=f"Total composite needed at {pcm_mass_fraction}% PCM fraction"
                )
                st.caption(f"({composite_results['composite_mass_kg']:.1f} kg)")
            with col2:
                if 'composite_volume_liters' in composite_results:
                    st.metric(
                        "Composite Volume",
                        f"{composite_results['composite_volume_liters']:.2f} L"
                    )
                    st.caption(f"({composite_results['composite_volume_m3']:.3f} m³)")

            comp_per_area_kg_m2 = composite_results['composite_mass_kg'] / wall_area_m2
            comp_per_area_lb_ft2 = composite_results['composite_mass_lb'] / wall_area_ft2
            st.info(
                f"**Composite per unit area:** {comp_per_area_lb_ft2:.3f} lb/ft² "
                f"({comp_per_area_kg_m2:.3f} kg/m²)"
            )

        # ====================================================================
        # DISPLAY: ECONOMIC ANALYSIS
        # ====================================================================
        cost_results = None
        if enable_economic:
            st.header("💰 Economic Analysis")

            cost_results = compute_cost_analysis(
                total_pcm_kg, wall_area_m2, pcm_cost_per_kg,
                installation_cost_per_m2, labor_markup_percent,
                transport_cost, waste_factor_percent,
                use_composite,
                composite_results['composite_mass_kg'] if composite_results else None,
                composite_cost_per_kg
            )

            if cost_results:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Raw Material Cost",
                              f"${cost_results['raw_material_cost']:,.2f}")
                    st.metric(
                        "Adjusted Material Cost",
                        f"${cost_results['adjusted_material_cost']:,.2f}",
                        help=f"After {waste_factor_percent}% waste factor"
                    )

                with col2:
                    st.metric("Installation Cost",
                              f"${cost_results['installation_cost']:,.2f}")
                    st.metric(
                        "Labor Cost",
                        f"${cost_results['labor_cost']:,.2f}",
                        help=f"At {labor_markup_percent}% markup"
                    )

                with col3:
                    st.metric("Transport Cost",
                              f"${cost_results['transport_cost']:,.2f}")
                    st.metric("Gross Project Cost",
                              f"${cost_results['total_cost']:,.2f}",
                              help="Sum of all PCM/installation cost components")

                # HVAC downsizing offset
                net_project_cost = cost_results['total_cost'] - hvac_downsizing_savings

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "HVAC Downsizing Saving",
                        f"-${hvac_downsizing_savings:,.2f}",
                        help="Capital saving from smaller HVAC equipment at design time"
                    )
                with col2:
                    st.metric(
                        "Net Project Cost",
                        f"${max(net_project_cost, 0):,.2f}",
                        delta=f"-${hvac_downsizing_savings:,.0f} vs gross" if hvac_downsizing_savings > 0 else None,
                        delta_color="inverse",
                        help="Gross cost minus HVAC downsizing savings — used for payback calc"
                    )
                with col3:
                    st.metric("Cost per ft²",
                              f"${cost_results['cost_per_ft2']:,.2f}/ft²")
                    st.caption(f"(${cost_results['cost_per_m2']:,.2f}/m²)")

        # ====================================================================
        # DISPLAY: TOU ENERGY SAVINGS & LIFECYCLE PAYBACK
        # ====================================================================
        savings_results = None
        payback_results = None

        if enable_savings and cost_results:
            st.header("⚡ Energy Savings & Lifecycle Payback Analysis")

            savings_results = compute_tou_energy_savings(
                energy_per_area_j_m2, wall_area_m2,
                annual_cooling_days=annual_cooling_days,
                cooling_cop=cooling_cop,
                load_reduction_percent=load_reduction_percent,
                peak_rate_per_kwh=peak_rate_per_kwh,
                offpeak_rate_per_kwh=offpeak_rate_per_kwh,
                pct_load_shifted_to_offpeak=pct_load_shifted,
                peak_demand_reduction_kw=peak_demand_reduction_kw,
                demand_charge_per_kw_month=demand_charge_per_kw_month,
                annual_heating_days=annual_heating_days,
                heating_efficiency=heating_efficiency,
                heating_energy_cost_per_kwh=heating_energy_cost_per_kwh
            )

            if savings_results:
                net_project_cost = cost_results['total_cost'] - hvac_downsizing_savings
                payback_results = compute_lifecycle_payback(
                    max(net_project_cost, 0),
                    savings_results['annual_total_savings']
                )

                # --- Savings breakdown ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Cooling TOU Savings",
                        f"${savings_results['cooling_tou_savings']:,.2f}/yr",
                        help="Savings from shifting cooling load to off-peak hours"
                    )
                    st.metric(
                        "Cooling kWh Shifted",
                        f"{savings_results['cooling_shifted_elec_kwh']:,.0f} kWh/yr"
                    )

                with col2:
                    st.metric(
                        "Heating TOU Savings",
                        f"${savings_results['heating_tou_savings']:,.2f}/yr",
                        help="Savings from shifting heating load to off-peak hours"
                    )
                    st.metric(
                        "Heating kWh Shifted",
                        f"{savings_results['heating_shifted_elec_kwh']:,.0f} kWh/yr"
                    )

                with col3:
                    st.metric(
                        "Demand Charge Savings",
                        f"${savings_results['annual_demand_savings']:,.2f}/yr",
                        help="Annual savings from reducing peak kW demand charges"
                    )
                    st.metric(
                        "Total Annual Savings",
                        f"${savings_results['annual_total_savings']:,.2f}/yr",
                        help="Sum of TOU + demand charge savings across both seasons"
                    )

                st.markdown("---")

                # --- Waterfall chart ---
                fig_wf = create_savings_waterfall_chart(savings_results)
                st.pyplot(fig_wf)
                plt.close(fig_wf)

                st.markdown("---")

                # --- Lifecycle payback ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    pb = payback_results['payback_years']
                    st.metric(
                        "Simple Payback Period",
                        f"{pb:.1f} years" if pb < float('inf') else "∞ years",
                        help="Time to recover net project cost from annual savings"
                    )

                with col2:
                    st.metric(
                        "25-Year Lifecycle ROI",
                        f"{payback_results['roi_25yr']:.1f}%",
                        help="Return on investment over 25 years — a standard horizon for building systems"
                    )

                with col3:
                    st.metric(
                        "50-Year Lifecycle ROI",
                        f"{payback_results['roi_50yr']:.1f}%",
                        help="Return on investment over 50 years — appropriate for passive building materials"
                    )

                cumulative_25 = savings_results['annual_total_savings'] * 25
                cumulative_50 = savings_results['annual_total_savings'] * 50

                st.info(
                    f"📊 **Lifecycle Value:**  "
                    f"25-yr cumulative savings = **${cumulative_25:,.0f}**  |  "
                    f"50-yr cumulative savings = **${cumulative_50:,.0f}**  |  "
                    f"Net project cost = **${max(net_project_cost, 0):,.0f}**"
                )

        # ====================================================================
        # DISPLAY: MATERIAL COMPARISON
        # ====================================================================
        if enable_comparison:
            st.header("🧱 Alternative Material Comparison")
            comparison_results = compute_alternative_material_comparison(
                energy_per_area_j_m2, wall_area_m2
            )

            if comparison_results and cost_results:
                comparison_data = {
                    'Material': ['Basalt PCM', 'EPS Insulation', 'Concrete Thermal Mass'],
                    'Thickness (mm)': [
                        f"{thickness_mm:.1f}",
                        f"{comparison_results['eps']['thickness_mm']:.1f}",
                        f"{comparison_results['concrete']['thickness_mm']:.1f}"
                    ],
                    'Cost ($/m²)': [
                        f"${cost_results['cost_per_m2']:.2f}",
                        f"${comparison_results['eps']['cost_per_m2']:.2f}",
                        f"${comparison_results['concrete']['cost_per_m2']:.2f}"
                    ],
                    'Total Cost ($)': [
                        f"${cost_results['total_cost']:.2f}",
                        f"${comparison_results['eps']['total_cost']:.2f}",
                        f"${comparison_results['concrete']['total_cost']:.2f}"
                    ],
                    'Performance Note': [
                        'Latent heat storage — peak load shifting',
                        comparison_results['eps']['performance_note'],
                        comparison_results['concrete']['performance_note']
                    ]
                }
                df_comparison = pd.DataFrame(comparison_data)
                st.table(df_comparison)
                st.info(
                    "**Comparison Notes:** PCM excels at peak load shifting via latent heat. "
                    "EPS reduces steady-state heat transfer. Concrete delays heat transfer via "
                    "sensible heat storage. Each material suits different thermal strategies."
                )

        # ====================================================================
        # DISPLAY: FEASIBILITY ASSESSMENT
        # ====================================================================
        st.header("⚠️ Feasibility Assessment")

        feasibility_warnings = check_feasibility(
            thickness_mm,
            cost_results['cost_per_m2'] if cost_results else 0,
            payback_results['payback_years'] if payback_results else float('inf'),
            pcm_mass_fraction if use_composite else None
        )

        if feasibility_warnings:
            for w in feasibility_warnings:
                if w['level'] == 'error':
                    st.error(w['message'])
                else:
                    st.warning(w['message'])
        else:
            st.success("✅ No major feasibility concerns identified. Project appears viable.")

        # ====================================================================
        # EXPLANATION AND FORMULAS
        # ====================================================================
        st.header("📚 Model Explanation")

        st.markdown(
            "This calculator uses a simplified energy-balance model to size PCM requirements "
            "and a realistic **Time-of-Use (TOU) economic model** to value those savings."
        )

        with st.expander("View Detailed Formulas and Assumptions", expanded=False):
            st.markdown("""
### Calculation Steps

**Step 1 – Energy to Buffer**

`Energy per Area (J/m²) = U-value × ΔT × Duration (s)`

**Step 2 – PCM Effective Energy Capacity**

`Effective Energy (kJ/kg) = Latent Heat + Cp × Usable Temp Swing`

**Step 3 – PCM Mass Required**

`PCM Mass per Area (kg/m²) = [Energy per Area (kJ/m²) / Effective Energy] × Safety Factor`

**Step 4 – Cost Model**

`Gross Project Cost = (Material × waste factor) + Installation + Labour + Transport`

`Net Project Cost = Gross − HVAC Downsizing Savings`

**Step 5 – TOU Savings Model (v3.0)**

The PCM does **not** reduce total energy consumed; it shifts *when* energy is consumed.

*Cooling TOU savings:*
```
Cooling kWh Shifted = (Daily Thermal × Cooling Days × Load Reduction% × Shifted%) / COP
Cooling Savings ($/yr) = kWh Shifted × (Peak Rate − Off-Peak Rate)
```

*Heating TOU savings:*
```
Heating kWh Shifted = (Daily Thermal × Heating Days × Load Reduction% × Shifted%) / η
Heating Savings ($/yr) = kWh Shifted × (Peak Rate − Off-Peak Rate)
```

*Demand charge savings:*
```
Demand Savings ($/yr) = Peak kW Reduction × Demand Charge ($/kW/mo) × Cooling Months
```

`Total Annual Savings = Cooling TOU + Heating TOU + Demand Savings`

**Step 6 – Lifecycle Payback**

`Simple Payback (yrs) = Net Project Cost / Annual Savings`

`25-yr ROI (%) = [(25 × Annual Savings) − Net Cost] / Net Cost × 100`

`50-yr ROI (%) = [(50 × Annual Savings) − Net Cost] / Net Cost × 100`

### Key Assumptions

1. Steady-state ΔT over the buffering period
2. Linear cost scaling with quantity
3. Constant COP / heating efficiency (no part-load correction)
4. No performance degradation over service life
5. TOU rate differential is the primary economic driver
6. HVAC downsizing savings are a one-time upfront capital offset
""")

        # ====================================================================
        # SENSITIVITY PLOTS
        # ====================================================================
        st.header("📊 Sensitivity Analysis")

        st.subheader("PCM Requirement vs. Buffering Duration")
        fig1 = create_sensitivity_plot_duration(
            u_value_w_m2_k, delta_t_k, latent_heat_kj_kg,
            cp_pcm_kj_kg_k, usable_temp_swing_k, safety_factor, wall_area_ft2
        )
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption(
            f"U-value = {u_value_w_m2_k:.2f} W/(m²·K), ΔT = {delta_t_k:.2f} K"
        )

        st.subheader("PCM Requirement vs. Temperature Difference")
        fig2 = create_sensitivity_plot_delta_t(
            u_value_w_m2_k, duration_hours, latent_heat_kj_kg,
            cp_pcm_kj_kg_k, usable_temp_swing_k, safety_factor, wall_area_ft2
        )
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption(
            f"U-value = {u_value_w_m2_k:.2f} W/(m²·K), Duration = {duration_hours:.1f} hrs"
        )

        # ====================================================================
        # CSV EXPORT
        # ====================================================================
        st.header("💾 Export Results")

        inputs_dict = {
            'Wall Area (ft²)': wall_area_ft2,
            'Wall Area (m²)': wall_area_m2,
            'Climate Type': climate_type,
            'Duration (hours)': duration_hours,
            'Temperature Difference (K)': delta_t_k,
            'U-value (W/(m²·K))': u_value_w_m2_k,
            'Safety Factor': safety_factor,
            'Latent Heat (kJ/kg)': latent_heat_kj_kg,
            'Specific Heat (kJ/(kg·K))': cp_pcm_kj_kg_k,
            'Usable Temp Swing (K)': usable_temp_swing_k,
            'PCM Density (kg/m³)': pcm_density_kg_m3
        }

        outputs_dict = {
            'PCM Required (lb/ft²)': f"{pcm_mass_per_area_lb_ft2:.4f}",
            'PCM Required (kg/m²)': f"{pcm_mass_per_area_kg_m2:.4f}",
            'Total PCM (lb)': f"{total_pcm_lb:.2f}",
            'Total PCM (kg)': f"{total_pcm_kg:.2f}",
            'PCM Volume (liters)': f"{volume_liters:.3f}",
            'Equivalent Thickness (mm)': f"{thickness_mm:.3f}",
        }

        if cost_results:
            inputs_dict.update({
                'PCM Cost ($/kg)': pcm_cost_per_kg,
                'Installation Cost ($/m²)': installation_cost_per_m2,
                'Labor Markup (%)': labor_markup_percent,
                'Transport Cost ($)': transport_cost,
                'Waste Factor (%)': waste_factor_percent,
                'HVAC Downsizing Savings ($)': hvac_downsizing_savings
            })
            outputs_dict.update({
                'Gross Project Cost ($)': f"{cost_results['total_cost']:.2f}",
                'Net Project Cost ($)': f"{max(cost_results['total_cost'] - hvac_downsizing_savings, 0):.2f}",
                'Cost per m² ($/m²)': f"{cost_results['cost_per_m2']:.2f}"
            })

        if savings_results and payback_results:
            inputs_dict.update({
                'Peak Rate ($/kWh)': peak_rate_per_kwh,
                'Off-Peak Rate ($/kWh)': offpeak_rate_per_kwh,
                '% Load Shifted to Off-Peak': pct_load_shifted,
                'Demand Charge ($/kW/month)': demand_charge_per_kw_month,
                'Cooling COP': cooling_cop,
                'Heating Efficiency': heating_efficiency,
                'Annual Cooling Days': annual_cooling_days,
                'Annual Heating Days': annual_heating_days
            })
            outputs_dict.update({
                'Cooling TOU Savings ($/yr)': f"{savings_results['cooling_tou_savings']:.2f}",
                'Heating TOU Savings ($/yr)': f"{savings_results['heating_tou_savings']:.2f}",
                'Demand Charge Savings ($/yr)': f"{savings_results['annual_demand_savings']:.2f}",
                'Total Annual Savings ($/yr)': f"{savings_results['annual_total_savings']:.2f}",
                'Simple Payback (years)': f"{payback_results['payback_years']:.1f}",
                '25-Year ROI (%)': f"{payback_results['roi_25yr']:.1f}",
                '50-Year ROI (%)': f"{payback_results['roi_50yr']:.1f}"
            })

        csv_data = generate_csv_summary(inputs_dict, outputs_dict)

        st.download_button(
            label="📥 Download Complete Summary as CSV",
            data=csv_data,
            file_name="pcm_calculator_professional_summary_v3.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"""
❌ **Calculation Error**

An error occurred: `{str(e)}`

Please check your input values and try again.
""")
        st.exception(e)

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;'>"
        "<p><strong>Basalt-PCM Thermal Buffer Sizing Calculator — Professional Edition v3.0</strong></p>"
        "<p>TOU economic model · Year-round savings · HVAC downsizing offsets · 50-year lifecycle ROI</p>"
        "<p><em>Estimates only. Consult thermal engineering and financial professionals for critical applications.</em></p>"
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
