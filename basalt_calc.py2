"""
Basalt-PCM Thermal Buffer Sizing Calculator - Professional Edition
A Streamlit app for calculating PCM requirements for thermal buffering in walls.

Features:
- Core PCM sizing calculations
- Economic analysis and cost estimation
- Energy savings and payback analysis
- Climate presets
- Material comparison mode
- Feasibility warnings
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from io import StringIO


# ============================================================================
# UNIT CONVERSION FUNCTIONS
# ============================================================================

def ft2_to_m2(area_ft2):
    """Convert square feet to square meters."""
    return area_ft2 * 0.09290304


def m2_to_ft2(area_m2):
    """Convert square meters to square feet."""
    return area_m2 / 0.09290304


def kg_to_lb(mass_kg):
    """Convert kilograms to pounds."""
    return mass_kg * 2.20462262


def lb_to_kg(mass_lb):
    """Convert pounds to kilograms."""
    return mass_lb / 2.20462262


def btu_hr_ft2_f_to_w_m2_k(u_imperial):
    """Convert U-value from BTU/(hr·ft²·°F) to W/(m²·K)."""
    return u_imperial * 5.678263


def w_m2_k_to_btu_hr_ft2_f(u_metric):
    """Convert U-value from W/(m²·K) to BTU/(hr·ft²·°F)."""
    return u_metric / 5.678263


def celsius_to_fahrenheit_delta(delta_c):
    """Convert temperature difference from Celsius to Fahrenheit."""
    return delta_c * 1.8


def fahrenheit_to_celsius_delta(delta_f):
    """Convert temperature difference from Fahrenheit to Celsius."""
    return delta_f / 1.8


def liters_to_cubic_inches(liters):
    """Convert liters to cubic inches."""
    return liters * 61.0237


def m3_to_liters(volume_m3):
    """Convert cubic meters to liters."""
    return volume_m3 * 1000


def mm_to_inches(mm):
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
        'description': 'Desert climates with high diurnal temperature swings'
    },
    'Tropical-Humid': {
        'delta_t_c': 8.0,
        'delta_t_f': 14.4,
        'duration_hours': 6.0,
        'cooling_days': 300,
        'description': 'Hot and humid with moderate daily temperature variation'
    },
    'Temperate': {
        'delta_t_c': 12.0,
        'delta_t_f': 21.6,
        'duration_hours': 8.0,
        'cooling_days': 120,
        'description': 'Moderate climate with seasonal variations'
    },
    'Cold': {
        'delta_t_c': 15.0,
        'delta_t_f': 27.0,
        'duration_hours': 12.0,
        'cooling_days': 60,
        'description': 'Cold climates with heating focus'
    }
}


# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================

def compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration_hours):
    """
    Calculate the energy per unit area that needs to be buffered.
    
    Parameters:
    -----------
    u_value_w_m2_k : float
        Overall heat transfer coefficient in W/(m²·K)
    delta_t_k : float
        Temperature difference in Kelvin (or Celsius difference)
    duration_hours : float
        Duration of buffering in hours
    
    Returns:
    --------
    float
        Energy per unit area in J/m²
    """
    if u_value_w_m2_k <= 0 or delta_t_k <= 0 or duration_hours <= 0:
        raise ValueError("All parameters must be positive values")
    
    duration_seconds = duration_hours * 3600
    heat_flux = u_value_w_m2_k * delta_t_k  # W/m²
    energy_per_area = heat_flux * duration_seconds  # J/m²
    return energy_per_area


def compute_effective_energy_per_kg(latent_heat_kj_kg, cp_kj_kg_k, usable_temp_swing_k):
    """
    Calculate the effective energy storage capacity of PCM per kilogram.
    
    Parameters:
    -----------
    latent_heat_kj_kg : float
        Latent heat of fusion in kJ/kg
    cp_kj_kg_k : float
        Specific heat capacity of PCM in kJ/(kg·K)
    usable_temp_swing_k : float
        Usable temperature swing around melting point in K
    
    Returns:
    --------
    float
        Effective energy per kg in kJ/kg
    """
    if latent_heat_kj_kg <= 0 or cp_kj_kg_k <= 0 or usable_temp_swing_k <= 0:
        raise ValueError("All parameters must be positive values")
    
    effective_energy = latent_heat_kj_kg + (cp_kj_kg_k * usable_temp_swing_k)
    return effective_energy


def compute_pcm_mass_per_area(energy_per_area_j_m2, effective_energy_kj_kg, safety_factor):
    """
    Calculate the required PCM mass per unit area.
    
    Parameters:
    -----------
    energy_per_area_j_m2 : float
        Energy to be buffered per unit area in J/m²
    effective_energy_kj_kg : float
        Effective energy storage capacity in kJ/kg
    safety_factor : float
        Safety factor multiplier (typically 1.0-2.0)
    
    Returns:
    --------
    float
        PCM mass per unit area in kg/m²
    """
    if energy_per_area_j_m2 <= 0 or effective_energy_kj_kg <= 0 or safety_factor <= 0:
        raise ValueError("All parameters must be positive values")
    
    energy_per_area_kj_m2 = energy_per_area_j_m2 / 1000
    pcm_mass_per_area = (energy_per_area_kj_m2 / effective_energy_kj_kg) * safety_factor
    return pcm_mass_per_area


def compute_pcm_volume_and_thickness(pcm_mass_kg, pcm_density_kg_m3, wall_area_m2):
    """
    Calculate PCM volume and equivalent uniform thickness.
    
    Parameters:
    -----------
    pcm_mass_kg : float
        Total PCM mass in kg
    pcm_density_kg_m3 : float
        PCM density in kg/m³
    wall_area_m2 : float
        Wall area in m²
    
    Returns:
    --------
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


def compute_composite_requirements(pcm_mass_kg, pcm_mass_fraction_percent, composite_density_kg_m3=None):
    """
    Calculate composite material requirements.
    
    Parameters:
    -----------
    pcm_mass_kg : float
        Required PCM mass in kg
    pcm_mass_fraction_percent : float
        PCM mass fraction in composite (percent)
    composite_density_kg_m3 : float, optional
        Composite density in kg/m³
    
    Returns:
    --------
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

def compute_cost_analysis(pcm_mass_kg, wall_area_m2, pcm_cost_per_kg, 
                         installation_cost_per_m2, labor_markup_percent,
                         transport_cost, waste_factor_percent,
                         use_composite=False, composite_mass_kg=None,
                         composite_cost_per_kg=None):
    """
    Calculate comprehensive cost analysis for PCM installation.
    
    Parameters:
    -----------
    pcm_mass_kg : float
        Total PCM mass in kg
    wall_area_m2 : float
        Wall area in m²
    pcm_cost_per_kg : float
        Cost of PCM per kilogram ($/kg)
    installation_cost_per_m2 : float
        Installation cost per square meter ($/m²)
    labor_markup_percent : float
        Labor markup as percentage
    transport_cost : float
        Flat transport cost ($)
    waste_factor_percent : float
        Waste factor as percentage
    use_composite : bool
        Whether using composite material
    composite_mass_kg : float, optional
        Total composite mass if using composite
    composite_cost_per_kg : float, optional
        Cost per kg of composite material
    
    Returns:
    --------
    dict
        Dictionary containing all cost components
    """
    try:
        # Calculate raw material cost
        if use_composite and composite_mass_kg and composite_cost_per_kg:
            raw_material_cost = composite_mass_kg * composite_cost_per_kg
        else:
            raw_material_cost = pcm_mass_kg * pcm_cost_per_kg
        
        # Apply waste factor
        waste_multiplier = 1 + (waste_factor_percent / 100)
        adjusted_material_cost = raw_material_cost * waste_multiplier
        
        # Calculate installation cost
        base_installation_cost = wall_area_m2 * installation_cost_per_m2
        
        # Calculate labor cost with markup
        labor_cost = base_installation_cost * (labor_markup_percent / 100)
        
        # Calculate total project cost
        total_cost = adjusted_material_cost + base_installation_cost + labor_cost + transport_cost
        
        # Calculate cost per area
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


def compute_energy_savings(energy_per_area_j_m2, wall_area_m2, 
                          cooling_days_per_year, load_reduction_percent,
                          electricity_cost_per_kwh, cooling_cop):
    """
    Calculate annual energy savings and payback period.
    
    Parameters:
    -----------
    energy_per_area_j_m2 : float
        Daily energy buffered per area (J/m²)
    wall_area_m2 : float
        Wall area (m²)
    cooling_days_per_year : int
        Number of cooling days per year
    load_reduction_percent : float
        Estimated cooling load reduction (%)
    electricity_cost_per_kwh : float
        Electricity cost ($/kWh)
    cooling_cop : float
        Coefficient of Performance for cooling system
    
    Returns:
    --------
    dict
        Dictionary containing savings calculations
    """
    try:
        # Total daily thermal energy buffered (J)
        daily_thermal_energy_j = energy_per_area_j_m2 * wall_area_m2
        
        # Convert to kWh (1 kWh = 3.6e6 J)
        daily_thermal_energy_kwh = daily_thermal_energy_j / 3.6e6
        
        # Annual thermal energy buffered (kWh)
        annual_thermal_energy_kwh = daily_thermal_energy_kwh * cooling_days_per_year
        
        # Apply load reduction factor
        effective_thermal_reduction_kwh = annual_thermal_energy_kwh * (load_reduction_percent / 100)
        
        # Electrical energy saved (accounting for COP)
        # For cooling: electrical_energy = thermal_energy / COP
        if cooling_cop > 0:
            annual_electrical_savings_kwh = effective_thermal_reduction_kwh / cooling_cop
        else:
            annual_electrical_savings_kwh = 0
        
        # Annual cost savings
        annual_cost_savings = annual_electrical_savings_kwh * electricity_cost_per_kwh
        
        # 10-year savings
        ten_year_savings = annual_cost_savings * 10
        
        return {
            'daily_thermal_energy_kwh': daily_thermal_energy_kwh,
            'annual_thermal_energy_kwh': annual_thermal_energy_kwh,
            'effective_thermal_reduction_kwh': effective_thermal_reduction_kwh,
            'annual_electrical_savings_kwh': annual_electrical_savings_kwh,
            'annual_cost_savings': annual_cost_savings,
            'ten_year_savings': ten_year_savings
        }
    except Exception as e:
        st.error(f"Error in energy savings calculation: {str(e)}")
        return None


def compute_payback(total_cost, annual_savings):
    """
    Calculate simple payback period and ROI.
    
    Parameters:
    -----------
    total_cost : float
        Total project cost ($)
    annual_savings : float
        Annual cost savings ($/year)
    
    Returns:
    --------
    dict
        Dictionary containing payback and ROI
    """
    try:
        if annual_savings > 0:
            payback_years = total_cost / annual_savings
            ten_year_roi = ((annual_savings * 10) - total_cost) / total_cost * 100
        else:
            payback_years = float('inf')
            ten_year_roi = -100
        
        return {
            'payback_years': payback_years,
            'ten_year_roi': ten_year_roi
        }
    except Exception as e:
        st.error(f"Error in payback calculation: {str(e)}")
        return None


# ============================================================================
# MATERIAL COMPARISON FUNCTIONS
# ============================================================================

def compute_alternative_material_comparison(energy_per_area_j_m2, wall_area_m2):
    """
    Compare PCM with alternative materials (EPS insulation and concrete thermal mass).
    
    Parameters:
    -----------
    energy_per_area_j_m2 : float
        Energy to be buffered per area (J/m²)
    wall_area_m2 : float
        Wall area (m²)
    
    Returns:
    --------
    dict
        Comparison data for alternative materials
    """
    try:
        # EPS Insulation properties
        # R-value per inch: ~4.2 (ft²·°F·h/BTU)
        # Density: ~15 kg/m³
        # Cost: ~$8/m² per inch
        eps_r_per_mm = 0.037  # m²·K/W per mm
        eps_cost_per_m2_per_mm = 0.315  # $/m² per mm
        
        # Estimate required R-value for equivalent energy reduction
        # Simplified: higher R-value reduces heat transfer
        # Assume we need to reduce U-value proportionally
        target_r_value = 2.0  # m²·K/W (example target)
        eps_thickness_mm = target_r_value / eps_r_per_mm
        eps_cost_per_m2 = eps_thickness_mm * eps_cost_per_m2_per_mm
        eps_total_cost = eps_cost_per_m2 * wall_area_m2
        
        # Concrete Thermal Mass properties
        # Specific heat: ~0.88 kJ/(kg·K)
        # Density: ~2400 kg/m³
        # Cost: ~$100/m³
        concrete_cp = 0.88  # kJ/(kg·K)
        concrete_density = 2400  # kg/m³
        concrete_cost_per_m3 = 100  # $
        
        # Required concrete mass for equivalent energy storage
        # E = m × cp × ΔT (assume ΔT = 10K for sensible storage)
        delta_t_concrete = 10  # K
        energy_per_area_kj_m2 = energy_per_area_j_m2 / 1000
        concrete_mass_per_m2 = energy_per_area_kj_m2 / (concrete_cp * delta_t_concrete)
        concrete_thickness_m = concrete_mass_per_m2 / concrete_density
        concrete_thickness_mm = concrete_thickness_m * 1000
        concrete_volume_per_m2 = concrete_thickness_m
        concrete_cost_per_m2 = concrete_volume_per_m2 * concrete_cost_per_m3
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

def check_feasibility(thickness_mm, cost_per_m2, payback_years, 
                     pcm_mass_fraction_percent=None):
    """
    Check project feasibility and generate warnings.
    
    Parameters:
    -----------
    thickness_mm : float
        Required PCM thickness (mm)
    cost_per_m2 : float
        Cost per square meter ($/m²)
    payback_years : float
        Payback period (years)
    pcm_mass_fraction_percent : float, optional
        PCM mass fraction in composite (%)
    
    Returns:
    --------
    list
        List of warning dictionaries with 'level' and 'message'
    """
    warnings = []
    
    # Thickness warning
    if thickness_mm > 50:
        warnings.append({
            'level': 'warning',
            'message': f"⚠️ Required thickness ({thickness_mm:.1f} mm) exceeds 50 mm. This may present installation challenges and require structural modifications."
        })
    
    # Cost warning
    typical_insulation_cost = 40  # $/m²
    if cost_per_m2 > typical_insulation_cost:
        warnings.append({
            'level': 'warning',
            'message': f"⚠️ Cost per m² (${cost_per_m2:.2f}) exceeds typical insulation cost (${typical_insulation_cost}/m²). Consider cost-effectiveness."
        })
    
    # Payback warning
    if payback_years > 15:
        warnings.append({
            'level': 'error',
            'message': f"🚨 Payback period ({payback_years:.1f} years) exceeds 15 years. Project may not be economically viable."
        })
    elif payback_years > 10:
        warnings.append({
            'level': 'warning',
            'message': f"⚠️ Payback period ({payback_years:.1f} years) is quite long. Verify energy savings assumptions."
        })
    
    # Composite fraction warning
    if pcm_mass_fraction_percent and pcm_mass_fraction_percent > 70:
        warnings.append({
            'level': 'warning',
            'message': f"⚠️ PCM mass fraction ({pcm_mass_fraction_percent:.0f}%) exceeds 70%. This may affect structural integrity and workability of the composite."
        })
    
    return warnings


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_inputs(u_value, delta_t, duration, latent_heat, cp, usable_swing, density, safety_factor):
    """
    Validate user inputs and return warnings if any.
    
    Parameters:
    -----------
    u_value : float
        U-value in W/(m²·K)
    delta_t : float
        Temperature difference in K
    duration : float
        Duration in hours
    latent_heat : float
        Latent heat in kJ/kg
    cp : float
        Specific heat in kJ/(kg·K)
    usable_swing : float
        Temperature swing in K
    density : float
        Density in kg/m³
    safety_factor : float
        Safety factor multiplier
    
    Returns:
    --------
    list
        List of warning messages
    """
    warnings = []
    
    if u_value <= 0:
        warnings.append("U-value must be positive.")
    if u_value > 10:
        warnings.append("U-value > 10 W/(m²·K) is unusually high. Check your input.")
    if delta_t <= 0.1:
        warnings.append("Temperature difference is very small. Results may be near zero.")
    if delta_t > 50:
        warnings.append("Temperature difference > 50K is very large. Verify your input.")
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
        warnings.append("Warning: Usable temperature swing > 20K is unusually large for typical PCMs.")
    if density <= 0:
        warnings.append("PCM density must be positive.")
    if density < 100:
        warnings.append("Density < 100 kg/m³ is very low. Verify your input.")
    if density > 3000:
        warnings.append("Density > 3000 kg/m³ is very high. Verify your input.")
    if safety_factor < 1.0:
        warnings.append("Safety factor should typically be >= 1.0.")
    if safety_factor > 3.0:
        warnings.append("Safety factor > 3.0 is very conservative. Results will be significantly oversized.")
    
    return warnings


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_sensitivity_plot_duration(u_value_w_m2_k, delta_t_k, latent_heat, cp, usable_swing, 
                                     safety_factor, wall_area_ft2, duration_range=None):
    """
    Create sensitivity plot of PCM requirement vs. duration.
    
    Parameters:
    -----------
    u_value_w_m2_k : float
        U-value in W/(m²·K)
    delta_t_k : float
        Temperature difference in K
    latent_heat : float
        Latent heat in kJ/kg
    cp : float
        Specific heat in kJ/(kg·K)
    usable_swing : float
        Temperature swing in K
    safety_factor : float
        Safety factor
    wall_area_ft2 : float
        Wall area in ft²
    duration_range : array-like, optional
        Custom duration range for plotting
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    try:
        if duration_range is None:
            duration_range = np.linspace(1, 24, 50)
        
        pcm_lb_ft2_values = []
        
        for duration in duration_range:
            try:
                energy_per_area = compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration)
                effective_energy = compute_effective_energy_per_kg(latent_heat, cp, usable_swing)
                pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(energy_per_area, effective_energy, safety_factor)
                
                # Convert to lb/ft²
                pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
                pcm_lb_ft2_values.append(pcm_mass_per_area_lb_ft2)
            except Exception as e:
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
        # Return a blank figure with error message if plotting fails
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig


def create_sensitivity_plot_delta_t(u_value_w_m2_k, duration_hours, latent_heat, cp, usable_swing,
                                     safety_factor, wall_area_ft2, delta_t_range=None):
    """
    Create sensitivity plot of PCM requirement vs. temperature difference.
    
    Parameters:
    -----------
    u_value_w_m2_k : float
        U-value in W/(m²·K)
    duration_hours : float
        Duration in hours
    latent_heat : float
        Latent heat in kJ/kg
    cp : float
        Specific heat in kJ/(kg·K)
    usable_swing : float
        Temperature swing in K
    safety_factor : float
        Safety factor
    wall_area_ft2 : float
        Wall area in ft²
    delta_t_range : array-like, optional
        Custom temperature range for plotting
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    try:
        if delta_t_range is None:
            delta_t_range = np.linspace(1, 30, 50)
        
        pcm_lb_ft2_values = []
        
        for delta_t in delta_t_range:
            try:
                energy_per_area = compute_energy_per_area(u_value_w_m2_k, delta_t, duration_hours)
                effective_energy = compute_effective_energy_per_kg(latent_heat, cp, usable_swing)
                pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(energy_per_area, effective_energy, safety_factor)
                
                # Convert to lb/ft²
                pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
                pcm_lb_ft2_values.append(pcm_mass_per_area_lb_ft2)
            except Exception as e:
                pcm_lb_ft2_values.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(delta_t_range, pcm_lb_ft2_values, linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Temperature Difference (°C or K)', fontsize=12)
        ax.set_ylabel('PCM Required (lb/ft²)', fontsize=12)
        ax.set_title('PCM Requirement vs. Temperature Difference', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(delta_t_range[0], delta_t_range[-1])
        
        return fig
    except Exception as e:
        # Return a blank figure with error message if plotting fails
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig


# ============================================================================
# CSV EXPORT FUNCTION
# ============================================================================

def generate_csv_summary(inputs, outputs):
    """
    Generate a CSV summary of inputs and outputs using proper CSV formatting.
    
    Parameters:
    -----------
    inputs : dict
        Dictionary of input parameters
    outputs : dict
        Dictionary of output results
    
    Returns:
    --------
    str
        Properly formatted CSV string
    """
    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    
    # Add header
    writer.writerow(['Basalt-PCM Thermal Buffer Sizing Calculator - Professional Edition'])
    writer.writerow([])
    
    # Add inputs
    writer.writerow(['INPUTS'])
    for key, value in inputs.items():
        writer.writerow([key, value])
    writer.writerow([])
    
    # Add outputs
    writer.writerow(['OUTPUTS'])
    for key, value in outputs.items():
        writer.writerow([key, value])
    
    return output.getvalue()


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main function to run the Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title="PCM Thermal Buffer Calculator - Professional Edition",
        page_icon="🔥",
        layout="wide"
    )
    
    # Title and description
    st.title("🔥 Basalt-PCM Thermal Buffer Sizing Calculator - Professional Edition")
    st.markdown("""
    This application calculates the required Phase Change Material (PCM) mass needed to buffer 
    temperature fluctuations in a wall assembly, including economic analysis, energy savings projections,
    and material comparisons.
    """)
    
    # Sidebar for inputs
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
    else:
        preset_delta_t_c = None
        preset_delta_t_f = None
        preset_duration = None
        preset_cooling_days = None
    
    # ========================================================================
    # SECTION A: Geometry and Design Target
    # ========================================================================
    st.sidebar.subheader("A. Geometry and Design Target")
    
    wall_area_ft2 = st.sidebar.number_input(
        "Wall Area (ft²)",
        min_value=0.1,
        value=100.0,
        step=1.0,
        help="Total wall area to be analyzed"
    )
    
    units_system = st.sidebar.radio(
        "Preferred Units",
        options=["Imperial", "Metric"],
        help="Primary unit system for inputs (outputs always include both)"
    )
    
    # Duration input (with climate preset option)
    if preset_duration is not None:
        duration_hours = st.sidebar.number_input(
            "Duration of Buffering (hours)",
            min_value=0.1,
            value=preset_duration,
            step=0.5,
            help=f"Climate preset suggests {preset_duration} hours"
        )
    else:
        duration_hours = st.sidebar.number_input(
            "Duration of Buffering (hours)",
            min_value=0.1,
            value=8.0,
            step=0.5,
            help="Time period over which thermal buffering is required"
        )
    
    # Temperature difference input (with climate preset option)
    if units_system == "Imperial":
        if preset_delta_t_f is not None:
            delta_t_input = st.sidebar.number_input(
                "Temperature Difference (°F)",
                min_value=0.1,
                value=preset_delta_t_f,
                step=1.0,
                help=f"Climate preset suggests {preset_delta_t_f}°F"
            )
        else:
            delta_t_input = st.sidebar.number_input(
                "Temperature Difference (°F)",
                min_value=0.1,
                value=20.0,
                step=1.0,
                help="Temperature difference between outside and inside"
            )
        delta_t_k = fahrenheit_to_celsius_delta(delta_t_input)
    else:
        if preset_delta_t_c is not None:
            delta_t_input = st.sidebar.number_input(
                "Temperature Difference (°C)",
                min_value=0.1,
                value=preset_delta_t_c,
                step=0.5,
                help=f"Climate preset suggests {preset_delta_t_c}°C"
            )
        else:
            delta_t_input = st.sidebar.number_input(
                "Temperature Difference (°C)",
                min_value=0.1,
                value=11.1,
                step=0.5,
                help="Temperature difference between outside and inside"
            )
        delta_t_k = delta_t_input
    
    # U-value input
    u_value_units = st.sidebar.selectbox(
        "U-value Units",
        options=["W/(m²·K)", "BTU/(hr·ft²·°F)"],
        help="Select units for thermal transmittance input"
    )
    
    if u_value_units == "W/(m²·K)":
        u_value_input = st.sidebar.number_input(
            "U-value (W/(m²·K))",
            min_value=0.01,
            value=0.5,
            step=0.1,
            help="Overall heat transfer coefficient"
        )
        u_value_w_m2_k = u_value_input
    else:
        u_value_input = st.sidebar.number_input(
            "U-value (BTU/(hr·ft²·°F))",
            min_value=0.01,
            value=0.088,
            step=0.01,
            help="Overall heat transfer coefficient"
        )
        u_value_w_m2_k = btu_hr_ft2_f_to_w_m2_k(u_value_input)
    
    safety_factor = st.sidebar.slider(
        "Safety Factor",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="Multiplier to account for uncertainties and ensure adequate capacity"
    )
    
    # ========================================================================
    # SECTION B: PCM Properties
    # ========================================================================
    st.sidebar.subheader("B. PCM Properties")
    
    latent_heat_kj_kg = st.sidebar.number_input(
        "Latent Heat of Fusion (kJ/kg)",
        min_value=1.0,
        value=200.0,
        step=10.0,
        help="Latent heat of phase change for the PCM"
    )
    
    cp_pcm_kj_kg_k = st.sidebar.number_input(
        "Specific Heat Capacity (kJ/(kg·K))",
        min_value=0.1,
        value=2.0,
        step=0.1,
        help="Specific heat capacity of the PCM"
    )
    
    usable_temp_swing_k = st.sidebar.number_input(
        "Usable Temperature Swing (K)",
        min_value=0.1,
        value=6.0,
        step=0.5,
        help="Effective temperature range around melting point where PCM operates"
    )
    
    pcm_density_kg_m3 = st.sidebar.number_input(
        "PCM Density (kg/m³)",
        min_value=1.0,
        value=800.0,
        step=50.0,
        help="Density of the phase change material"
    )
    
    # ========================================================================
    # SECTION C: Optional Composite Constraints
    # ========================================================================
    st.sidebar.subheader("C. Optional: Composite Constraints")
    
    use_composite = st.sidebar.checkbox(
        "Calculate Composite Requirements",
        value=False,
        help="Enable to calculate total composite material needed"
    )
    
    composite_density_kg_m3 = None
    pcm_mass_fraction = None
    
    if use_composite:
        pcm_mass_fraction = st.sidebar.number_input(
            "PCM Mass Fraction in Composite (%)",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="Percentage of PCM in the composite material by mass"
        )
        
        composite_density_kg_m3 = st.sidebar.number_input(
            "Composite Density (kg/m³)",
            min_value=1.0,
            value=1200.0,
            step=50.0,
            help="Density of the composite material"
        )
    
    # ========================================================================
    # SECTION D: Economic Analysis
    # ========================================================================
    st.sidebar.subheader("D. 💰 Economic Analysis")
    
    enable_economic = st.sidebar.checkbox(
        "Enable Cost Estimation",
        value=True,
        help="Calculate project costs and economic feasibility"
    )
    
    if enable_economic:
        pcm_cost_per_kg = st.sidebar.number_input(
            "PCM Cost ($/kg)",
            min_value=0.01,
            value=5.0,
            step=0.5,
            help="Cost per kilogram of PCM"
        )
        
        installation_cost_per_m2 = st.sidebar.number_input(
            "Installation Cost ($/m²)",
            min_value=0.0,
            value=15.0,
            step=1.0,
            help="Installation cost per square meter"
        )
        
        labor_markup_percent = st.sidebar.number_input(
            "Labor Markup (%)",
            min_value=0.0,
            value=30.0,
            step=5.0,
            help="Labor cost as percentage of installation cost"
        )
        
        transport_cost = st.sidebar.number_input(
            "Transport Cost ($)",
            min_value=0.0,
            value=200.0,
            step=50.0,
            help="Flat transport/delivery cost"
        )
        
        waste_factor_percent = st.sidebar.number_input(
            "Waste Factor (%)",
            min_value=0.0,
            value=10.0,
            step=1.0,
            help="Material waste factor as percentage"
        )
        
        if use_composite:
            composite_cost_per_kg = st.sidebar.number_input(
                "Composite Cost ($/kg)",
                min_value=0.01,
                value=3.0,
                step=0.5,
                help="Cost per kilogram of composite material"
            )
        else:
            composite_cost_per_kg = None
    
    # ========================================================================
    # SECTION E: Energy Savings & Payback
    # ========================================================================
    st.sidebar.subheader("E. ⚡ Energy Savings & Payback")
    
    enable_savings = st.sidebar.checkbox(
        "Enable Savings Analysis",
        value=True,
        help="Calculate energy savings and payback period"
    )
    
    if enable_savings:
        electricity_cost_per_kwh = st.sidebar.number_input(
            "Electricity Cost ($/kWh)",
            min_value=0.01,
            value=0.12,
            step=0.01,
            help="Cost of electricity per kilowatt-hour"
        )
        
        cooling_cop = st.sidebar.number_input(
            "Cooling COP",
            min_value=1.0,
            value=3.0,
            step=0.5,
            help="Coefficient of Performance for cooling system"
        )
        
        load_reduction_percent = st.sidebar.number_input(
            "Estimated Cooling Load Reduction (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            help="Percentage of cooling load reduced by PCM buffering"
        )
        
        # Cooling days input (with climate preset option)
        if preset_cooling_days is not None:
            annual_cooling_days = st.sidebar.number_input(
                "Annual Cooling Days",
                min_value=0,
                value=preset_cooling_days,
                step=10,
                help=f"Climate preset suggests {preset_cooling_days} cooling days"
            )
        else:
            annual_cooling_days = st.sidebar.number_input(
                "Annual Cooling Days",
                min_value=0,
                value=120,
                step=10,
                help="Number of days per year requiring cooling"
            )
    
    # ========================================================================
    # SECTION F: Material Comparison
    # ========================================================================
    st.sidebar.subheader("F. 🧱 Material Comparison")
    
    enable_comparison = st.sidebar.checkbox(
        "Enable Alternative Material Comparison",
        value=False,
        help="Compare PCM with EPS insulation and concrete thermal mass"
    )
    
    # ========================================================================
    # PERFORM CALCULATIONS
    # ========================================================================
    
    try:
        # Convert wall area to m²
        wall_area_m2 = ft2_to_m2(wall_area_ft2)
        
        # Validate inputs
        warnings = validate_inputs(
            u_value_w_m2_k, delta_t_k, duration_hours,
            latent_heat_kj_kg, cp_pcm_kj_kg_k, usable_temp_swing_k,
            pcm_density_kg_m3, safety_factor
        )
        
        # Display warnings if any
        if warnings:
            st.sidebar.warning("⚠️ Input Validation Warnings:")
            for warning in warnings:
                st.sidebar.warning(f"• {warning}")
        
        # Calculate energy per area
        energy_per_area_j_m2 = compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration_hours)
        
        # Calculate effective energy per kg
        effective_energy_kj_kg = compute_effective_energy_per_kg(
            latent_heat_kj_kg, cp_pcm_kj_kg_k, usable_temp_swing_k
        )
        
        # Calculate PCM mass per area
        pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(
            energy_per_area_j_m2, effective_energy_kj_kg, safety_factor
        )
        
        # Convert to lb/ft²
        pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
        
        # Calculate total PCM mass
        total_pcm_kg = pcm_mass_per_area_kg_m2 * wall_area_m2
        total_pcm_lb = kg_to_lb(total_pcm_kg)
        
        # Calculate volume and thickness
        volume_m3, thickness_m, volume_liters, thickness_mm = compute_pcm_volume_and_thickness(
            total_pcm_kg, pcm_density_kg_m3, wall_area_m2
        )
        
        volume_cubic_inches = liters_to_cubic_inches(volume_liters)
        thickness_inches = mm_to_inches(thickness_mm)
        
        # Calculate composite requirements if enabled
        composite_results = None
        if use_composite and pcm_mass_fraction and composite_density_kg_m3:
            composite_results = compute_composite_requirements(
                total_pcm_kg, pcm_mass_fraction, composite_density_kg_m3
            )
        
        # ========================================================================
        # DISPLAY MAIN OUTPUTS
        # ========================================================================
        
        st.header("📈 Calculation Results")
        
        # Create columns for main outputs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="PCM Required",
                value=f"{pcm_mass_per_area_lb_ft2:.3f} lb/ft²",
                help="Primary output: PCM mass per unit area in imperial units"
            )
            st.caption(f"({pcm_mass_per_area_kg_m2:.3f} kg/m²)")
        
        with col2:
            st.metric(
                label="Total PCM Mass",
                value=f"{total_pcm_lb:.1f} lb",
                help="Total PCM mass required for the entire wall"
            )
            st.caption(f"({total_pcm_kg:.1f} kg)")
        
        with col3:
            st.metric(
                label="PCM Volume",
                value=f"{volume_liters:.2f} L",
                help="Total volume of PCM required"
            )
            st.caption(f"({volume_cubic_inches:.1f} in³)")
        
        with col4:
            st.metric(
                label="Equivalent PCM Thickness",
                value=f"{thickness_mm:.2f} mm",
                help="Thickness if PCM is spread uniformly over wall area"
            )
            st.caption(f"({thickness_inches:.3f} in)")
        
        # ========================================================================
        # COMPOSITE REQUIREMENTS (if enabled)
        # ========================================================================
        
        if composite_results:
            st.subheader("🧱 Composite Material Requirements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Total Composite Mass",
                    value=f"{composite_results['composite_mass_lb']:.1f} lb",
                    help=f"Total composite mass needed to achieve {pcm_mass_fraction}% PCM fraction"
                )
                st.caption(f"({composite_results['composite_mass_kg']:.1f} kg)")
            
            with col2:
                if 'composite_volume_liters' in composite_results:
                    st.metric(
                        label="Composite Volume",
                        value=f"{composite_results['composite_volume_liters']:.2f} L",
                        help="Total volume of composite material"
                    )
                    st.caption(f"({composite_results['composite_volume_m3']:.3f} m³)")
            
            composite_mass_per_area_kg_m2 = composite_results['composite_mass_kg'] / wall_area_m2
            composite_mass_per_area_lb_ft2 = composite_results['composite_mass_lb'] / wall_area_ft2
            
            st.info(f"""
            **Composite per unit area:** {composite_mass_per_area_lb_ft2:.3f} lb/ft² 
            ({composite_mass_per_area_kg_m2:.3f} kg/m²)
            """)
        
        # ========================================================================
        # ECONOMIC ANALYSIS
        # ========================================================================
        
        cost_results = None
        if enable_economic:
            st.header("💰 Economic Analysis")
            
            # Determine which mass to use for cost calculation
            if composite_results:
                material_mass_kg = composite_results['composite_mass_kg']
                material_cost_per_kg = composite_cost_per_kg if composite_cost_per_kg else pcm_cost_per_kg
            else:
                material_mass_kg = total_pcm_kg
                material_cost_per_kg = pcm_cost_per_kg
            
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
                    st.metric(
                        label="Raw Material Cost",
                        value=f"${cost_results['raw_material_cost']:.2f}",
                        help="Base cost of materials before waste factor"
                    )
                    st.metric(
                        label="Adjusted Material Cost",
                        value=f"${cost_results['adjusted_material_cost']:.2f}",
                        help=f"Material cost with {waste_factor_percent}% waste factor"
                    )
                
                with col2:
                    st.metric(
                        label="Installation Cost",
                        value=f"${cost_results['installation_cost']:.2f}",
                        help="Base installation cost"
                    )
                    st.metric(
                        label="Labor Cost",
                        value=f"${cost_results['labor_cost']:.2f}",
                        help=f"Labor at {labor_markup_percent}% markup"
                    )
                
                with col3:
                    st.metric(
                        label="Transport Cost",
                        value=f"${cost_results['transport_cost']:.2f}",
                        help="Flat transport/delivery cost"
                    )
                    st.metric(
                        label="Total Project Cost",
                        value=f"${cost_results['total_cost']:.2f}",
                        help="Sum of all cost components"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Cost per m²",
                        value=f"${cost_results['cost_per_m2']:.2f}/m²",
                        help="Total cost per square meter"
                    )
                with col2:
                    st.metric(
                        label="Cost per ft²",
                        value=f"${cost_results['cost_per_ft2']:.2f}/ft²",
                        help="Total cost per square foot"
                    )
        
        # ========================================================================
        # ENERGY SAVINGS & PAYBACK
        # ========================================================================
        
        savings_results = None
        payback_results = None
        
        if enable_savings and cost_results:
            st.header("⚡ Energy Savings & Payback Analysis")
            
            savings_results = compute_energy_savings(
                energy_per_area_j_m2, wall_area_m2,
                annual_cooling_days, load_reduction_percent,
                electricity_cost_per_kwh, cooling_cop
            )
            
            if savings_results:
                payback_results = compute_payback(
                    cost_results['total_cost'],
                    savings_results['annual_cost_savings']
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Daily Thermal Energy Buffered",
                        value=f"{savings_results['daily_thermal_energy_kwh']:.2f} kWh",
                        help="Thermal energy buffered per day"
                    )
                    st.metric(
                        label="Annual Thermal Energy",
                        value=f"{savings_results['annual_thermal_energy_kwh']:.1f} kWh",
                        help=f"Total thermal energy buffered over {annual_cooling_days} cooling days"
                    )
                
                with col2:
                    st.metric(
                        label="Effective Load Reduction",
                        value=f"{savings_results['effective_thermal_reduction_kwh']:.1f} kWh/yr",
                        help=f"Annual thermal reduction at {load_reduction_percent}% efficiency"
                    )
                    st.metric(
                        label="Electrical Savings",
                        value=f"{savings_results['annual_electrical_savings_kwh']:.1f} kWh/yr",
                        help=f"Electrical energy saved (COP = {cooling_cop})"
                    )
                
                with col3:
                    st.metric(
                        label="Annual Cost Savings",
                        value=f"${savings_results['annual_cost_savings']:.2f}/yr",
                        help=f"Annual savings at ${electricity_cost_per_kwh}/kWh"
                    )
                    st.metric(
                        label="10-Year Savings",
                        value=f"${savings_results['ten_year_savings']:.2f}",
                        help="Total savings over 10 years"
                    )
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if payback_results['payback_years'] < float('inf'):
                        st.metric(
                            label="Simple Payback Period",
                            value=f"{payback_results['payback_years']:.1f} years",
                            help="Time to recover initial investment"
                        )
                    else:
                        st.metric(
                            label="Simple Payback Period",
                            value="∞ years",
                            help="Project does not pay back (no savings)"
                        )
                
                with col2:
                    st.metric(
                        label="10-Year ROI",
                        value=f"{payback_results['ten_year_roi']:.1f}%",
                        help="Return on investment over 10 years"
                    )
        
        # ========================================================================
        # MATERIAL COMPARISON
        # ========================================================================
        
        if enable_comparison:
            st.header("🧱 Alternative Material Comparison")
            
            comparison_results = compute_alternative_material_comparison(
                energy_per_area_j_m2, wall_area_m2
            )
            
            if comparison_results and cost_results:
                # Create comparison table
                comparison_data = {
                    'Material': ['PCM', 'EPS Insulation', 'Concrete Thermal Mass'],
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
                        'Latent heat storage (peak reduction)',
                        comparison_results['eps']['performance_note'],
                        comparison_results['concrete']['performance_note']
                    ]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.table(df_comparison)
                
                st.info("""
                **Comparison Notes:**
                - **PCM**: Provides peak load reduction through latent heat storage
                - **EPS Insulation**: Reduces steady-state heat transfer (R-value based)
                - **Concrete Thermal Mass**: Provides time-lag through sensible heat storage
                
                Each material serves different thermal management strategies and should be selected based on specific project goals.
                """)
        
        # ========================================================================
        # FEASIBILITY WARNINGS
        # ========================================================================
        
        st.header("⚠️ Feasibility Assessment")
        
        feasibility_warnings = check_feasibility(
            thickness_mm,
            cost_results['cost_per_m2'] if cost_results else 0,
            payback_results['payback_years'] if payback_results else float('inf'),
            pcm_mass_fraction if use_composite else None
        )
        
        if feasibility_warnings:
            for warning in feasibility_warnings:
                if warning['level'] == 'error':
                    st.error(warning['message'])
                else:
                    st.warning(warning['message'])
        else:
            st.success("✅ No major feasibility concerns identified. Project appears viable.")
        
        # ========================================================================
        # EXPLANATION AND FORMULAS
        # ========================================================================
        
        st.header("📚 Model Explanation")
        
        st.markdown("""
        This calculator uses a simplified energy balance model to estimate PCM requirements for thermal buffering.
        The approach assumes that the PCM absorbs and releases heat primarily through latent heat near its melting point,
        which slows temperature changes in the wall assembly.
        """)
        
        with st.expander("View Detailed Formulas and Assumptions", expanded=False):
            st.markdown("""
            ### Calculation Steps:
            
            **Step 1: Calculate Energy to be Buffered**
            
            Energy per Area (J/m²) = U-value (W/(m²·K)) × Temperature Difference (K) × Duration (seconds)
            
            Where Duration (seconds) = Duration (hours) × 3600
            
            **Step 2: Calculate Effective PCM Energy Storage Capacity**
            
            Effective Energy (kJ/kg) = Latent Heat (kJ/kg) + [Specific Heat (kJ/(kg·K)) × Usable Temp Swing (K)]
            
            **Step 3: Calculate Required PCM Mass**
            
            PCM Mass per Area (kg/m²) = [Energy per Area (kJ/m²) / Effective Energy (kJ/kg)] × Safety Factor
            
            **Step 4: Economic Calculations**
            
            - Raw Material Cost = Mass (kg) × Cost per kg ($/kg)
            - Adjusted Cost = Raw Cost × (1 + Waste Factor)
            - Total Cost = Material + Installation + Labor + Transport
            
            **Step 5: Energy Savings**
            
            - Daily Thermal Energy (kWh) = Energy per Area (J/m²) × Area (m²) / 3.6×10⁶
            - Annual Thermal Energy (kWh) = Daily Energy × Cooling Days
            - Electrical Savings (kWh) = Thermal Energy / COP × Load Reduction
            - Annual Savings ($) = Electrical Savings × Electricity Cost
            
            **Step 6: Payback**
            
            - Payback Period (years) = Total Cost / Annual Savings
            - ROI (%) = [(10-Year Savings) - Total Cost] / Total Cost × 100
            
            ### Key Assumptions:
            
            1. **Steady-state temperature difference:** Constant deltaT over buffering period
            2. **Lumped energy buffering:** Heat storage as lumped parameter
            3. **Linear cost model:** Costs scale linearly with quantities
            4. **Simplified COP:** Constant cooling efficiency
            5. **No degradation:** No performance loss over time
            6. **Simplified alternative materials:** Basic comparison model
            """)
        
        # ========================================================================
        # SENSITIVITY PLOTS
        # ========================================================================
        
        st.header("📊 Sensitivity Analysis")
        
        st.markdown("""
        The following charts show how the PCM requirement changes with different parameters,
        helping you understand the sensitivity of the design to key variables.
        """)
        
        # Plot 1: Duration sensitivity
        st.subheader("PCM Requirement vs. Buffering Duration")
        fig1 = create_sensitivity_plot_duration(
            u_value_w_m2_k, delta_t_k, latent_heat_kj_kg,
            cp_pcm_kj_kg_k, usable_temp_swing_k, safety_factor, wall_area_ft2
        )
        st.pyplot(fig1)
        plt.close(fig1)
        
        st.caption(f"""
        This plot shows how PCM requirements scale with the duration of buffering, 
        assuming U-value = {u_value_w_m2_k:.2f} W/(m²·K) and ΔT = {delta_t_k:.2f} K.
        """)
        
        # Plot 2: Temperature difference sensitivity
        st.subheader("PCM Requirement vs. Temperature Difference")
        fig2 = create_sensitivity_plot_delta_t(
            u_value_w_m2_k, duration_hours, latent_heat_kj_kg,
            cp_pcm_kj_kg_k, usable_temp_swing_k, safety_factor, wall_area_ft2
        )
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.caption(f"""
        This plot shows how PCM requirements scale with temperature difference, 
        assuming U-value = {u_value_w_m2_k:.2f} W/(m²·K) and Duration = {duration_hours:.1f} hours.
        """)
        
        # ========================================================================
        # DOWNLOAD SUMMARY
        # ========================================================================
        
        st.header("💾 Export Results")
        
        # Prepare inputs and outputs for CSV
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
            'PCM Volume (cubic inches)': f"{volume_cubic_inches:.2f}",
            'Equivalent Thickness (mm)': f"{thickness_mm:.3f}",
            'Equivalent Thickness (inches)': f"{thickness_inches:.4f}",
            'Energy per Area (kJ/m²)': f"{energy_per_area_j_m2/1000:.2f}",
            'Effective Energy per kg (kJ/kg)': f"{effective_energy_kj_kg:.2f}"
        }
        
        # Add composite data
        if composite_results:
            inputs_dict['PCM Mass Fraction (%)'] = pcm_mass_fraction
            inputs_dict['Composite Density (kg/m³)'] = composite_density_kg_m3
            outputs_dict['Composite Mass (lb)'] = f"{composite_results['composite_mass_lb']:.2f}"
            outputs_dict['Composite Mass (kg)'] = f"{composite_results['composite_mass_kg']:.2f}"
            if 'composite_volume_liters' in composite_results:
                outputs_dict['Composite Volume (liters)'] = f"{composite_results['composite_volume_liters']:.2f}"
        
        # Add economic data
        if cost_results:
            inputs_dict['PCM Cost ($/kg)'] = pcm_cost_per_kg
            inputs_dict['Installation Cost ($/m²)'] = installation_cost_per_m2
            inputs_dict['Labor Markup (%)'] = labor_markup_percent
            inputs_dict['Transport Cost ($)'] = transport_cost
            inputs_dict['Waste Factor (%)'] = waste_factor_percent
            
            outputs_dict['Raw Material Cost ($)'] = f"{cost_results['raw_material_cost']:.2f}"
            outputs_dict['Adjusted Material Cost ($)'] = f"{cost_results['adjusted_material_cost']:.2f}"
            outputs_dict['Installation Cost ($)'] = f"{cost_results['installation_cost']:.2f}"
            outputs_dict['Labor Cost ($)'] = f"{cost_results['labor_cost']:.2f}"
            outputs_dict['Total Project Cost ($)'] = f"{cost_results['total_cost']:.2f}"
            outputs_dict['Cost per m² ($/m²)'] = f"{cost_results['cost_per_m2']:.2f}"
            outputs_dict['Cost per ft² ($/ft²)'] = f"{cost_results['cost_per_ft2']:.2f}"
        
        # Add savings data
        if savings_results and payback_results:
            inputs_dict['Electricity Cost ($/kWh)'] = electricity_cost_per_kwh
            inputs_dict['Cooling COP'] = cooling_cop
            inputs_dict['Load Reduction (%)'] = load_reduction_percent
            inputs_dict['Annual Cooling Days'] = annual_cooling_days
            
            outputs_dict['Annual Electrical Savings (kWh)'] = f"{savings_results['annual_electrical_savings_kwh']:.1f}"
            outputs_dict['Annual Cost Savings ($/yr)'] = f"{savings_results['annual_cost_savings']:.2f}"
            outputs_dict['10-Year Savings ($)'] = f"{savings_results['ten_year_savings']:.2f}"
            outputs_dict['Payback Period (years)'] = f"{payback_results['payback_years']:.1f}"
            outputs_dict['10-Year ROI (%)'] = f"{payback_results['ten_year_roi']:.1f}"
        
        csv_data = generate_csv_summary(inputs_dict, outputs_dict)
        
        st.download_button(
            label="📥 Download Complete Summary as CSV",
            data=csv_data,
            file_name="pcm_calculator_professional_summary.csv",
            mime="text/csv",
            help="Download a CSV file containing all inputs and outputs including economic analysis"
        )
        
    except Exception as e:
        st.error(f"""
        ❌ **Calculation Error**
        
        An error occurred during calculation: {str(e)}
        
        Please check your input values and try again.
        """)
        st.exception(e)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Basalt-PCM Thermal Buffer Sizing Calculator - Professional Edition</strong></p>
    <p>Comprehensive analysis including thermal, economic, and feasibility assessment | Version 2.0</p>
    <p><em>Note: This calculator provides estimates based on simplified assumptions. 
    For critical applications, consult with thermal engineering and financial professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
