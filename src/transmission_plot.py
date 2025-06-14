#!/usr/bin/env python3
"""
Educational Transmission Line Application
=========================================

This application demonstrates key transmission line concepts including:
- Characteristic impedance calculation
- Reflection coefficient and return loss
- VSWR (Voltage Standing Wave Ratio)
- Input impedance transformation
- Power calculations
- Basic Smith chart operations

Author: Educational Tool
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import cmath

@dataclass
class TransmissionLine:
    """Represents a transmission line with its parameters."""
    Z0: float  # Characteristic impedance (Ohms)
    length: float  # Physical length (meters)
    frequency: float  # Operating frequency (Hz)
    velocity_factor: float = 0.66  # Velocity factor (typical for coax)
    loss_db_per_100m: float = 0.0  # Loss in dB per 100m
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength in the transmission line."""
        c = 299792458  # Speed of light in m/s
        return (c * self.velocity_factor) / self.frequency
    
    @property
    def electrical_length_degrees(self) -> float:
        """Calculate electrical length in degrees."""
        return (self.length / self.wavelength) * 360
    
    @property
    def electrical_length_radians(self) -> float:
        """Calculate electrical length in radians."""
        return (self.length / self.wavelength) * 2 * np.pi
    
    @property
    def attenuation_constant(self) -> float:
        """Calculate attenuation constant in Np/m."""
        # Convert dB/100m to Np/m
        return (self.loss_db_per_100m / 100) * (np.log(10) / 20)

class TransmissionLineCalculator:
    """Main calculator class for transmission line operations."""
    
    def __init__(self):
        self.history = []
    
    def reflection_coefficient(self, ZL: complex, Z0: float) -> complex:
        """
        Calculate reflection coefficient.
        
        Args:
            ZL: Load impedance (complex)
            Z0: Characteristic impedance (real)
            
        Returns:
            Complex reflection coefficient
        """
        return (ZL - Z0) / (ZL + Z0)
    
    def vswr(self, gamma: complex) -> float:
        """
        Calculate VSWR from reflection coefficient.
        
        Args:
            gamma: Reflection coefficient (complex)
            
        Returns:
            VSWR (real, >= 1)
        """
        gamma_mag = abs(gamma)
        if gamma_mag >= 1:
            return float('inf')
        return (1 + gamma_mag) / (1 - gamma_mag)
    
    def return_loss_db(self, gamma: complex) -> float:
        """
        Calculate return loss in dB.
        
        Args:
            gamma: Reflection coefficient (complex)
            
        Returns:
            Return loss in dB (positive value)
        """
        return -20 * np.log10(abs(gamma)) if abs(gamma) > 0 else float('inf')
    
    def input_impedance(self, ZL: complex, tline: TransmissionLine) -> complex:
        """
        Calculate input impedance of a transmission line.
        
        Args:
            ZL: Load impedance (complex)
            tline: Transmission line parameters
            
        Returns:
            Input impedance (complex)
        """
        beta = tline.electrical_length_radians
        alpha = tline.attenuation_constant * tline.length
        gamma_complex = alpha + 1j * beta
        
        if abs(ZL) == float('inf'):  # Open circuit
            return -1j * tline.Z0 / np.tan(beta)
        elif ZL == 0:  # Short circuit
            return 1j * tline.Z0 * np.tan(beta)
        else:
            # General case with loss
            tanh_gamma = np.tanh(gamma_complex)
            return tline.Z0 * (ZL + tline.Z0 * tanh_gamma) / (tline.Z0 + ZL * tanh_gamma)
    
    def power_calculations(self, ZL: complex, Z0: float, V_forward: float) -> dict:
        """
        Calculate power-related parameters.
        
        Args:
            ZL: Load impedance (complex)
            Z0: Characteristic impedance
            V_forward: Forward voltage amplitude
            
        Returns:
            Dictionary with power calculations
        """
        gamma = self.reflection_coefficient(ZL, Z0)
        
        P_incident = (V_forward ** 2) / (2 * Z0)
        P_reflected = P_incident * (abs(gamma) ** 2)
        P_delivered = P_incident - P_reflected
        
        efficiency = (P_delivered / P_incident) * 100 if P_incident > 0 else 0
        
        return {
            'P_incident_W': P_incident,
            'P_reflected_W': P_reflected,
            'P_delivered_W': P_delivered,
            'efficiency_percent': efficiency,
            'reflection_coefficient': gamma
        }
    
    def quarter_wave_transformer(self, Z1: float, Z2: float) -> float:
        """
        Calculate characteristic impedance for quarter-wave matching transformer.
        
        Args:
            Z1: Source impedance
            Z2: Load impedance
            
        Returns:
            Required characteristic impedance for matching
        """
        return np.sqrt(Z1 * Z2)
    
    def frequency_response(self, ZL: complex, tline: TransmissionLine, 
                          freq_range: Tuple[float, float], num_points: int = 100) -> dict:
        """
        Calculate frequency response of input impedance.
        
        Args:
            ZL: Load impedance
            tline: Transmission line (frequency will be varied)
            freq_range: (start_freq, end_freq) in Hz
            num_points: Number of frequency points
            
        Returns:
            Dictionary with frequency arrays and impedance data
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], num_points)
        Z_in = []
        vswr_list = []
        
        for freq in frequencies:
            tline_temp = TransmissionLine(
                Z0=tline.Z0,
                length=tline.length,
                frequency=freq,
                velocity_factor=tline.velocity_factor,
                loss_db_per_100m=tline.loss_db_per_100m
            )
            
            z_in = self.input_impedance(ZL, tline_temp)
            gamma = self.reflection_coefficient(z_in, tline.Z0)
            vswr_val = self.vswr(gamma)
            
            Z_in.append(z_in)
            vswr_list.append(vswr_val)
        
        return {
            'frequencies': frequencies,
            'Z_in': np.array(Z_in),
            'vswr': np.array(vswr_list)
        }

def create_smith_chart_basic():
    """Create a basic Smith chart for educational purposes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle')
    
    # Draw constant resistance circles
    r_values = [0.2, 0.5, 1.0, 2.0, 5.0]
    for r in r_values:
        center = r / (1 + r)
        radius = 1 / (1 + r)
        circle_theta = np.linspace(0, 2*np.pi, 1000)
        x = center + radius * np.cos(circle_theta)
        y = radius * np.sin(circle_theta)
        ax.plot(x, y, 'b--', alpha=0.7, linewidth=1)
        ax.text(center + radius - 0.05, 0, f'R={r}', fontsize=8, ha='right')
    
    # Draw constant reactance circles
    x_values = [0.2, 0.5, 1.0, 2.0, 5.0]
    for x in x_values:
        center_x = 1
        center_y = 1/x
        radius = 1/x
        
        # Create a full circle and then filter points inside unit circle
        circle_theta = np.linspace(0, 2*np.pi, 1000)
        x_coords = center_x + radius * np.cos(circle_theta)
        y_coords = center_y + radius * np.sin(circle_theta)
        
        # For positive reactance (upper half)
        # Only plot points inside unit circle and in upper half
        mask_pos = (x_coords**2 + y_coords**2 <= 1.01) & (y_coords >= -0.01)
        if np.any(mask_pos):
            ax.plot(x_coords[mask_pos], y_coords[mask_pos], 'r--', alpha=0.7, linewidth=1)
            # Label for positive reactance
            label_y = min(1/x, 0.9)
            ax.text(1.05, label_y, f'X={x}', fontsize=8, ha='left')
        
        # For negative reactance (lower half)
        # Reflect the circle center to negative y
        y_coords_neg = -center_y + radius * np.sin(circle_theta)
        mask_neg = (x_coords**2 + y_coords_neg**2 <= 1.01) & (y_coords_neg <= 0.01)
        if np.any(mask_neg):
            ax.plot(x_coords[mask_neg], y_coords_neg[mask_neg], 'r--', alpha=0.7, linewidth=1)
            # Label for negative reactance
            label_y_neg = max(-1/x, -0.9)
            ax.text(1.05, label_y_neg, f'X={-x}', fontsize=8, ha='left')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Basic Smith Chart', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real Part of Γ')
    ax.set_ylabel('Imaginary Part of Γ')
    
    return fig, ax

def educational_examples():
    """Run several educational examples."""
    calc = TransmissionLineCalculator()
    
    print("=" * 60)
    print("TRANSMISSION LINE EDUCATIONAL EXAMPLES")
    print("=" * 60)
    
    # Example 1: Basic calculations
    print("\n1. BASIC TRANSMISSION LINE CALCULATIONS")
    print("-" * 40)
    
    # Create a 50-ohm coaxial cable
    coax = TransmissionLine(
        Z0=50.0,  # 50-ohm coax
        length=0.25,  # 25 cm
        frequency=1e9,  # 1 GHz
        velocity_factor=0.66,
        loss_db_per_100m=2.0
    )
    
    print(f"Transmission Line Parameters:")
    print(f"  Characteristic Impedance: {coax.Z0} Ω")
    print(f"  Physical Length: {coax.length} m")
    print(f"  Frequency: {coax.frequency/1e9:.1f} GHz")
    print(f"  Wavelength in line: {coax.wavelength:.3f} m")
    print(f"  Electrical Length: {coax.electrical_length_degrees:.1f}°")
    
    # Different load impedances
    loads = [
        ("Matched Load", 50 + 0j),
        ("Short Circuit", 0 + 0j),
        ("Open Circuit", complex('inf')),
        ("Resistive Mismatch", 75 + 0j),
        ("Complex Load", 50 + 25j)
    ]
    
    print(f"\nLoad Analysis:")
    for name, ZL in loads:
        if abs(ZL) == float('inf'):
            ZL_display = "∞"
            gamma = calc.reflection_coefficient(1e10, coax.Z0)  # Approximate open
        else:
            ZL_display = f"{ZL.real:.1f} + {ZL.imag:.1f}j Ω"
            gamma = calc.reflection_coefficient(ZL, coax.Z0)
        
        vswr_val = calc.vswr(gamma)
        rl_db = calc.return_loss_db(gamma)
        
        print(f"\n  {name}: {ZL_display}")
        print(f"    Γ = {gamma.real:.3f} + {gamma.imag:.3f}j")
        print(f"    |Γ| = {abs(gamma):.3f}")
        print(f"    VSWR = {vswr_val:.2f}")
        print(f"    Return Loss = {rl_db:.1f} dB")
    
    # Example 2: Input impedance transformation
    print("\n\n2. INPUT IMPEDANCE TRANSFORMATION")
    print("-" * 40)
    
    ZL = 75 + 0j  # 75-ohm resistive load
    Z_in = calc.input_impedance(ZL, coax)
    
    print(f"Load Impedance: {ZL.real:.1f} + {ZL.imag:.1f}j Ω")
    print(f"Input Impedance: {Z_in.real:.1f} + {Z_in.imag:.1f}j Ω")
    
    # Show how input impedance varies with electrical length
    lengths_deg = np.linspace(0, 180, 19)
    print(f"\nInput Impedance vs Electrical Length:")
    print(f"{'Length (°)':<12} {'Z_in (Ω)':<20} {'|Z_in| (Ω)':<12}")
    
    for length_deg in lengths_deg:
        temp_line = TransmissionLine(
            Z0=coax.Z0,
            length=coax.length * (length_deg / coax.electrical_length_degrees),
            frequency=coax.frequency,
            velocity_factor=coax.velocity_factor
        )
        Z_temp = calc.input_impedance(ZL, temp_line)
        print(f"{length_deg:8.0f}     {Z_temp.real:6.1f} + {Z_temp.imag:6.1f}j     {abs(Z_temp):8.1f}")
    
    # Example 3: Quarter-wave transformer
    print("\n\n3. QUARTER-WAVE MATCHING TRANSFORMER")
    print("-" * 40)
    
    Z_source = 50.0  # Source impedance
    Z_load = 200.0   # Load to be matched
    Z_transformer = calc.quarter_wave_transformer(Z_source, Z_load)
    
    print(f"Source Impedance: {Z_source:.0f} Ω")
    print(f"Load Impedance: {Z_load:.0f} Ω")
    print(f"Required Transformer Z0: {Z_transformer:.1f} Ω")
    
    # Verify the matching
    quarter_wave = TransmissionLine(
        Z0=Z_transformer,
        length=0.25 * (299792458 * 0.66) / 1e9,  # Quarter wavelength at 1 GHz
        frequency=1e9,
        velocity_factor=0.66
    )
    
    Z_in_matched = calc.input_impedance(Z_load, quarter_wave)
    print(f"Input impedance with transformer: {Z_in_matched.real:.1f} + {Z_in_matched.imag:.1f}j Ω")
    
    # Example 4: Power calculations
    print("\n\n4. POWER ANALYSIS")
    print("-" * 40)
    
    V_forward = 10.0  # 10V forward voltage
    ZL = 25 + 0j      # 25-ohm load
    
    power_results = calc.power_calculations(ZL, coax.Z0, V_forward)
    
    print(f"Forward Voltage: {V_forward:.1f} V")
    print(f"Load Impedance: {ZL.real:.0f} Ω")
    print(f"Characteristic Impedance: {coax.Z0:.0f} Ω")
    print(f"\nPower Analysis:")
    print(f"  Incident Power: {power_results['P_incident_W']:.3f} W")
    print(f"  Reflected Power: {power_results['P_reflected_W']:.3f} W")
    print(f"  Delivered Power: {power_results['P_delivered_W']:.3f} W")
    print(f"  Efficiency: {power_results['efficiency_percent']:.1f}%")

def plot_frequency_response():
    """Create frequency response plots."""
    calc = TransmissionLineCalculator()
    
    # Create transmission line
    tline = TransmissionLine(
        Z0=50.0,
        length=0.1,  # 10 cm
        frequency=1e9,  # Will be varied
        velocity_factor=0.66
    )
    
    # Different loads to compare
    loads = [
        ("50Ω (Matched)", 50 + 0j),
        ("25Ω (Resistive)", 25 + 0j),
        ("100Ω (Resistive)", 100 + 0j),
        ("50+25j Ω (Complex)", 50 + 25j)
    ]
    
    freq_range = (100e6, 2e9)  # 100 MHz to 2 GHz
    
    plt.figure(figsize=(15, 10))
    
    # Plot input impedance magnitude
    plt.subplot(2, 2, 1)
    for name, ZL in loads:
        response = calc.frequency_response(ZL, tline, freq_range)
        plt.plot(response['frequencies']/1e9, np.abs(response['Z_in']), 
                label=name, linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('|Z_in| (Ω)')
    plt.title('Input Impedance Magnitude vs Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 200)
    
    # Plot input impedance phase
    plt.subplot(2, 2, 2)
    for name, ZL in loads:
        response = calc.frequency_response(ZL, tline, freq_range)
        plt.plot(response['frequencies']/1e9, np.angle(response['Z_in'], deg=True), 
                label=name, linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Phase of Z_in (degrees)')
    plt.title('Input Impedance Phase vs Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot VSWR
    plt.subplot(2, 2, 3)
    for name, ZL in loads:
        response = calc.frequency_response(ZL, tline, freq_range)
        plt.plot(response['frequencies']/1e9, response['vswr'], 
                label=name, linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('VSWR')
    plt.title('VSWR vs Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(1, 10)
    
    # Plot return loss
    plt.subplot(2, 2, 4)
    for name, ZL in loads:
        response = calc.frequency_response(ZL, tline, freq_range)
        gamma_vals = [calc.reflection_coefficient(z, tline.Z0) for z in response['Z_in']]
        rl_vals = [calc.return_loss_db(g) for g in gamma_vals]
        plt.plot(response['frequencies']/1e9, rl_vals, 
                label=name, linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Return Loss (dB)')
    plt.title('Return Loss vs Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 40)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run educational examples
    educational_examples()
    
    # Create plots
    print(f"\n\n5. GENERATING PLOTS...")
    print("-" * 40)
    print("Creating frequency response plots...")
    plot_frequency_response()
    
    print("Creating Smith chart...")
    fig, ax = create_smith_chart_basic()
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("EDUCATIONAL EXERCISES FOR STUDENTS")
    print("=" * 60)
    print("\nTry these exercises to deepen your understanding:")
    print("1. Modify the load impedance and observe VSWR changes")
    print("2. Change the transmission line length and see input impedance variation")
    print("3. Calculate the required length for impedance matching")
    print("4. Explore the effect of transmission line losses")
    print("5. Design a quarter-wave transformer for different impedance ratios")
    print("\nThis application demonstrates fundamental transmission line concepts.")
    print("Experiment with different parameters to build intuition!")