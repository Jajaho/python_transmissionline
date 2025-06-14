import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time

class TransmissionLinePhysics:
    """Core physics calculations for transmission line phenomena"""
    
    def __init__(self):
        self.c = 3e8  # Speed of light in m/s
    
    def calculate_reflection_coefficient(self, z_load, z_char):
        """Calculate reflection coefficient Γ = (ZL - Z0) / (ZL + Z0)"""
        if z_load + z_char == 0:
            return 1.0
        return (z_load - z_char) / (z_load + z_char)
    
    def calculate_transmission_coefficient(self, z_load, z_char):
        """Calculate transmission coefficient T = 2*ZL / (ZL + Z0)"""
        if z_load + z_char == 0:
            return 0.0
        return (2 * z_load) / (z_load + z_char)
    
    def calculate_velocity(self, dielectric_constant):
        """Calculate propagation velocity v = c / sqrt(εr)"""
        return self.c / np.sqrt(dielectric_constant)
    
    def calculate_attenuation(self, distance, alpha):
        """Calculate attenuation factor: e^(-α*d)"""
        return np.exp(-alpha * distance)
    
    def generate_incident_pulse(self, t, pulse_width=1e-9, amplitude=1.0):
        """Generate a Gaussian pulse"""
        return amplitude * np.exp(-(t / pulse_width) ** 2)
    
    def propagate_signal(self, signal, distance, velocity, attenuation_factor):
        """Propagate signal with delay and attenuation"""
        delay = distance / velocity
        attenuated_amplitude = signal * attenuation_factor
        return attenuated_amplitude, delay

class SignalProcessor:
    """Handles signal generation and processing"""
    
    def __init__(self, physics_engine):
        self.physics = physics_engine
        self.time_step = 1e-11  # 10 ps
        self.max_time = 2e-8   # 20 ns
        self.line_length = 1.0  # 1 meter
        
    def create_time_array(self):
        """Create time array for simulation"""
        return np.arange(0, self.max_time, self.time_step)
    
    def simulate_transmission_line(self, z_char, z_load, dielectric_const, 
                                 attenuation_coeff, pulse_amplitude, pulse_width):
        """Complete transmission line simulation"""
        time_array = self.create_time_array()
        
        # Calculate parameters
        gamma = self.physics.calculate_reflection_coefficient(z_load, z_char)
        tau = self.physics.calculate_transmission_coefficient(z_load, z_char)
        velocity = self.physics.calculate_velocity(dielectric_const)
        atten_factor = self.physics.calculate_attenuation(self.line_length, attenuation_coeff)
        
        # Propagation delay
        one_way_delay = self.line_length / velocity
        
        # Initialize signals
        incident_signal = np.zeros_like(time_array)
        reflected_signal = np.zeros_like(time_array)
        transmitted_signal = np.zeros_like(time_array)
        
        # Generate incident pulse
        for i, t in enumerate(time_array):
            incident_signal[i] = self.physics.generate_incident_pulse(
                t, pulse_width, pulse_amplitude)
        
        # Calculate reflected and transmitted signals
        for i, t in enumerate(time_array):
            # Incident signal at load (delayed and attenuated)
            if t >= one_way_delay:
                delayed_index = int((t - one_way_delay) / self.time_step)
                if delayed_index < len(incident_signal):
                    incident_at_load = incident_signal[delayed_index] * atten_factor
                    
                    # Reflected signal (returns to source)
                    if t >= 2 * one_way_delay:
                        reflected_index = int((t - 2 * one_way_delay) / self.time_step)
                        if reflected_index < len(incident_signal):
                            reflected_signal[i] = (gamma * incident_signal[reflected_index] * 
                                                 atten_factor ** 2)
                    
                    # Transmitted signal
                    transmitted_signal[i] = tau * incident_at_load
        
        return {
            'time': time_array * 1e9,  # Convert to ns
            'incident': incident_signal,
            'reflected': reflected_signal,
            'transmitted': transmitted_signal,
            'total_source': incident_signal + reflected_signal,
            'reflection_coeff': gamma,
            'transmission_coeff': tau,
            'velocity': velocity,
            'one_way_delay': one_way_delay * 1e9  # Convert to ns
        }

class PlotManager:
    """Manages matplotlib plots and visualization"""
    
    def __init__(self, parent_frame):
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        
        self.fig.tight_layout(pad=3.0)
        
    def plot_signals(self, results):
        """Plot all signals and analysis"""
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        time_ns = results['time']
        
        # Plot 1: Source signals
        self.ax1.plot(time_ns, results['incident'], 'b-', label='Incident', linewidth=2)
        self.ax1.plot(time_ns, results['reflected'], 'r-', label='Reflected', linewidth=2)
        self.ax1.plot(time_ns, results['total_source'], 'g-', label='Total at Source', linewidth=2)
        self.ax1.set_xlabel('Time (ns)')
        self.ax1.set_ylabel('Amplitude (V)')
        self.ax1.set_title('Signals at Source End')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Load signal
        self.ax2.plot(time_ns, results['transmitted'], 'm-', label='Transmitted', linewidth=2)
        self.ax2.set_xlabel('Time (ns)')
        self.ax2.set_ylabel('Amplitude (V)')
        self.ax2.set_title('Signal at Load End')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reflection analysis
        gamma = results['reflection_coeff']
        tau = results['transmission_coeff']
        
        self.ax3.bar(['Reflection\nCoefficient', 'Transmission\nCoefficient'], 
                    [abs(gamma), abs(tau)], 
                    color=['red', 'blue'], alpha=0.7)
        self.ax3.set_ylabel('Magnitude')
        self.ax3.set_title('Reflection Analysis')
        self.ax3.grid(True, alpha=0.3)
        
        # Add text annotations
        self.ax3.text(0, abs(gamma) + 0.05, f'{gamma:.3f}', 
                     ha='center', va='bottom', fontweight='bold')
        self.ax3.text(1, abs(tau) + 0.05, f'{tau:.3f}', 
                     ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Signal propagation info
        self.ax4.axis('off')
        info_text = f"""Signal Propagation Parameters:
        
Velocity: {results['velocity']/1e8:.2f} × 10⁸ m/s
One-way Delay: {results['one_way_delay']:.2f} ns
        
Reflection Coefficient (Γ): {gamma:.3f}
Transmission Coefficient (τ): {tau:.3f}
        
Power Reflection: {abs(gamma)**2:.1%}
Power Transmission: {abs(tau)**2:.1%}"""
        
        self.ax4.text(0.05, 0.95, info_text, transform=self.ax4.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        self.canvas.draw()

class ControlPanel:
    """GUI control panel for parameter adjustment"""
    
    def __init__(self, parent_frame, update_callback):
        self.frame = ttk.Frame(parent_frame)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        self.update_callback = update_callback
        
        self.create_controls()
        
    def create_controls(self):
        """Create all control widgets"""
        # Impedance controls
        impedance_frame = ttk.LabelFrame(self.frame, text="Impedance Settings")
        impedance_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(impedance_frame, text="Characteristic Impedance (Ω):").grid(row=0, column=0, sticky=tk.W)
        self.z_char_var = tk.StringVar(value="50")
        ttk.Entry(impedance_frame, textvariable=self.z_char_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(impedance_frame, text="Load Impedance (Ω):").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.z_load_var = tk.StringVar(value="75")
        ttk.Entry(impedance_frame, textvariable=self.z_load_var, width=10).grid(row=0, column=3, padx=5)
        
        # Line parameters
        line_frame = ttk.LabelFrame(self.frame, text="Line Parameters")
        line_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(line_frame, text="Dielectric Constant:").grid(row=0, column=0, sticky=tk.W)
        self.dielectric_var = tk.StringVar(value="2.2")
        ttk.Entry(line_frame, textvariable=self.dielectric_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(line_frame, text="Attenuation (Np/m):").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.attenuation_var = tk.StringVar(value="0.1")
        ttk.Entry(line_frame, textvariable=self.attenuation_var, width=10).grid(row=0, column=3, padx=5)
        
        # Pulse parameters
        pulse_frame = ttk.LabelFrame(self.frame, text="Pulse Parameters")
        pulse_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(pulse_frame, text="Amplitude (V):").grid(row=0, column=0, sticky=tk.W)
        self.amplitude_var = tk.StringVar(value="1.0")
        ttk.Entry(pulse_frame, textvariable=self.amplitude_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(pulse_frame, text="Pulse Width (ns):").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.pulse_width_var = tk.StringVar(value="1.0")
        ttk.Entry(pulse_frame, textvariable=self.pulse_width_var, width=10).grid(row=0, column=3, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Update Simulation", 
                  command=self.update_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preset: Matched", 
                  command=self.preset_matched).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preset: Open Circuit", 
                  command=self.preset_open).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preset: Short Circuit", 
                  command=self.preset_short).pack(side=tk.LEFT, padx=5)
        
    def get_parameters(self):
        """Get current parameter values"""
        try:
            return {
                'z_char': float(self.z_char_var.get()),
                'z_load': float(self.z_load_var.get()),
                'dielectric_const': float(self.dielectric_var.get()),
                'attenuation_coeff': float(self.attenuation_var.get()),
                'pulse_amplitude': float(self.amplitude_var.get()),
                'pulse_width': float(self.pulse_width_var.get()) * 1e-9  # Convert to seconds
            }
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")
            return None
    
    def update_simulation(self):
        """Trigger simulation update"""
        params = self.get_parameters()
        if params:
            self.update_callback(params)
    
    def reset_defaults(self):
        """Reset all parameters to defaults"""
        self.z_char_var.set("50")
        self.z_load_var.set("75")
        self.dielectric_var.set("2.2")
        self.attenuation_var.set("0.1")
        self.amplitude_var.set("1.0")
        self.pulse_width_var.set("1.0")
        self.update_simulation()
    
    def preset_matched(self):
        """Preset for matched impedances"""
        self.z_char_var.set("50")
        self.z_load_var.set("50")
        self.update_simulation()
    
    def preset_open(self):
        """Preset for open circuit"""
        self.z_char_var.set("50")
        self.z_load_var.set("1000000")  # Very high impedance
        self.update_simulation()
    
    def preset_short(self):
        """Preset for short circuit"""
        self.z_char_var.set("50")
        self.z_load_var.set("0.001")  # Very low impedance
        self.update_simulation()

class TransmissionLineSimulator:
    """Main application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Transmission Line Signal Simulator")
        self.root.geometry("1200x900")
        
        # Initialize components
        self.physics = TransmissionLinePhysics()
        self.processor = SignalProcessor(self.physics)
        
        self.setup_gui()
        self.initial_simulation()
        
    def setup_gui(self):
        """Setup the main GUI"""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize components
        self.control_panel = ControlPanel(control_frame, self.update_simulation)
        self.plot_manager = PlotManager(plot_frame)
        
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
    
    def update_simulation(self, params):
        """Update simulation with new parameters"""
        try:
            results = self.processor.simulate_transmission_line(**params)
            self.plot_manager.plot_signals(results)
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Error running simulation: {e}")
    
    def initial_simulation(self):
        """Run initial simulation with default parameters"""
        params = self.control_panel.get_parameters()
        if params:
            self.update_simulation(params)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Transmission Line Signal Simulator
        
This application simulates electrical signal propagation along transmission lines,
demonstrating key phenomena including:

• Signal attenuation
• Propagation delay
• Impedance mismatches
• Reflection and transmission coefficients
• Standing wave patterns

Developed for educational and engineering analysis purposes."""
        
        messagebox.showinfo("About", about_text)
    
    def show_instructions(self):
        """Show instructions dialog"""
        instructions = """How to Use:

1. Adjust impedance values to see reflection effects
2. Modify line parameters to observe propagation changes
3. Change pulse parameters to study different signals
4. Use preset buttons for common scenarios:
   - Matched: No reflections
   - Open Circuit: Full positive reflection
   - Short Circuit: Full negative reflection

Key Concepts:
• Γ = (ZL - Z0) / (ZL + Z0) - Reflection coefficient
• When |Γ| = 0: Perfect match, no reflections
• When |Γ| = 1: Total reflection
• Velocity = c / √εr where c is speed of light"""
        
        messagebox.showinfo("Instructions", instructions)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TransmissionLineSimulator()
    app.run()