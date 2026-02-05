#!/usr/bin/env python3
"""
GUI Control Panel
=================

Interactive GUI for real-time simulation control.

Features:
- Parameter sliders (immune cells, O2, methylation, etc.)
- Play/Pause/Reset buttons
- Real-time statistics display
- Module enable/disable
- Scenario selection
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time


class ControlPanel:
    """
    Interactive GUI control panel for simulation
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.running = False
        self.sim_thread = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("cognisom Control Panel")
        self.root.geometry("800x600")
        
        # Setup UI
        self._setup_ui()
        
        # Update timer
        self.update_stats()
    
    def _setup_ui(self):
        """Setup user interface"""
        # Title
        title = tk.Label(self.root, text="cognisom Control Panel", 
                        font=("Arial", 18, "bold"))
        title.pack(pady=10)
        
        # Create notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Control tab
        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="Control")
        self._setup_control_tab(control_frame)
        
        # Parameters tab
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self._setup_parameters_tab(params_frame)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        self._setup_statistics_tab(stats_frame)
        
        # Scenarios tab
        scenarios_frame = ttk.Frame(notebook)
        notebook.add(scenarios_frame, text="Scenarios")
        self._setup_scenarios_tab(scenarios_frame)
    
    def _setup_control_tab(self, parent):
        """Setup control buttons"""
        # Buttons frame
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=20)
        
        # Play button
        self.play_btn = tk.Button(btn_frame, text="▶ Play", 
                                  command=self.play, width=10,
                                  bg="green", fg="white", font=("Arial", 12))
        self.play_btn.grid(row=0, column=0, padx=5)
        
        # Pause button
        self.pause_btn = tk.Button(btn_frame, text="⏸ Pause", 
                                   command=self.pause, width=10,
                                   bg="orange", fg="white", font=("Arial", 12))
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        # Reset button
        self.reset_btn = tk.Button(btn_frame, text="⟲ Reset", 
                                   command=self.reset, width=10,
                                   bg="red", fg="white", font=("Arial", 12))
        self.reset_btn.grid(row=0, column=2, padx=5)
        
        # Status
        self.status_label = tk.Label(parent, text="Status: Stopped", 
                                     font=("Arial", 14))
        self.status_label.pack(pady=10)
        
        # Progress
        self.progress = ttk.Progressbar(parent, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        # Time display
        self.time_label = tk.Label(parent, text="Time: 0.00 hours", 
                                   font=("Arial", 12))
        self.time_label.pack(pady=5)
    
    def _setup_parameters_tab(self, parent):
        """Setup parameter sliders"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parameters
        self.sliders = {}
        
        # Immune parameters
        tk.Label(scrollable_frame, text="Immune System", 
                font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=10)
        
        self._add_slider(scrollable_frame, "T Cells", "immune.n_t_cells", 
                        0, 100, 15, 1)
        self._add_slider(scrollable_frame, "NK Cells", "immune.n_nk_cells", 
                        0, 100, 10, 2)
        self._add_slider(scrollable_frame, "Macrophages", "immune.n_macrophages", 
                        0, 50, 8, 3)
        
        # Vascular parameters
        tk.Label(scrollable_frame, text="Vascular System", 
                font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=3, pady=10)
        
        self._add_slider(scrollable_frame, "Capillaries", "vascular.n_capillaries", 
                        1, 20, 8, 5)
        self._add_slider(scrollable_frame, "Arterial O2 (%)", "vascular.arterial_O2", 
                        5, 21, 21, 6, resolution=0.1)
        
        # Epigenetic parameters
        tk.Label(scrollable_frame, text="Epigenetics", 
                font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3, pady=10)
        
        self._add_slider(scrollable_frame, "Methylation Rate", "epigenetic.methylation_rate", 
                        0, 0.1, 0.01, 8, resolution=0.001)
        
        # Circadian parameters
        tk.Label(scrollable_frame, text="Circadian", 
                font=("Arial", 12, "bold")).grid(row=9, column=0, columnspan=3, pady=10)
        
        self._add_slider(scrollable_frame, "Coupling Strength", "circadian.coupling_strength", 
                        0, 1, 0.1, 10, resolution=0.01)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _add_slider(self, parent, label, param, min_val, max_val, default, row, resolution=1):
        """Add parameter slider"""
        tk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=10)
        
        slider = tk.Scale(parent, from_=min_val, to=max_val, 
                         orient='horizontal', length=300,
                         resolution=resolution)
        slider.set(default)
        slider.grid(row=row, column=1, padx=10)
        
        value_label = tk.Label(parent, text=f"{default}")
        value_label.grid(row=row, column=2, padx=10)
        
        # Update value label
        def update_label(val):
            value_label.config(text=f"{float(val):.3f}")
            self._apply_parameter(param, float(val))
        
        slider.config(command=update_label)
        
        self.sliders[param] = slider
    
    def _apply_parameter(self, param, value):
        """Apply parameter change to engine"""
        parts = param.split('.')
        if len(parts) == 2:
            module_name, param_name = parts
            if module_name in self.engine.modules:
                try:
                    self.engine.set_parameter(module_name, param_name, value)
                except:
                    pass
    
    def _setup_statistics_tab(self, parent):
        """Setup statistics display"""
        # Create text widget
        self.stats_text = tk.Text(parent, font=("Courier", 10), 
                                 width=80, height=30)
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, command=self.stats_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.stats_text.config(yscrollcommand=scrollbar.set)
    
    def _setup_scenarios_tab(self, parent):
        """Setup scenario selection"""
        tk.Label(parent, text="Pre-built Scenarios", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        scenarios = [
            ("Cancer Immunotherapy", "Boost immune system to fight cancer"),
            ("Circadian Disruption", "Simulate jet lag effects"),
            ("Hypoxia Response", "Low oxygen environment"),
            ("Epigenetic Therapy", "DNA methylation inhibitors"),
            ("Chronotherapy", "Time-based drug delivery"),
        ]
        
        for name, description in scenarios:
            frame = tk.Frame(parent, relief='raised', borderwidth=1)
            frame.pack(fill='x', padx=20, pady=5)
            
            tk.Label(frame, text=name, font=("Arial", 12, "bold")).pack(anchor='w', padx=10)
            tk.Label(frame, text=description, font=("Arial", 10)).pack(anchor='w', padx=10)
            
            btn = tk.Button(frame, text="Run", 
                          command=lambda n=name: self.run_scenario(n))
            btn.pack(anchor='e', padx=10, pady=5)
    
    def play(self):
        """Start simulation"""
        if not self.running:
            self.running = True
            self.status_label.config(text="Status: Running", fg="green")
            self.sim_thread = threading.Thread(target=self._run_simulation)
            self.sim_thread.start()
    
    def pause(self):
        """Pause simulation"""
        self.running = False
        self.status_label.config(text="Status: Paused", fg="orange")
    
    def reset(self):
        """Reset simulation"""
        self.pause()
        self.engine.reset()
        self.time_label.config(text="Time: 0.00 hours")
        self.progress['value'] = 0
        self.status_label.config(text="Status: Reset", fg="blue")
    
    def _run_simulation(self):
        """Run simulation in background thread"""
        while self.running:
            self.engine.step()
            time.sleep(0.01)  # 10ms delay
    
    def update_stats(self):
        """Update statistics display"""
        if hasattr(self, 'stats_text'):
            state = self.engine.get_state()
            
            # Format statistics
            stats = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    SIMULATION STATISTICS                       ║
╚═══════════════════════════════════════════════════════════════╝

Time: {state['time']:.2f} hours
Steps: {state['step_count']}

CELLULAR
  Total Cells: {state['cellular']['n_cells']}
  Cancer: {state['cellular']['n_cancer']}
  Normal: {state['cellular']['n_normal']}
  Divisions: {state['cellular']['total_divisions']}
  Deaths: {state['cellular']['total_deaths']}

IMMUNE
  Active: {state['immune']['n_immune_cells']}
  Activated: {state['immune']['n_activated']}
  Kills: {state['immune']['total_kills']}

VASCULAR
  Capillaries: {state['vascular']['n_capillaries']}
  Avg O2: {state['vascular']['avg_cell_O2']:.3f}
  Hypoxic Regions: {state['vascular']['hypoxic_regions']}
"""
            
            if 'lymphatic' in state:
                stats += f"""
LYMPHATIC
  Vessels: {state['lymphatic']['n_vessels']}
  Metastases: {state['lymphatic']['total_metastases']}
"""
            
            if 'epigenetic' in state:
                stats += f"""
EPIGENETIC
  Avg Methylation: {state['epigenetic']['avg_methylation']:.3f}
  Silenced Genes: {state['epigenetic']['silenced_genes']}
"""
            
            if 'circadian' in state:
                stats += f"""
CIRCADIAN
  Master Phase: {state['circadian']['master_phase']:.1f}h
  Time of Day: {state['circadian']['master_time_of_day']}
  Synchrony: {state['circadian']['synchrony']:.3f}
"""
            
            if 'morphogen' in state:
                stats += f"""
MORPHOGEN
  Gradients: {state['morphogen']['n_gradients']}
  Fates Determined: {state['morphogen']['total_fates_determined']}
"""
            
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert('1.0', stats)
            
            # Update time and progress
            self.time_label.config(text=f"Time: {state['time']:.2f} hours")
            self.progress['value'] = (state['time'] / 24.0) * 100
        
        # Schedule next update
        self.root.after(1000, self.update_stats)
    
    def run_scenario(self, scenario_name):
        """Run selected scenario"""
        messagebox.showinfo("Scenario", f"Running: {scenario_name}")
        # TODO: Implement scenario loading
    
    def run(self):
        """Run GUI"""
        self.root.mainloop()


# Test
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from core import SimulationEngine, SimulationConfig
    from modules import CellularModule, ImmuneModule, VascularModule
    
    print("Starting GUI Control Panel...")
    
    # Create engine
    engine = SimulationEngine(SimulationConfig())
    engine.register_module('cellular', CellularModule)
    engine.register_module('immune', ImmuneModule)
    engine.register_module('vascular', VascularModule)
    engine.initialize()
    
    # Create and run GUI
    panel = ControlPanel(engine)
    panel.run()
