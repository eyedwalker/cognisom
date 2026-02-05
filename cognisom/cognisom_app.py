#!/usr/bin/env python3
"""
cognisom Application
====================

Main entry point for cognisom simulation platform.

Usage:
    python3 cognisom_app.py
"""

import sys
from core import SimulationEngine, SimulationConfig
from ui import MenuSystem


class CognisomApp:
    """
    Main application with menu system
    """
    
    def __init__(self):
        self.engine = None
        self.menu = MenuSystem()
        self.config = SimulationConfig()
    
    def initialize_engine(self):
        """Initialize simulation engine with modules"""
        print("\nInitializing cognisom...")
        
        self.engine = SimulationEngine(self.config)
        self.menu.engine = self.engine
        
        # Register available modules
        from modules import (MolecularModule, CellularModule, ImmuneModule, 
                           VascularModule, LymphaticModule, SpatialModule)
        
        if self.config.modules_enabled.get('molecular', True):
            self.engine.register_module('molecular', MolecularModule, {
                'transcription_rate': 0.5,
                'exosome_release_rate': 0.1
            })
        
        if self.config.modules_enabled.get('cellular', True):
            self.engine.register_module('cellular', CellularModule, {
                'n_normal_cells': 80,
                'n_cancer_cells': 20
            })
        
        if self.config.modules_enabled.get('immune', True):
            self.engine.register_module('immune', ImmuneModule, {
                'n_t_cells': 15,
                'n_nk_cells': 10,
                'n_macrophages': 8
            })
        
        if self.config.modules_enabled.get('vascular', True):
            self.engine.register_module('vascular', VascularModule, {
                'n_capillaries': 8,
                'exchange_rate': 1.0
            })
        
        if self.config.modules_enabled.get('lymphatic', True):
            self.engine.register_module('lymphatic', LymphaticModule, {
                'n_vessels': 4,
                'metastasis_probability': 0.001
            })
        
        if self.config.modules_enabled.get('spatial', True):
            self.engine.register_module('spatial', SpatialModule, {
                'grid_size': (20, 20, 10),
                'resolution': 10.0
            })
        
        print("✓ Engine initialized with modules")
        print()
    
    def run_simulation(self, scenario='quick_start'):
        """Run simulation"""
        if not self.engine:
            self.initialize_engine()
        
        print(f"\nRunning {scenario} scenario...")
        
        # Initialize modules
        self.engine.initialize()
        
        # Link modules
        if 'molecular' in self.engine.modules and 'cellular' in self.engine.modules:
            molecular = self.engine.modules['molecular']
            cellular = self.engine.modules['cellular']
            
            # Add cells to molecular tracking
            for cell_id in cellular.cells.keys():
                molecular.add_cell(cell_id)
        
        if 'immune' in self.engine.modules and 'cellular' in self.engine.modules:
            immune = self.engine.modules['immune']
            cellular = self.engine.modules['cellular']
            
            # Link immune to cellular for target access
            immune.set_cellular_module(cellular)
        
        if 'vascular' in self.engine.modules and 'cellular' in self.engine.modules:
            vascular = self.engine.modules['vascular']
            cellular = self.engine.modules['cellular']
            
            # Link vascular to cellular for exchange
            vascular.set_cellular_module(cellular)
        
        if 'lymphatic' in self.engine.modules:
            lymphatic = self.engine.modules['lymphatic']
            
            if 'cellular' in self.engine.modules:
                lymphatic.set_cellular_module(self.engine.modules['cellular'])
            
            if 'immune' in self.engine.modules:
                lymphatic.set_immune_module(self.engine.modules['immune'])
        
        # Run simulation
        duration = 2.0 if scenario == 'quick_start' else self.config.duration
        self.engine.run(duration=duration)
        
        # Show results
        print("\nSimulation Results:")
        state = self.engine.get_state()
        for module_name, module_state in state.items():
            if module_name not in ['time', 'step_count', 'running']:
                print(f"\n{module_name}:")
                for key, value in module_state.items():
                    print(f"  {key}: {value}")
        
        print()
        input("Press Enter to continue...")
    
    def configure_settings(self):
        """Configure settings"""
        while True:
            choice = self.menu.show_settings_menu()
            
            if choice == '1':
                self.toggle_module()
            elif choice == '2':
                self.change_time_settings()
            elif choice == '3':
                self.change_space_settings()
            elif choice == '4':
                self.change_performance_settings()
            elif choice == '5':
                self.config = SimulationConfig()
                print("\n✓ Reset to defaults")
                input("Press Enter to continue...")
            elif choice == 'b':
                break
    
    def toggle_module(self):
        """Toggle module on/off"""
        print("\nAvailable modules:")
        modules = list(self.config.modules_enabled.keys())
        for i, name in enumerate(modules, 1):
            status = "✓" if self.config.modules_enabled[name] else "✗"
            print(f"  {i}. {status} {name}")
        
        print()
        choice = input("Module number to toggle (or 'b' to go back): ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(modules):
                name = modules[idx]
                self.config.modules_enabled[name] = not self.config.modules_enabled[name]
                status = "enabled" if self.config.modules_enabled[name] else "disabled"
                print(f"\n✓ {name} {status}")
        
        input("Press Enter to continue...")
    
    def change_time_settings(self):
        """Change time settings"""
        print("\nCurrent time settings:")
        print(f"  dt: {self.config.dt} hours")
        print(f"  duration: {self.config.duration} hours")
        print()
        
        try:
            dt = input(f"New dt (current: {self.config.dt}): ").strip()
            if dt:
                self.config.dt = float(dt)
            
            duration = input(f"New duration (current: {self.config.duration}): ").strip()
            if duration:
                self.config.duration = float(duration)
            
            print("\n✓ Time settings updated")
        except ValueError:
            print("\n✗ Invalid input")
        
        input("Press Enter to continue...")
    
    def change_space_settings(self):
        """Change space settings"""
        print("\nCurrent space settings:")
        print(f"  grid_size: {self.config.grid_size}")
        print(f"  resolution: {self.config.resolution} μm/voxel")
        print()
        
        print("(Space settings modification coming soon)")
        input("Press Enter to continue...")
    
    def change_performance_settings(self):
        """Change performance settings"""
        print("\nCurrent performance settings:")
        print(f"  GPU: {self.config.use_gpu}")
        print(f"  Threads: {self.config.num_threads}")
        print()
        
        use_gpu = input("Use GPU? (y/n): ").strip().lower()
        if use_gpu == 'y':
            self.config.use_gpu = True
        elif use_gpu == 'n':
            self.config.use_gpu = False
        
        try:
            threads = input(f"Number of threads (current: {self.config.num_threads}): ").strip()
            if threads:
                self.config.num_threads = int(threads)
            
            print("\n✓ Performance settings updated")
        except ValueError:
            print("\n✗ Invalid input")
        
        input("Press Enter to continue...")
    
    def run(self):
        """Run application"""
        print("\n" + "=" * 70)
        print("Welcome to cognisom!".center(70))
        print("Multi-Scale Cellular Simulation Platform".center(70))
        print("=" * 70)
        print()
        print("Status: Core architecture complete, modules being integrated")
        print()
        input("Press Enter to continue...")
        
        while True:
            choice = self.menu.show_main_menu()
            
            if choice == '1':
                scenario_choice = self.menu.show_simulation_menu()
                if scenario_choice == '1':
                    self.run_simulation('quick_start')
                elif scenario_choice == '2':
                    self.run_simulation('cancer_transmission')
                elif scenario_choice == '3':
                    self.run_simulation('immune_response')
                elif scenario_choice == '4':
                    self.run_simulation('tissue_development')
                elif scenario_choice == '5':
                    self.run_simulation('custom')
            
            elif choice == '2':
                self.configure_settings()
            
            elif choice == '3':
                print("\nResults viewing coming soon...")
                input("Press Enter to continue...")
            
            elif choice == '4':
                print("\nScenarios coming soon...")
                input("Press Enter to continue...")
            
            elif choice == '5':
                self.menu.show_module_status()
            
            elif choice == '6':
                self.menu.show_help()
            
            elif choice == 'q':
                print("\n" + "=" * 70)
                print("Thank you for using cognisom!".center(70))
                print("=" * 70)
                print()
                sys.exit(0)
            
            else:
                print("\n✗ Invalid choice")
                input("Press Enter to continue...")


def main():
    """Main entry point"""
    app = CognisomApp()
    app.run()


if __name__ == '__main__':
    main()
