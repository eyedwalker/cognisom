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
        
        # TODO: Register modules here when they're ready
        # self.engine.register_module('molecular', MolecularModule)
        # self.engine.register_module('cellular', CellularModule)
        # self.engine.register_module('immune', ImmuneModule)
        # self.engine.register_module('vascular', VascularModule)
        # self.engine.register_module('lymphatic', LymphaticModule)
        # self.engine.register_module('spatial', SpatialModule)
        
        print("✓ Engine initialized (modules will be added soon)")
        print()
    
    def run_simulation(self, scenario='quick_start'):
        """Run simulation"""
        if not self.engine:
            self.initialize_engine()
        
        print(f"\nRunning {scenario} scenario...")
        print("(Full simulation coming soon - modules being integrated)")
        print()
        
        # TODO: Initialize and run based on scenario
        # self.engine.initialize()
        # self.engine.run()
        
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
