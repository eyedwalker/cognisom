#!/usr/bin/env python3
"""
Menu System
===========

Interactive CLI menu for cognisom.
"""

import sys
import os


class MenuSystem:
    """
    Interactive menu system for cognisom
    """
    
    def __init__(self, engine=None):
        self.engine = engine
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def show_header(self, title):
        """Show header"""
        print("\n" + "=" * 70)
        print(title.center(70))
        print("=" * 70)
        print()
    
    def show_main_menu(self):
        """Display main menu"""
        self.clear_screen()
        self.show_header("cognisom: Multi-Scale Cellular Simulation Platform")
        
        print("1. Run Simulation")
        print("2. Configure Settings")
        print("3. View Results")
        print("4. Run Scenario")
        print("5. Module Status")
        print("6. Help")
        print("q. Quit")
        print()
        
        return input("Choice: ").strip().lower()
    
    def show_simulation_menu(self):
        """Simulation configuration menu"""
        self.clear_screen()
        self.show_header("Run Simulation")
        
        print("1. Quick Start (Default settings)")
        print("2. Cancer Transmission Scenario")
        print("3. Immune Response Scenario")
        print("4. Tissue Development Scenario")
        print("5. Custom Configuration")
        print("b. Back to Main Menu")
        print()
        
        return input("Choice: ").strip().lower()
    
    def show_settings_menu(self):
        """Settings configuration menu"""
        self.clear_screen()
        self.show_header("Settings")
        
        if self.engine:
            print("Modules:")
            for name, module in self.engine.modules.items():
                status = "✓" if module.enabled else "✗"
                print(f"  {status} {name}")
            print()
            
            print("Time Settings:")
            print(f"  dt: {self.engine.config.dt} hours")
            print(f"  duration: {self.engine.config.duration} hours")
            print()
            
            print("Space Settings:")
            print(f"  grid_size: {self.engine.config.grid_size}")
            print(f"  resolution: {self.engine.config.resolution} μm/voxel")
            print()
            
            print("Performance:")
            print(f"  GPU: {self.engine.config.use_gpu}")
            print(f"  Threads: {self.engine.config.num_threads}")
            print()
        
        print("1. Toggle Module")
        print("2. Change Time Settings")
        print("3. Change Space Settings")
        print("4. Performance Settings")
        print("5. Reset to Defaults")
        print("b. Back to Main Menu")
        print()
        
        return input("Choice: ").strip().lower()
    
    def show_module_status(self):
        """Show module status"""
        self.clear_screen()
        self.show_header("Module Status")
        
        if self.engine:
            for name, module in self.engine.modules.items():
                print(f"\n{name}:")
                print(f"  Enabled: {module.enabled}")
                print(f"  State: {module.get_state()}")
        
        print()
        input("Press Enter to continue...")
    
    def show_help(self):
        """Show help"""
        self.clear_screen()
        self.show_header("Help")
        
        print("""
cognisom is a multi-scale cellular simulation platform that models
biological systems from molecules to tissues.

Features:
- Molecular level: DNA/RNA with actual sequences
- Cellular level: Metabolism, division, death
- Tissue level: Vasculature, lymphatics, immune system
- Real-time visualization
- Modular architecture

Modules:
- molecular: DNA/RNA dynamics, exosome transfer
- cellular: Cell cycle, metabolism
- immune: T cells, NK cells, macrophages
- vascular: Capillaries, O2/nutrient exchange
- lymphatic: Drainage, metastasis
- spatial: 3D diffusion, gradients

Usage:
1. Select "Run Simulation" to start
2. Choose a scenario or custom configuration
3. View results in real-time
4. Adjust parameters as needed

For more information, see documentation in the repository.
        """)
        
        print()
        input("Press Enter to continue...")


# Test
if __name__ == '__main__':
    menu = MenuSystem()
    
    while True:
        choice = menu.show_main_menu()
        
        if choice == '1':
            print("\nSimulation would run here...")
            input("Press Enter to continue...")
        elif choice == '2':
            menu.show_settings_menu()
        elif choice == '5':
            menu.show_module_status()
        elif choice == '6':
            menu.show_help()
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice")
            input("Press Enter to continue...")
