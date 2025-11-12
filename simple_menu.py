#!/usr/bin/env python3
"""
Simple Text Menu
================

Text-based menu that works in terminal.
"""

from core import SimulationEngine, SimulationConfig
from modules import CellularModule, ImmuneModule, VascularModule

def show_menu():
    """Show main menu"""
    print("\n" + "=" * 70)
    print("🧬 cognisom - Simple Control Menu")
    print("=" * 70)
    print()
    print("1. Run Quick Simulation (0.5 hours)")
    print("2. Run Full Simulation (24 hours)")
    print("3. View Current State")
    print("4. Export Data (CSV)")
    print("5. Export Data (JSON)")
    print("6. Run Immunotherapy Scenario")
    print("7. Run Hypoxia Scenario")
    print("8. Generate Report")
    print("9. Show Statistics")
    print("q. Quit")
    print()
    return input("Enter choice: ").strip()

def main():
    """Main menu loop"""
    # Create engine
    print("\nInitializing engine...")
    engine = SimulationEngine(SimulationConfig(dt=0.01))
    
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 5
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 8,
        'n_nk_cells': 5
    })
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 6
    })
    
    engine.initialize()
    
    # Link modules
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    vascular = engine.modules['vascular']
    
    immune.set_cellular_module(cellular)
    vascular.set_cellular_module(cellular)
    
    print("✓ Engine ready!")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            print("\n🚀 Running quick simulation (0.5 hours)...")
            engine.run(duration=0.5)
            print("✓ Complete!")
            
        elif choice == '2':
            print("\n🚀 Running full simulation (24 hours)...")
            print("This may take a minute...")
            engine.run(duration=24.0)
            print("✓ Complete!")
            
        elif choice == '3':
            state = engine.get_state()
            print("\n" + "=" * 70)
            print("CURRENT STATE")
            print("=" * 70)
            print(f"\nTime: {state['time']:.2f} hours")
            print(f"Steps: {state['step_count']}")
            print(f"\nCells: {state['cellular']['n_cells']}")
            print(f"  Cancer: {state['cellular']['n_cancer']}")
            print(f"  Normal: {state['cellular']['n_normal']}")
            print(f"  Deaths: {state['cellular']['total_deaths']}")
            print(f"\nImmune: {state['immune']['n_immune_cells']}")
            print(f"  Activated: {state['immune']['n_activated']}")
            print(f"  Kills: {state['immune']['total_kills']}")
            print(f"\nVascular:")
            print(f"  Avg O2: {state['vascular']['avg_cell_O2']:.3f}")
            print(f"  Hypoxic: {state['vascular']['hypoxic_regions']}")
            
        elif choice == '4':
            engine.export_to_csv('simulation_results.csv')
            print("\n✓ Exported to simulation_results.csv")
            
        elif choice == '5':
            engine.export_to_json('simulation_results.json')
            print("\n✓ Exported to simulation_results.json")
            
        elif choice == '6':
            print("\n💉 Running immunotherapy scenario...")
            from scenarios import run_immunotherapy_scenario
            result = run_immunotherapy_scenario()
            print(f"\n✓ Complete! Cancer cells: {result['cellular']['n_cancer']}")
            
        elif choice == '7':
            print("\n🫁 Running hypoxia scenario...")
            from scenarios import run_hypoxia_scenario
            result = run_hypoxia_scenario()
            print(f"\n✓ Complete! Hypoxic regions: {result['vascular']['hypoxic_regions']}")
            
        elif choice == '8':
            print("\n📑 Generating report...")
            from api.publisher import Publisher
            publisher = Publisher(engine)
            files = publisher.generate_all_formats('report')
            print("\n✓ Reports generated:")
            for fmt, filename in files.items():
                print(f"  - {filename}")
            
        elif choice == '9':
            state = engine.get_state()
            print("\n" + "=" * 70)
            print("DETAILED STATISTICS")
            print("=" * 70)
            
            for module_name, module_state in state.items():
                if module_name not in ['time', 'step_count', 'running']:
                    print(f"\n{module_name.upper()}:")
                    for key, value in module_state.items():
                        if isinstance(value, (int, float, str)):
                            print(f"  {key}: {value}")
            
        elif choice.lower() == 'q':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice. Try again.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
