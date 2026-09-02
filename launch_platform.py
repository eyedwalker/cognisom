#!/usr/bin/env python3
"""
cognisom Platform Launcher
===========================

Central hub to launch all components.

Components:
- REST API Server
- Web Dashboard
- GUI Control Panel
- Visualization
- All modules
"""

import subprocess
import webbrowser
import time
import sys
from pathlib import Path


class PlatformLauncher:
    """
    Launch all platform components
    """
    
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
    
    def launch_api_server(self):
        """Launch REST API server"""
        print("🚀 Launching REST API server...")
        proc = subprocess.Popen(
            [sys.executable, 'api/rest_server.py'],
            cwd=self.base_dir
        )
        self.processes.append(('API Server', proc))
        print("   ✓ API Server started on http://localhost:5000")
        return proc
    
    def launch_web_dashboard(self):
        """Open web dashboard in browser"""
        print("🌐 Opening web dashboard...")
        dashboard_path = self.base_dir / 'web' / 'dashboard.html'
        webbrowser.open(f'file://{dashboard_path}')
        print("   ✓ Dashboard opened in browser")
    
    def launch_gui_panel(self):
        """Launch GUI control panel"""
        print("🖥️  Launching GUI control panel...")
        proc = subprocess.Popen(
            [sys.executable, 'ui/control_panel.py'],
            cwd=self.base_dir
        )
        self.processes.append(('GUI Panel', proc))
        print("   ✓ GUI Panel launched")
        return proc
    
    def launch_visualization(self):
        """Launch complete visualization"""
        print("📊 Launching visualization...")
        proc = subprocess.Popen(
            [sys.executable, 'visualize_complete.py'],
            cwd=self.base_dir
        )
        self.processes.append(('Visualization', proc))
        print("   ✓ Visualization launched")
        return proc
    
    def show_menu(self):
        """Show launch menu"""
        print("=" * 70)
        print("🧬 cognisom Platform Launcher")
        print("=" * 70)
        print()
        print("Select components to launch:")
        print()
        print("  1. 🚀 Full Platform (API + Web Dashboard + GUI)")
        print("  2. 🌐 Web Platform (API + Dashboard)")
        print("  3. 🖥️  Desktop Platform (GUI + Visualization)")
        print("  4. 🔬 API Server Only")
        print("  5. 📊 Visualization Only")
        print("  6. 🎯 Run Scenario")
        print("  7. 📝 Generate Report")
        print("  q. Quit")
        print()
        
        choice = input("Enter choice: ").strip()
        return choice
    
    def launch_full_platform(self):
        """Launch everything"""
        print("\n" + "=" * 70)
        print("Launching Full Platform...")
        print("=" * 70)
        print()
        
        # Launch API server
        self.launch_api_server()
        time.sleep(2)  # Wait for server to start
        
        # Launch web dashboard
        self.launch_web_dashboard()
        
        # Launch GUI
        self.launch_gui_panel()
        
        print()
        print("=" * 70)
        print("✓ Full Platform Launched!")
        print("=" * 70)
        print()
        print("Access points:")
        print("  - Web Dashboard: file://web/dashboard.html")
        print("  - API Server: http://localhost:5000")
        print("  - GUI Panel: Desktop window")
        print()
        print("Press Ctrl+C to stop all components")
        print()
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.cleanup()
    
    def launch_web_platform(self):
        """Launch web components"""
        print("\n" + "=" * 70)
        print("Launching Web Platform...")
        print("=" * 70)
        print()
        
        self.launch_api_server()
        time.sleep(2)
        self.launch_web_dashboard()
        
        print()
        print("✓ Web Platform Launched!")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.cleanup()
    
    def launch_desktop_platform(self):
        """Launch desktop components"""
        print("\n" + "=" * 70)
        print("Launching Desktop Platform...")
        print("=" * 70)
        print()
        
        self.launch_gui_panel()
        time.sleep(1)
        self.launch_visualization()
        
        print()
        print("✓ Desktop Platform Launched!")
        print("Close windows to exit")
        print()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.cleanup()
    
    def run_scenario_menu(self):
        """Show scenario menu"""
        print("\n" + "=" * 70)
        print("Available Scenarios:")
        print("=" * 70)
        print()
        print("  1. 💉 Cancer Immunotherapy")
        print("  2. 🕐 Chronotherapy")
        print("  3. 🫁 Hypoxia Response")
        print("  4. 🧬 Epigenetic Therapy")
        print("  5. ⏰ Circadian Disruption")
        print()
        
        choice = input("Select scenario (1-5): ").strip()
        
        scenarios = {
            '1': 'immunotherapy',
            '2': 'chronotherapy',
            '3': 'hypoxia',
            '4': 'epigenetic_therapy',
            '5': 'circadian_disruption'
        }
        
        if choice in scenarios:
            scenario = scenarios[choice]
            print(f"\nRunning {scenario} scenario...")
            subprocess.run([sys.executable, f'scenarios/{scenario}.py'], cwd=self.base_dir)
        else:
            print("Invalid choice")
    
    def generate_report_menu(self):
        """Generate report"""
        print("\n" + "=" * 70)
        print("Generate Report")
        print("=" * 70)
        print()
        
        from cognisom.api.publisher import Publisher
        from cognisom.core import SimulationEngine, SimulationConfig
        from cognisom.modules import CellularModule, ImmuneModule
        
        print("Running quick simulation...")
        engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
        engine.register_module('cellular', CellularModule)
        engine.register_module('immune', ImmuneModule)
        engine.initialize()
        
        immune = engine.modules['immune']
        cellular = engine.modules['cellular']
        immune.set_cellular_module(cellular)
        
        engine.run()
        
        print("\nGenerating reports...")
        publisher = Publisher(engine)
        files = publisher.generate_all_formats('platform_report')
        
        print("\n✓ Reports generated:")
        for format_type, filename in files.items():
            print(f"  - {format_type}: {filename}")
        print()
    
    def cleanup(self):
        """Clean up processes"""
        print("Stopping all components...")
        for name, proc in self.processes:
            print(f"  Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("✓ All components stopped")
    
    def run(self):
        """Run launcher"""
        while True:
            choice = self.show_menu()
            
            if choice == '1':
                self.launch_full_platform()
                break
            elif choice == '2':
                self.launch_web_platform()
                break
            elif choice == '3':
                self.launch_desktop_platform()
                break
            elif choice == '4':
                self.launch_api_server()
                print("\nAPI Server running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.cleanup()
                break
            elif choice == '5':
                self.launch_visualization()
                print("\nVisualization running. Close window to exit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.cleanup()
                break
            elif choice == '6':
                self.run_scenario_menu()
            elif choice == '7':
                self.generate_report_menu()
            elif choice.lower() == 'q':
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice. Try again.\n")


if __name__ == '__main__':
    launcher = PlatformLauncher()
    launcher.run()
