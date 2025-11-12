#!/usr/bin/env python3
"""
REST API Server
===============

Flask-based REST API for remote simulation control.

Endpoints:
- POST /api/simulation/start
- POST /api/simulation/stop
- GET  /api/simulation/state
- POST /api/simulation/parameter
- GET  /api/simulation/results
- POST /api/simulation/scenario
- GET  /api/simulation/export
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import time
import sys
sys.path.insert(0, '..')

from core import SimulationEngine, SimulationConfig
from modules import (MolecularModule, CellularModule, ImmuneModule,
                    VascularModule, LymphaticModule, SpatialModule,
                    EpigeneticModule, CircadianModule, MorphogenModule)
from scenarios import *

app = Flask(__name__)
CORS(app)  # Enable CORS for web dashboard

# Global simulation engine
engine = None
sim_thread = None
running = False


def initialize_engine():
    """Initialize simulation engine with all modules"""
    global engine
    
    config = SimulationConfig(dt=0.01, duration=24.0)
    engine = SimulationEngine(config)
    
    # Register all modules
    engine.register_module('molecular', MolecularModule)
    engine.register_module('cellular', CellularModule, {
        'n_normal_cells': 20,
        'n_cancer_cells': 5
    })
    engine.register_module('immune', ImmuneModule, {
        'n_t_cells': 8,
        'n_nk_cells': 5,
        'n_macrophages': 3
    })
    engine.register_module('vascular', VascularModule, {
        'n_capillaries': 8
    })
    engine.register_module('lymphatic', LymphaticModule, {
        'n_vessels': 4
    })
    engine.register_module('spatial', SpatialModule)
    engine.register_module('epigenetic', EpigeneticModule)
    engine.register_module('circadian', CircadianModule)
    engine.register_module('morphogen', MorphogenModule)
    
    # Initialize
    engine.initialize()
    
    # Link modules
    molecular = engine.modules['molecular']
    cellular = engine.modules['cellular']
    immune = engine.modules['immune']
    vascular = engine.modules['vascular']
    lymphatic = engine.modules['lymphatic']
    epigenetic = engine.modules['epigenetic']
    circadian = engine.modules['circadian']
    morphogen = engine.modules['morphogen']
    
    for cell_id, cell in cellular.cells.items():
        molecular.add_cell(cell_id)
        epigenetic.add_cell(cell_id, cell.cell_type)
        circadian.add_cell(cell_id)
        morphogen.add_cell(cell_id, cell.position)
    
    immune.set_cellular_module(cellular)
    vascular.set_cellular_module(cellular)
    lymphatic.set_cellular_module(cellular)
    lymphatic.set_immune_module(immune)
    
    return engine


def run_simulation():
    """Run simulation in background thread"""
    global running
    while running:
        engine.step()
        time.sleep(0.01)


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start simulation"""
    global engine, sim_thread, running
    
    if engine is None:
        initialize_engine()
    
    if not running:
        running = True
        sim_thread = threading.Thread(target=run_simulation)
        sim_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Simulation started'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Simulation already running'
        }), 400


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop simulation"""
    global running
    
    running = False
    
    return jsonify({
        'status': 'success',
        'message': 'Simulation stopped'
    })


@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation"""
    global engine, running
    
    running = False
    
    if engine:
        engine.reset()
    
    return jsonify({
        'status': 'success',
        'message': 'Simulation reset'
    })


@app.route('/api/simulation/state', methods=['GET'])
def get_state():
    """Get current simulation state"""
    if engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Engine not initialized'
        }), 400
    
    state = engine.get_state()
    
    # Convert numpy types
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    state = convert(state)
    
    return jsonify({
        'status': 'success',
        'data': state
    })


@app.route('/api/simulation/parameter', methods=['POST'])
def set_parameter():
    """Set simulation parameter"""
    data = request.json
    
    module_name = data.get('module')
    param_name = data.get('parameter')
    value = data.get('value')
    
    if not all([module_name, param_name, value is not None]):
        return jsonify({
            'status': 'error',
            'message': 'Missing required fields'
        }), 400
    
    try:
        engine.set_parameter(module_name, param_name, value)
        
        return jsonify({
            'status': 'success',
            'message': f'Parameter {module_name}.{param_name} set to {value}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/simulation/scenario', methods=['POST'])
def run_scenario():
    """Run pre-built scenario"""
    data = request.json
    scenario_name = data.get('scenario')
    
    scenarios = {
        'immunotherapy': run_immunotherapy_scenario,
        'chronotherapy': run_chronotherapy_scenario,
        'hypoxia': run_hypoxia_scenario,
        'epigenetic_therapy': run_epigenetic_therapy_scenario,
        'circadian_disruption': run_circadian_disruption_scenario
    }
    
    if scenario_name not in scenarios:
        return jsonify({
            'status': 'error',
            'message': f'Unknown scenario: {scenario_name}'
        }), 400
    
    try:
        result = scenarios[scenario_name]()
        
        # Convert result
        def convert(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        result = convert(result)
        
        return jsonify({
            'status': 'success',
            'scenario': scenario_name,
            'results': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/simulation/export', methods=['GET'])
def export_data():
    """Export simulation data"""
    format_type = request.args.get('format', 'json')
    
    if engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Engine not initialized'
        }), 400
    
    if format_type == 'csv':
        engine.export_to_csv('export.csv')
        return send_file('export.csv', as_attachment=True)
    elif format_type == 'json':
        engine.export_to_json('export.json')
        return send_file('export.json', as_attachment=True)
    else:
        return jsonify({
            'status': 'error',
            'message': f'Unknown format: {format_type}'
        }), 400


@app.route('/api/modules', methods=['GET'])
def get_modules():
    """Get available modules"""
    if engine is None:
        initialize_engine()
    
    modules = {}
    for name, module in engine.modules.items():
        modules[name] = {
            'name': name,
            'enabled': module.enabled,
            'description': module.__class__.__doc__.split('\n')[0] if module.__class__.__doc__ else ''
        }
    
    return jsonify({
        'status': 'success',
        'modules': modules
    })


@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Get available scenarios"""
    scenarios = [
        {
            'id': 'immunotherapy',
            'name': 'Cancer Immunotherapy',
            'description': 'Boost immune system to fight cancer'
        },
        {
            'id': 'chronotherapy',
            'name': 'Chronotherapy',
            'description': 'Time treatment to circadian rhythm'
        },
        {
            'id': 'hypoxia',
            'name': 'Hypoxia Response',
            'description': 'Low oxygen environment'
        },
        {
            'id': 'epigenetic_therapy',
            'name': 'Epigenetic Therapy',
            'description': 'DNA methyltransferase inhibitors'
        },
        {
            'id': 'circadian_disruption',
            'name': 'Circadian Disruption',
            'description': 'Jet lag simulation'
        }
    ]
    
    return jsonify({
        'status': 'success',
        'scenarios': scenarios
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'cognisom API is running',
        'version': '1.0.0'
    })


@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        'name': 'cognisom REST API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/simulation/start': 'Start simulation',
            'POST /api/simulation/stop': 'Stop simulation',
            'POST /api/simulation/reset': 'Reset simulation',
            'GET  /api/simulation/state': 'Get current state',
            'POST /api/simulation/parameter': 'Set parameter',
            'POST /api/simulation/scenario': 'Run scenario',
            'GET  /api/simulation/export': 'Export data',
            'GET  /api/modules': 'List modules',
            'GET  /api/scenarios': 'List scenarios',
            'GET  /api/health': 'Health check'
        }
    })


if __name__ == '__main__':
    print("=" * 70)
    print("cognisom REST API Server")
    print("=" * 70)
    print()
    print("Starting server on http://localhost:5000")
    print()
    print("Endpoints:")
    print("  POST http://localhost:5000/api/simulation/start")
    print("  POST http://localhost:5000/api/simulation/stop")
    print("  GET  http://localhost:5000/api/simulation/state")
    print("  POST http://localhost:5000/api/simulation/parameter")
    print("  POST http://localhost:5000/api/simulation/scenario")
    print("  GET  http://localhost:5000/api/simulation/export")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
