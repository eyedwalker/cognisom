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

from flask import Flask, request, jsonify, send_file, g
from flask_cors import CORS
import os
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
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32).hex())

# CORS: restrict to configured origins
_default_origins = "https://cognisom.com,https://www.cognisom.com,http://localhost:8501,http://localhost:8080"
_allowed_origins = os.environ.get("CORS_ORIGINS", _default_origins).split(",")
CORS(app, origins=[o.strip() for o in _allowed_origins])

# ── Auth integration ─────────────────────────────────────────────────

# Add project root to path for cognisom imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from cognisom.auth import AuthManager
    _auth = AuthManager(data_dir=os.path.join(_project_root, "data", "auth"))
    _AUTH_ENABLED = True
except ImportError:
    _auth = None
    _AUTH_ENABLED = False


def _authenticate():
    """Authenticate request via session token or API key.

    Returns User or None. Does NOT block — endpoints decide whether to require auth.
    """
    if not _AUTH_ENABLED:
        return None

    # Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        user = _auth.validate_session(auth_header[7:])
        if user:
            return user

    # API key header
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return _auth.validate_api_key(api_key)

    # API key query param
    api_key = request.args.get("api_key", "")
    if api_key:
        return _auth.validate_api_key(api_key)

    return None


@app.before_request
def _before_request():
    """Set g.current_user on every request."""
    g.current_user = _authenticate()


def _require_auth(f):
    """Decorator: require authentication for an endpoint."""
    import functools

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if g.current_user is None:
            return jsonify({"error": "Authentication required. Provide Bearer token or X-API-Key header."}), 401
        return f(*args, **kwargs)
    return decorated


def _require_permission(perm):
    """Decorator: require a specific permission."""
    def decorator(f):
        import functools

        @functools.wraps(f)
        def decorated(*args, **kwargs):
            user = g.current_user
            if user is None:
                return jsonify({"error": "Authentication required"}), 401
            if not user.has_permission(perm):
                return jsonify({"error": f"Permission denied: {perm}"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

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
@_require_permission("simulation:run")
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
@_require_permission("simulation:run")
def stop_simulation():
    """Stop simulation"""
    global running
    
    running = False
    
    return jsonify({
        'status': 'success',
        'message': 'Simulation stopped'
    })


@app.route('/api/simulation/reset', methods=['POST'])
@_require_permission("simulation:run")
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
@_require_auth
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
@_require_permission("simulation:configure")
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
@_require_permission("simulation:run")
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
@_require_permission("simulation:export")
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
@_require_auth
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


# ── Auth API endpoints ───────────────────────────────────────────────


@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user account."""
    if not _AUTH_ENABLED:
        return jsonify({'error': 'Auth not available'}), 503

    data = request.json or {}
    ok, msg = _auth.register(
        username=data.get('username', ''),
        email=data.get('email', ''),
        password=data.get('password', ''),
        display_name=data.get('display_name', ''),
    )
    if ok:
        return jsonify({'status': 'success', 'message': msg})
    return jsonify({'error': msg}), 400


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Log in and get a session token."""
    if not _AUTH_ENABLED:
        return jsonify({'error': 'Auth not available'}), 503

    data = request.json or {}
    session, msg = _auth.login(
        username=data.get('username', ''),
        password=data.get('password', ''),
        ip_address=request.remote_addr or '',
        user_agent=request.headers.get('User-Agent', ''),
    )
    if session:
        return jsonify({
            'status': 'success',
            'session_id': session.session_id,
            'expires_at': session.expires_at,
        })
    return jsonify({'error': msg}), 401


@app.route('/api/auth/logout', methods=['POST'])
@_require_auth
def api_logout():
    """Invalidate current session."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        _auth.logout(auth_header[7:])
    return jsonify({'status': 'success', 'message': 'Logged out'})


@app.route('/api/auth/me', methods=['GET'])
@_require_auth
def api_me():
    """Get current user profile."""
    return jsonify({'status': 'success', 'user': g.current_user.to_public_dict()})


# ── Entity Library API endpoints ────────────────────────────────────────

_entity_manager = None


def _get_entity_manager():
    """Get or create entity manager."""
    global _entity_manager
    if _entity_manager is None:
        try:
            from cognisom.library.models import EntityManager
            _entity_manager = EntityManager(data_dir=os.path.join(_project_root, "data", "entities"))
        except ImportError:
            return None
    return _entity_manager


@app.route('/api/entities', methods=['GET'])
@_require_auth
def list_entities():
    """List and search entities.

    Query params:
        type: Filter by entity type (gene, protein, drug, pathway, etc.)
        q: Search query
        limit: Max results (default 100)
        offset: Pagination offset
    """
    manager = _get_entity_manager()
    if manager is None:
        return jsonify({'error': 'Entity library not available'}), 503

    entity_type = request.args.get('type')
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    try:
        if query:
            entities = manager.search(query, entity_type=entity_type, limit=limit)
        else:
            entities = manager.list_all(entity_type=entity_type, limit=limit, offset=offset)

        return jsonify({
            'status': 'success',
            'count': len(entities),
            'entities': [e.to_dict() for e in entities]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/entities/<entity_id>', methods=['GET'])
@_require_auth
def get_entity(entity_id: str):
    """Get a single entity by ID."""
    manager = _get_entity_manager()
    if manager is None:
        return jsonify({'error': 'Entity library not available'}), 503

    try:
        entity = manager.get(entity_id)
        if entity is None:
            return jsonify({'error': 'Entity not found'}), 404

        return jsonify({
            'status': 'success',
            'entity': entity.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/entities', methods=['POST'])
@_require_permission("library:write")
def create_entity():
    """Create a new entity.

    Request body:
        type: Entity type (required)
        name: Entity name (required)
        ... other fields depend on entity type
    """
    manager = _get_entity_manager()
    if manager is None:
        return jsonify({'error': 'Entity library not available'}), 503

    data = request.json or {}

    if 'type' not in data or 'name' not in data:
        return jsonify({'error': 'type and name are required'}), 400

    try:
        entity = manager.create(
            entity_type=data.pop('type'),
            name=data.pop('name'),
            **data
        )
        return jsonify({
            'status': 'success',
            'entity': entity.to_dict()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/entities/<entity_id>', methods=['PUT'])
@_require_permission("library:write")
def update_entity(entity_id: str):
    """Update an existing entity."""
    manager = _get_entity_manager()
    if manager is None:
        return jsonify({'error': 'Entity library not available'}), 503

    data = request.json or {}

    try:
        entity = manager.update(entity_id, **data)
        if entity is None:
            return jsonify({'error': 'Entity not found'}), 404

        return jsonify({
            'status': 'success',
            'entity': entity.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/entities/<entity_id>', methods=['DELETE'])
@_require_permission("library:delete")
def delete_entity(entity_id: str):
    """Delete an entity."""
    manager = _get_entity_manager()
    if manager is None:
        return jsonify({'error': 'Entity library not available'}), 503

    try:
        success = manager.delete(entity_id)
        if not success:
            return jsonify({'error': 'Entity not found'}), 404

        return jsonify({
            'status': 'success',
            'message': f'Entity {entity_id} deleted'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/entities/<entity_id>/relationships', methods=['GET'])
@_require_auth
def get_entity_relationships(entity_id: str):
    """Get relationships for an entity."""
    manager = _get_entity_manager()
    if manager is None:
        return jsonify({'error': 'Entity library not available'}), 503

    try:
        entity = manager.get(entity_id)
        if entity is None:
            return jsonify({'error': 'Entity not found'}), 404

        relationships = manager.get_relationships(entity_id)
        return jsonify({
            'status': 'success',
            'entity_id': entity_id,
            'relationships': relationships
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── External Data Sources API endpoints ────────────────────────────────


@app.route('/api/external/kegg/pathways', methods=['GET'])
@_require_auth
def kegg_search_pathways():
    """Search KEGG pathways."""
    query = request.args.get('q', '')
    organism = request.args.get('organism', 'hsa')

    if not query:
        return jsonify({'error': 'Query parameter q is required'}), 400

    try:
        from cognisom.library.external_sources import KEGGClient
        kegg = KEGGClient()
        pathways = kegg.search_pathways(query, organism=organism)
        return jsonify({
            'status': 'success',
            'count': len(pathways),
            'pathways': [{'id': p.pathway_id, 'name': p.name, 'url': p.url} for p in pathways]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/external/kegg/pathways/<pathway_id>', methods=['GET'])
@_require_auth
def kegg_get_pathway(pathway_id: str):
    """Get KEGG pathway details."""
    try:
        from cognisom.library.external_sources import KEGGClient
        kegg = KEGGClient()
        pathway = kegg.get_pathway(pathway_id)
        if pathway:
            return jsonify({
                'status': 'success',
                'pathway': {
                    'id': pathway.pathway_id,
                    'name': pathway.name,
                    'description': pathway.description,
                    'genes': pathway.genes[:100],
                    'compounds': pathway.compounds,
                    'url': pathway.url,
                }
            })
        return jsonify({'error': 'Pathway not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/external/pubchem/compounds', methods=['GET'])
@_require_auth
def pubchem_search_compounds():
    """Search PubChem compounds."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 5))

    if not query:
        return jsonify({'error': 'Query parameter q is required'}), 400

    try:
        from cognisom.library.external_sources import PubChemClient
        pubchem = PubChemClient()
        compounds = pubchem.search_compounds(query, limit=limit)
        return jsonify({
            'status': 'success',
            'count': len(compounds),
            'compounds': [{
                'cid': c.cid,
                'name': c.name,
                'smiles': c.smiles,
                'formula': c.molecular_formula,
                'weight': c.molecular_weight,
                'url': c.url,
            } for c in compounds]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/external/string/interactions', methods=['POST'])
@_require_auth
def string_get_interactions():
    """Get STRING protein interactions."""
    data = request.json or {}
    proteins = data.get('proteins', [])
    threshold = data.get('score_threshold', 400)

    if not proteins:
        return jsonify({'error': 'proteins list is required'}), 400

    try:
        from cognisom.library.external_sources import STRINGClient
        string = STRINGClient()
        interactions = string.get_interactions(proteins, score_threshold=threshold)
        return jsonify({
            'status': 'success',
            'count': len(interactions),
            'interactions': [{
                'protein_a': i.gene_a,
                'protein_b': i.gene_b,
                'score': i.combined_score,
            } for i in interactions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        'name': 'cognisom REST API',
        'version': '1.0.0',
        'auth_enabled': _AUTH_ENABLED,
        'endpoints': {
            'POST /api/auth/register': 'Register new account',
            'POST /api/auth/login': 'Login (returns session token)',
            'POST /api/auth/logout': 'Logout (invalidate session)',
            'GET  /api/auth/me': 'Current user profile',
            'POST /api/simulation/start': 'Start simulation (auth required)',
            'POST /api/simulation/stop': 'Stop simulation (auth required)',
            'POST /api/simulation/reset': 'Reset simulation (auth required)',
            'GET  /api/simulation/state': 'Get current state',
            'POST /api/simulation/parameter': 'Set parameter (admin only)',
            'POST /api/simulation/scenario': 'Run scenario (auth required)',
            'GET  /api/simulation/export': 'Export data (auth required)',
            'GET  /api/modules': 'List modules',
            'GET  /api/scenarios': 'List scenarios',
            'GET  /api/health': 'Health check',
            'GET  /api/entities': 'List/search entities',
            'GET  /api/entities/<id>': 'Get entity by ID',
            'POST /api/entities': 'Create entity',
            'PUT  /api/entities/<id>': 'Update entity',
            'DELETE /api/entities/<id>': 'Delete entity',
            'GET  /api/entities/<id>/relationships': 'Get entity relationships',
            'GET  /api/external/kegg/pathways': 'Search KEGG pathways',
            'GET  /api/external/kegg/pathways/<id>': 'Get KEGG pathway details',
            'GET  /api/external/pubchem/compounds': 'Search PubChem compounds',
            'POST /api/external/string/interactions': 'Get STRING interactions'
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
    
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)
