import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Test configuration
TEST_CONFIG = {
    'test_data_dir': os.path.join(os.path.dirname(__file__), 'test_data'),
    'temp_output_dir': os.path.join(os.path.dirname(__file__), 'temp_output'),
    'max_test_runtime': 30,  # seconds
}

# Ensure test directories exist
os.makedirs(TEST_CONFIG['test_data_dir'], exist_ok=True)
os.makedirs(TEST_CONFIG['temp_output_dir'], exist_ok=True)


def create_test_wing_parameters():
    """Create sample wing parameters for testing."""
    return {
        'main_wing_angle': 15.0,
        'flap_angle': 25.0,
        'wing_span': 1800.0,  # mm
        'chord_length': 300.0,  # mm
        'endplate_height': 200.0,  # mm
        'thickness': 5.0,  # mm
    }


def cleanup_test_files():
    """Clean up temporary test files."""
    import shutil
    temp_dir = TEST_CONFIG['temp_output_dir']
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)


# Test fixtures for common objects
def create_mock_config():
    """Create a mock configuration for testing."""
    return {
        "max_generations": 5,
        "population_size": 3,
        "neural_network_enabled": False,  # Disable for testing
        "save_frequency": 1,
        "max_runtime_hours": 0.1,  # Short runtime for tests
        
        "early_stopping": {
            "patience": 5,
            "min_delta": 0.01,
            "monitor": "fitness",
            "stagnation_threshold": 3,
            "convergence_threshold": 0.01
        },
        
        "genetic_algorithm": {
            "crossover_rate": 0.8,
            "mutation_rate": 0.7,
            "elite_ratio": 0.2,
            "tournament_size": 2
        },
        
        "cfd_analysis": {
            "enabled": False,  # Disable CFD for unit tests
            "parallel_processes": 1,
            "timeout_seconds": 10,
            "smart_skipping": True
        },
        
        "output": {
            "save_all_stl": False,
            "save_best_only": True,
            "generate_reports": False,
            "create_visualizations": False
        }
    }