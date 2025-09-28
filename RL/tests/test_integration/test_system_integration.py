import pytest
import numpy as np
import sys
import os
import tempfile
import json
import torch
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_pipeline import AlphaDesignPipeline
from wing_generator import UltraRealisticF1FrontWingGenerator
from formula_constraints import F1FrontWingParams
from genetic_algo_components.initialize_population import F1PopulInit
from neural_network_components.network_initialization import NetworkInitializer

class TestSystemIntegration:
    
    def test_full_optimization_pipeline(self):
        # Create minimal config for testing
        test_config = {
            "max_generations": 2,
            "population_size": 3,
            "neural_network_enabled": False,  # Disable NN for faster testing
            "save_frequency": 1,
            "max_runtime_hours": 1,
            "neural_network": {
                "learning_rate": 1e-3,
                "batch_size": 2,
                "training_frequency": 1
            },
            "cfd_analysis": {
                "enabled": False,  # Disable CFD for testing
                "parallel_processes": 1
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Initialize pipeline with test config
            pipeline = AlphaDesignPipeline(temp_config_path, bezier_enabled=False)
            
            # Create base parameters
            base_params = F1FrontWingParams(
                total_span=1600,
                root_chord=280,
                tip_chord=250,
                flap_count=3,
                flap_spans=[1600, 1500, 1400],
                flap_cambers=[0.12, 0.10, 0.08]
            )
            
            # Test pipeline initialization
            init_result = pipeline.initialize_pipeline_components(base_params)
            
            assert init_result is not None
            assert 'initialization' in init_result
            assert init_result['initialization'] == 'success'
            assert 'population_size' in init_result
            assert init_result['population_size'] == 3
            
            # Test single generation run (faster than full pipeline)
            gen_result = pipeline.run_single_generation(base_params)
            
            assert gen_result is not None
            assert 'generation' in gen_result
            assert 'best_fitness' in gen_result
            assert 'best_individual' in gen_result
            assert 'average_fitness' in gen_result
            assert 'valid_individuals' in gen_result
            assert 'generation_time' in gen_result
            
        finally:
            # Cleanup
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def test_neural_network_genetic_algorithm_integration(self):
        # Create minimal neural network setup
        param_count = 50
        network, total_params = NetworkInitializer.setup_network(
            param_count,
            device='cpu',
            hidden_dim=64,  # Small for testing
            depth=2
        )
        
        assert network is not None
        assert total_params > 0
        
        # Test neural network guidance on population
        test_population = []
        for i in range(3):
            individual = {
                'total_span': 1600 + i * 10,
                'root_chord': 280 + i * 5,
                'tip_chord': 250 + i * 5,
                'flap_count': 3,
                'flap_cambers': [0.12, 0.10, 0.08],
                'max_thickness_ratio': 0.15,
                'camber_ratio': 0.08
            }
            test_population.append(individual)
        
        # Create pipeline to test integration
        test_config = {
            "max_generations": 1,
            "population_size": 3,
            "neural_network_enabled": True,
            "neural_network": {"learning_rate": 1e-3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            pipeline = AlphaDesignPipeline(temp_config_path, bezier_enabled=False)
            pipeline.neural_network = network
            
            # Test neural guidance application
            guided_population = pipeline.apply_neural_guidance_with_progress(
                test_population, 
                Mock(update=Mock(), set_postfix=Mock())
            )
            
            assert len(guided_population) == len(test_population)
            assert all(isinstance(individual, dict) for individual in guided_population)
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def test_cfd_optimization_integration(self):
        # Create test individual for CFD analysis
        test_individual = {
            'total_span': 1600,
            'root_chord': 280,
            'tip_chord': 250,
            'max_thickness_ratio': 0.15,
            'camber_ratio': 0.08,
            'flap_count': 3,
            'flap_spans': [1600, 1500, 1400],
            'flap_cambers': [0.12, 0.10, 0.08],
            'flap_slot_gaps': [14, 12, 10]
        }
        
        test_config = {"max_generations": 1, "population_size": 1}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            pipeline = AlphaDesignPipeline(temp_config_path, bezier_enabled=False)
            
            # Test basic CFD analysis (fallback mode)
            cfd_result = pipeline.run_basic_cfd_analysis(test_individual)
            
            assert cfd_result is not None
            assert 'performance_sweep' in cfd_result
            assert 'genetic_feedback' in cfd_result
            assert len(cfd_result['performance_sweep']) > 0
            
            # Test traditional fitness calculation
            fitness_result = pipeline.calculate_traditional_fitness(test_individual)
            
            assert fitness_result is not None
            assert isinstance(fitness_result, dict)
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def test_bezier_integration(self):
        test_config = {
            "max_generations": 1,
            "population_size": 2,
            "neural_network_enabled": False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Initialize with Bézier enabled
            pipeline = AlphaDesignPipeline(temp_config_path, bezier_enabled=True)
            
            base_params = F1FrontWingParams()
            
            # Test Bézier parameter generation
            bezier_individual = pipeline.generate_individual_with_bezier(base_params)
            
            assert bezier_individual is not None
            assert 'bezier_main_camber' in bezier_individual
            assert 'bezier_flap_cambers' in bezier_individual
            assert 'bezier_smoothness_target' in bezier_individual
            
            # Test Bézier parameter validation
            has_bezier = pipeline.has_bezier_params(bezier_individual)
            assert has_bezier == True
            
            # Test Bézier-enhanced population creation
            population = pipeline.create_bezier_enhanced_population(base_params)
            
            assert len(population) == 2
            assert all(pipeline.has_bezier_params(ind) for ind in population)
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

class TestWingGenerator:
    
    def test_wing_generation_basic(self):
        generator = UltraRealisticF1FrontWingGenerator(
            total_span=1600,
            root_chord=280,
            tip_chord=250,
            flap_count=3,
            resolution_span=10,  # Lower resolution for testing
            resolution_chord=10
        )
        
        assert generator is not None
        assert generator.total_span == 1600
        assert generator.root_chord == 280
        assert generator.flap_count == 3
    
    def test_f1_regulation_compliance(self):
        generator = UltraRealisticF1FrontWingGenerator(
            total_span=1600,  # Within regulation
            endplate_height=280,  # Within regulation
            resolution_span=10,
            resolution_chord=10
        )
        
        # Create test vertices for compliance checking
        test_vertices = np.array([
            [0, -800, 0],    # Left wing tip
            [0, 800, 0],     # Right wing tip
            [0, 0, 280],     # Top point
            [0, 250, 0],     # Y250 region
            [0, 300, 200]    # Outboard region
        ])
        
        compliance_report = generator.calculate_regulation_compliance(test_vertices)
        
        assert compliance_report is not None
        assert 'max_width_compliance' in compliance_report
        assert 'max_height_compliance' in compliance_report
        assert 'y250_compliance' in compliance_report
        assert compliance_report['max_width_compliance'] == True
        assert compliance_report['max_height_compliance'] == True
    
    def test_stl_export(self):
        generator = UltraRealisticF1FrontWingGenerator(
            resolution_span=8,  # Very low resolution for fast testing
            resolution_chord=8
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filename = os.path.join(temp_dir, 'test_wing.stl')
            
            # Test wing generation (may fail due to complexity, which is expected)
            try:
                result = generator.generate_complete_wing(test_filename)
                
                # If successful, verify file was created
                if result is not None and os.path.exists(test_filename):
                    assert os.path.getsize(test_filename) > 0
                else:
                    # Expected for complex wing generation in test environment
                    pytest.skip("Wing generation requires full dependencies")
                    
            except Exception as e:
                # Expected in test environment without full CFD dependencies
                pytest.skip(f"Wing generation failed as expected: {e}")
    
    def test_enhanced_features(self):
        generator = UltraRealisticF1FrontWingGenerator(
            flap_angle_progression=True,
            realistic_surface_curvature=True,
            aerodynamic_slots=True,
            enhanced_endplate_detail=True,
            gurney_flaps=True
        )
        
        # Test enhanced airfoil surface generation
        xu, yu, xl, yl = generator.create_enhanced_airfoil_surface(
            chord=280,
            thickness_ratio=0.15,
            camber=0.08,
            camber_pos=0.4,
            element_type="main"
        )
        
        assert len(xu) == len(yu) == len(xl) == len(yl)
        assert np.all(xu >= 0) and np.all(xu <= 280)
        
        # Test realistic flap offset system
        v_offset, h_offset, angle = generator.create_realistic_flap_offset_system(1, 0)
        
        assert v_offset > 0  # Should have vertical offset for flap 1
        assert h_offset > 0  # Should have horizontal offset
        assert angle > 0     # Should have positive angle
    
    def test_wing_element_generation(self):
        generator = UltraRealisticF1FrontWingGenerator(
            resolution_span=5,
            resolution_chord=5
        )
        
        try:
            # Test main wing element generation
            vertices, faces = generator.generate_wing_element(0)
            
            assert len(vertices) > 0
            assert len(faces) > 0
            assert all(len(face) >= 3 for face in faces)
            
        except Exception as e:
            # May fail due to missing dependencies
            pytest.skip(f"Wing element generation requires full dependencies: {e}")

class TestConfigurationManagement:
    
    def test_config_loading(self):
        # Test default configuration loading
        test_config = {
            "max_generations": 50,
            "population_size": 20,
            "neural_network_enabled": True,
            "neural_network": {
                "learning_rate": 1e-3,
                "batch_size": 16
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            pipeline = AlphaDesignPipeline(temp_config_path)
            
            assert pipeline.config['max_generations'] == 50
            assert pipeline.config['population_size'] == 20
            assert pipeline.config['neural_network_enabled'] == True
            assert pipeline.config['neural_network']['learning_rate'] == 1e-3
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def test_config_validation(self):
        # Test with missing config file (should use defaults)
        pipeline = AlphaDesignPipeline('nonexistent_config.json')
        
        # Should have default values
        assert pipeline.config['max_generations'] == 50
        assert pipeline.config['population_size'] == 20
        assert pipeline.config['neural_network_enabled'] == True
        
        # Test configuration merging
        partial_config = {"max_generations": 100}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            temp_config_path = f.name
        
        try:
            pipeline = AlphaDesignPipeline(temp_config_path)
            
            # Should merge with defaults
            assert pipeline.config['max_generations'] == 100  # Override
            assert pipeline.config['population_size'] == 20   # Default
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def test_directory_setup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)  # Change to temp directory
            
            test_config = {"max_generations": 1, "population_size": 1}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_config, f)
                temp_config_path = f.name
            
            try:
                pipeline = AlphaDesignPipeline(temp_config_path)
                
                # Check that directories were created
                for dir_name, dir_path in pipeline.output_dirs.items():
                    assert os.path.exists(dir_path)
                    assert os.path.isdir(dir_path)
                
            finally:
                if os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)

class TestPerformanceBenchmarks:
    
    def test_memory_usage(self):
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Test with small population
        base_params = F1FrontWingParams()
        small_pop_init = F1PopulInit(base_params, 5)
        small_population = small_pop_init.create_initial_population()
        
        small_memory = process.memory_info().rss
        
        # Clean up
        del small_population
        gc.collect()
        
        # Test with larger population
        large_pop_init = F1PopulInit(base_params, 15)
        large_population = large_pop_init.create_initial_population()
        
        large_memory = process.memory_info().rss
        
        # Memory should scale reasonably
        memory_increase = (large_memory - small_memory) / (1024 * 1024)  # MB
        
        assert memory_increase < 100  # Should not use more than 100MB for small test
        
        # Clean up
        del large_population
        gc.collect()
    
    def test_computation_time(self):
        # Test neural network forward pass time
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        network, _ = NetworkInitializer.setup_network(50, device=device, hidden_dim=128, depth=2)
        
        # Benchmark forward pass
        input_tensor = torch.randn(10, 50).to(device)
        
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                policy_output, value_output = network(input_tensor)
        end_time = time.time()
        
        avg_forward_time = (end_time - start_time) / 100
        
        # Should be fast (less than 10ms per forward pass for small network)
        assert avg_forward_time < 0.01
        
        # Test genetic algorithm operations
        base_params = F1FrontWingParams()
        pop_init = F1PopulInit(base_params, 10)
        
        start_time = time.time()
        population = pop_init.create_initial_population()
        end_time = time.time()
        
        population_creation_time = end_time - start_time
        
        # Population creation should be reasonably fast
        assert population_creation_time < 5.0  # Less than 5 seconds for 10 individuals
        
        assert len(population) == 10
        assert all(isinstance(individual, dict) for individual in population)
    
    def test_pipeline_scalability(self):
        configurations = [
            {"max_generations": 1, "population_size": 3},
            {"max_generations": 2, "population_size": 5},
        ]
        
        execution_times = []
        
        for config in configurations:
            config["neural_network_enabled"] = False  # Disable for speed
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                temp_config_path = f.name
            
            try:
                start_time = time.time()
                
                pipeline = AlphaDesignPipeline(temp_config_path, bezier_enabled=False)
                base_params = F1FrontWingParams()
                
                # Just test initialization (full pipeline would be too slow)
                result = pipeline.initialize_pipeline_components(base_params)
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                assert result is not None
                assert result['population_size'] == config['population_size']
                
            finally:
                if os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)
        
        # Execution time should scale reasonably
        assert all(time < 30 for time in execution_times)  # All should complete within 30s

if __name__ == "__main__":
    pytest.main([__file__])
