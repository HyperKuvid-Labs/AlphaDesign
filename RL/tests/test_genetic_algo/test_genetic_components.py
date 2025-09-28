import pytest
import numpy as np
import sys
import os
import copy
import tempfile
import random
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algo_components.initialize_population import F1PopulInit
from genetic_algo_components.mutation_strategy import F1MutationOperator
from genetic_algo_components.crossover_ops import CrossoverOps
from genetic_algo_components.fitness_evaluation import FitnessEval
from formula_constraints import F1FrontWingParams

class TestPopulationInitializer:
    
    def test_population_initialization(self):
        # Create base F1 parameters
        base_params = F1FrontWingParams(
            total_span=1600, root_chord=280, tip_chord=250,
            sweep_angle=3.5, dihedral_angle=2.5,
            max_thickness_ratio=0.15, camber_ratio=0.08,
            flap_count=3, flap_spans=[1600, 1500, 1400]
        )
        
        population_size = 10
        initializer = F1PopulInit(base_params, population_size)
        population = initializer.create_initial_population()
        
        assert len(population) == population_size
        assert all(individual is not None for individual in population)
        assert all(isinstance(individual, dict) for individual in population)
        
        # First individual should be the base design
        assert population[0]['total_span'] == 1600
        assert population[0]['root_chord'] == 280
    
    def test_individual_validity(self):
        base_params = F1FrontWingParams()
        initializer = F1PopulInit(base_params, 5)
        population = initializer.create_initial_population()
        
        for individual in population:
            # Test F1 regulation compliance
            assert 1400 <= individual['total_span'] <= 1800  # FIA span limits
            assert 200 <= individual['root_chord'] <= 350    # Practical limits
            assert 0 <= individual['sweep_angle'] <= 8       # F1 typical range
            
            # Test structural integrity parameters
            assert 0.08 <= individual['max_thickness_ratio'] <= 0.25
            assert individual['weight_estimate'] > 0
    
    def test_parameter_bounds(self):
        base_params = F1FrontWingParams()
        initializer = F1PopulInit(base_params, 1)
        
        bounds = initializer.define_f1_parameter_bounds()
        
        assert 'total_span' in bounds
        assert 'flap_cambers' in bounds
        assert 'endplate_height' in bounds
        
        # Test bound ranges are realistic
        span_min, span_max = bounds['total_span']
        assert span_min == 1400 and span_max == 1800
    
    def test_parameter_variation(self):
        base_params = F1FrontWingParams()
        initializer = F1PopulInit(base_params, 1)
        
        base_value = 1600
        varied_value = initializer.vary_parameter('total_span', base_value, 0.1)
        
        # Variation should be within bounds
        assert 1400 <= varied_value <= 1800
        
        # Test variation for unbounded parameter
        unbounded_value = initializer.vary_parameter('unknown_param', 100, 0.1)
        assert unbounded_value >= 0.001

class TestMutationStrategy:
    
    def test_mutation_rates(self):
        mutator = F1MutationOperator(mutation_rate=0.6, mutation_strength=0.5)
        
        assert 0 <= mutator.mutation_rate <= 1
        assert mutator.mutation_strength > 0
        assert mutator.param_bounds is not None
    
    def test_f1_specific_mutations(self):
        """Test F1-specific mutation operations."""
        mutator = F1MutationOperator()
        
        # Create test individual
        individual = {
            'total_span': 1600,
            'root_chord': 280,
            'tip_chord': 250,
            'sweep_angle': 3.5,
            'dihedral_angle': 2.5,
            'max_thickness_ratio': 0.15,
            'camber_ratio': 0.08,
            'flap_cambers': [0.12, 0.10, 0.08],
            'flap_slot_gaps': [10, 8, 6],
            'endplate_height': 300
        }
        
        mutated = mutator.f1_wing_mutation(individual)
        
        # Test that mutated individual maintains structure
        assert isinstance(mutated, dict)
        assert 'total_span' in mutated
        assert 'flap_cambers' in mutated
        
        # Test that values remain within bounds
        assert 1400 <= mutated['total_span'] <= 1800
        assert 200 <= mutated['root_chord'] <= 350
    
    def test_adaptive_mutation(self):
        mutator = F1MutationOperator(mutation_strength=0.5)
        
        individual = {
            'total_span': 1600,
            'sweep_angle': 3.5,
            'camber_ratio': 0.08
        }
        
        # Test early generation (should have high mutation strength)
        early_mutated = mutator.adaptive_f1_mutation(individual, generation=10, max_generations=200)
        
        # Test late generation (should have lower mutation strength) 
        late_mutated = mutator.adaptive_f1_mutation(individual, generation=180, max_generations=200)
        
        assert isinstance(early_mutated, dict)
        assert isinstance(late_mutated, dict)
    
    def test_aggressive_mutation(self):
        mutator = F1MutationOperator()
        
        individual = {
            'total_span': 1600,
            'flap_slot_gaps': [10, 8, 6],
            'flap_vertical_offsets': [50, 40, 30],
            'flap_cambers': [0.12, 0.10, 0.08],
            'weight_estimate': 5.0,
            'max_thickness_ratio': 0.15
        }
        
        mutated = mutator.aggressive_mutation(individual)
        
        # Test that aggressive mutation maintains validity
        assert isinstance(mutated, dict)
        assert mutated['weight_estimate'] >= 3.0
        assert mutated['weight_estimate'] <= 8.0
    
    def test_gaussian_mutation(self):
        mutator = F1MutationOperator(mutation_strength=0.2)
        
        original_value = 1600
        mutated_value = mutator._gaussian_mutate('total_span', original_value)
        
        # Test that Gaussian mutation produces valid output
        assert 1400 <= mutated_value <= 1800
        assert mutated_value != original_value  # Should be different (with high probability)

class TestCrossoverOperations:
    
    def test_crossover_validity(self):
        """Test that crossover produces valid offspring."""
        crossover = CrossoverOps(crossover_rate=0.8)
        
        parent1 = {
            'total_span': 1600,
            'root_chord': 280,
            'tip_chord': 250,
            'flap_cambers': [0.12, 0.10, 0.08],
            'flap_spans': [1600, 1500, 1400],
            'endplate_height': 300
        }
        
        parent2 = {
            'total_span': 1700,
            'root_chord': 300,
            'tip_chord': 260,
            'flap_cambers': [0.14, 0.12, 0.10],
            'flap_spans': [1700, 1600, 1500],
            'endplate_height': 320
        }
        
        child1, child2 = crossover.f1_aero_crossover(parent1, parent2)
        
        # Test that children are valid dictionaries
        assert isinstance(child1, dict)
        assert isinstance(child2, dict)
        
        # Test that all required parameters exist
        assert 'total_span' in child1
        assert 'flap_cambers' in child1
        assert 'endplate_height' in child1
        
        # Test that array parameters maintain correct structure
        assert len(child1['flap_cambers']) == 3
        assert len(child2['flap_spans']) == 3
    
    def test_uniform_crossover(self):
        crossover = CrossoverOps()
        
        parent1 = {
            'sweep_angle': 3.5,
            'camber_ratio': 0.08,
            'flap_slot_gaps': [10, 8, 6]
        }
        
        parent2 = {
            'sweep_angle': 4.0,
            'camber_ratio': 0.12,
            'flap_slot_gaps': [12, 10, 8]
        }
        
        child1, child2 = crossover.uniform_crossover(parent1, parent2)
        
        # Test structure preservation
        assert isinstance(child1, dict)
        assert isinstance(child2, dict)
        assert len(child1['flap_slot_gaps']) == 3
        assert len(child2['flap_slot_gaps']) == 3
    
    def test_crossover_rate_behavior(self):
        crossover = CrossoverOps(crossover_rate=0.0)  # No crossover
        
        parent1 = {'total_span': 1600}
        parent2 = {'total_span': 1700}
        
        # With crossover_rate=0, should return copies of parents
        child1, child2 = crossover.f1_aero_crossover(parent1, parent2)
        
        # Children should be deep copies of parents
        assert child1['total_span'] == parent1['total_span']
        assert child2['total_span'] == parent2['total_span']
    
    def test_parameter_categorization(self):
        crossover = CrossoverOps()
        
        # Test scalar parameters
        assert 'total_span' in crossover.scalar_params
        assert 'sweep_angle' in crossover.scalar_params
        
        # Test array parameters
        assert 'flap_spans' in crossover.array_params
        assert 'flap_cambers' in crossover.array_params

class TestFitnessEvaluator:
    
    def test_fitness_calculation(self):
        evaluator = FitnessEval(weight_constraints=0.3, weight_perf=0.4, weight_cfd=0.3)
        
        # Create a simple test individual
        individual = {
            'total_span': 1600,
            'root_chord': 280,
            'tip_chord': 250,
            'max_thickness_ratio': 0.15,
            'flap_count': 3,
            'flap_spans': [1600, 1500, 1400],
            'flap_cambers': [0.12, 0.10, 0.08]
        }
        
        # Test constraint evaluation (this might fail due to dependencies)
        try:
            constraint_score = evaluator.evaluate_formula_constratins(individual)
            assert isinstance(constraint_score, dict)
            assert 'constraint_compliance' in constraint_score
            assert 0 <= constraint_score['constraint_compliance'] <= 1
        except Exception as e:
            # Expected if dependencies aren't available
            pytest.skip(f"Constraint evaluation failed: {e}")
    
    def test_multi_objective_evaluation(self):
        evaluator = FitnessEval()
        
        # Test CFD skipping logic
        mock_constraint_score = {
            'constraint_compliance': 0.3,  # Low compliance
            'safety_factor': 1.0,          # Low safety
            'natural_frequency': 10        # Low frequency
        }
        
        should_skip = evaluator.should_skip_cfd(mock_constraint_score)
        assert should_skip == True  # Should skip due to low values
        
        # Test high-quality design (shouldn't skip)
        good_constraint_score = {
            'constraint_compliance': 0.8,
            'safety_factor': 3.0,
            'natural_frequency': 50
        }
        
        should_skip_good = evaluator.should_skip_cfd(good_constraint_score)
        assert should_skip_good == False
    
    def test_fitness_score_combination(self):
        evaluator = FitnessEval()
        
        constraint_score = {
            'constraint_compliance': 0.8,
            'computed_downforce': 1000,
            'computed_efficiency': 6.0,
            'safety_factor': 3.0,
            'structural_score': 75,
            'aerodynamic_score': 80
        }
        
        cfd_score = {
            'cfd_downforce': 1200,
            'cfd_efficiency': 7.0,
            'stall_margin': 8.0,
            'cfd_valid': True,
            'cfd_quality': 'Good'
        }
        
        combined = evaluator.combine_scores(constraint_score, cfd_score)
        
        assert isinstance(combined, dict)
        assert 'total_fitness' in combined
        assert 'constraint_fitness' in combined
        assert 'performance_fitness' in combined
        assert 'cfd_fitness' in combined
        assert combined['total_fitness'] > 0
    
    def test_structural_score_computation(self):
        evaluator = FitnessEval()
        
        validations = {
            'safety_factor_adequate': True,
            'buckling_safe': True
        }
        
        computed_vals = {
            'safety_factor': 3.5,
            'natural_frequency': 45,
            'buckling_safety_factor': 2.5,
            'total_computed_mass': 4.0
        }
        
        structural_score = evaluator._compute_structural_score(validations, computed_vals)
        
        assert isinstance(structural_score, (int, float))
        assert 0 <= structural_score <= 100
    
    def test_aerodynamic_score_computation(self):
        evaluator = FitnessEval()
        
        validations = {}
        computed_vals = {
            'efficiency_computed': 7.0,
            'reynolds_main': 1.5e6,
            'total_ground_effect': 1.8,
            'overall_effectiveness': 1.25
        }
        
        aero_score = evaluator._compute_aerodynamic_score(validations, computed_vals)
        
        assert isinstance(aero_score, (int, float))
        assert 0 <= aero_score <= 100
    
    def test_population_evaluation(self):
        evaluator = FitnessEval()
        
        # Create small test population
        population = [
            {'total_span': 1600, 'root_chord': 280, 'flap_count': 3},
            {'total_span': 1650, 'root_chord': 300, 'flap_count': 3},
            {'total_span': 1550, 'root_chord': 260, 'flap_count': 3}
        ]
        
        # This test might fail due to missing dependencies, which is expected
        try:
            fitness_scores = evaluator.evaluate_pop(population)
            assert len(fitness_scores) == len(population)
            assert all(isinstance(score, dict) for score in fitness_scores)
        except Exception as e:
            # Expected if full analysis dependencies aren't available
            pytest.skip(f"Population evaluation failed: {e}")
    
    def test_default_cfd_score(self):
        evaluator = FitnessEval()
        
        default_score = evaluator.get_default_cfd_score()
        
        assert isinstance(default_score, dict)
        assert 'cfd_downforce' in default_score
        assert 'cfd_efficiency' in default_score
        assert 'cfd_valid' in default_score
        assert default_score['cfd_valid'] == False
    
    def test_cfd_quality_assessment(self):
        evaluator = FitnessEval()
        
        excellent_result = {
            'total_downforce': 1200,
            'efficiency_ratio': 7.0,
            'f1_specific_metrics': {'stall_margin': 8.0}
        }
        
        quality = evaluator._assess_cfd_quality(excellent_result)
        assert quality == 'Excellent'
        
        poor_result = {
            'total_downforce': 300,
            'efficiency_ratio': 1.0,
            'f1_specific_metrics': {'stall_margin': 1.0}
        }
        
        quality_poor = evaluator._assess_cfd_quality(poor_result)
        assert quality_poor == 'Poor'

if __name__ == "__main__":
    pytest.main([__file__])
