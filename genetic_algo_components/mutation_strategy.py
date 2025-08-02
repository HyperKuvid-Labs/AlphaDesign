import random
import numpy as np
from typing import Dict, Any

#so for the gaussain mutation, this is the def from gfg, which is kinda nice, people are literally mad in combining diff domains always, fascinating!!

# Gaussian mutation is a technique used in genetic algorithms (GAs) to introduce small random changes in the individuals of a population. It involves adding a random value from a Gaussian (normal) distribution to each element of an individual's vector to create a new offspring.
#  This method is particularly useful for fine-tuning solutions and exploring the domain effectively.

# In Gaussian mutation, the variance of the distribution is determined by parameters such as scale and shrink. The scale controls the standard deviation of the mutation at the first generation, while the shrink controls the rate at which the average amount of mutation decreases over generations.
#  This approach helps in maintaining a balance between exploration and exploitation in the search process.


class F1MutationOperator:
    def __init__(self, mutation_rate: float = 0.15, mutation_strength: float = 0.1):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.param_bounds = self._get_f1_parameter_bounds()
    
    def f1_wing_mutation(self, individual):
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            chord_scale = 1 + np.random.normal(0, self.mutation_strength * 0.5)
            mutated['root_chord'] *= chord_scale
            mutated['tip_chord'] *= chord_scale
            
            mutated['chord_taper_ratio'] = mutated['tip_chord'] / mutated['root_chord']
            
            # Apply bounds
            mutated['root_chord'] = np.clip(mutated['root_chord'], 200, 350)
            mutated['tip_chord'] = np.clip(mutated['tip_chord'], 180, 300)
        
        # Airfoil mutations
        airfoil_params = ['max_thickness_ratio', 'camber_ratio', 'camber_position']
        for param in airfoil_params:
            if random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
        
        # Flap system mutations (maintain element relationships)
        if random.random() < self.mutation_rate:
            # Mutate entire flap system coherently
            flap_scale = 1 + np.random.normal(0, self.mutation_strength * 0.3)
            
            for i in range(len(mutated['flap_cambers'])):
                mutated['flap_cambers'][i] *= flap_scale
                mutated['flap_cambers'][i] = np.clip(mutated['flap_cambers'][i], 0.05, 0.20)
                
                # Adjust slot gaps proportionally
                mutated['flap_slot_gaps'][i] *= (1 + np.random.normal(0, 0.05))
                mutated['flap_slot_gaps'][i] = np.clip(mutated['flap_slot_gaps'][i], 5, 25)
        
        # Y250 region mutations (regulation-compliant)
        y250_params = ['y250_step_height', 'y250_transition_length', 'central_slot_width']
        for param in y250_params:
            if random.random() < self.mutation_rate * 0.7:  # Lower rate for regulatory params
                mutated[param] = self._gaussian_mutate(param, mutated[param])
        
        # Endplate mutations
        endplate_params = ['endplate_height', 'endplate_max_width', 'endplate_min_width']
        for param in endplate_params:
            if random.random() < self.mutation_rate:
                mutated[param] = self._gaussian_mutate(param, mutated[param])
        
        return mutated
    
    def adaptive_f1_mutation(self, individual: Dict, generation: int, max_generations: int):
        """Adaptive mutation that reduces strength as evolution progresses"""
        adaptive_strength = self.mutation_strength * (1 - generation / max_generations)
        
        original_strength = self.mutation_strength
        self.mutation_strength = adaptive_strength
        
        mutated = self.f1_wing_mutation(individual)
        
        self.mutation_strength = original_strength
        
        return mutated
    
    def _gaussian_mutate(self, param_name: str, current_value: float):
        noise = np.random.normal(0, self.mutation_strength)
        new_value = current_value * (1 + noise)
        
        if param_name in self.param_bounds:
            min_val, max_val = self.param_bounds[param_name]
            new_value = np.clip(new_value, min_val, max_val)
        
        return new_value
    
    def _get_f1_parameter_bounds(self):
        return {
            'total_span': (1400, 1800),
            'root_chord': (200, 350),
            'tip_chord': (180, 300),
            'sweep_angle': (0, 8),
            'dihedral_angle': (0, 6),
            'max_thickness_ratio': (0.08, 0.25),
            'camber_ratio': (0.04, 0.18),
            'camber_position': (0.25, 0.55),
            'endplate_height': (200, 330),
            'endplate_max_width': (80, 180),
            'endplate_min_width': (25, 80),
            'y250_step_height': (10, 25),
            'y250_transition_length': (60, 150),
            'central_slot_width': (20, 50),
            'flap_cambers': (0.05, 0.20),
            'flap_slot_gaps': (5, 25),
        }
