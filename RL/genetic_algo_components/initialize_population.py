import numpy as np
import random
from typing import Dict, List, Any
from dataclasses import asdict
from formula_constraints import F1FrontWingParams

class F1PopulInit:
    def __init__(self, base_f1_params: F1FrontWingParams, population_size: int = 50):
        self.base_params = base_f1_params
        self.population_size = population_size
        self.param_bounds = self.define_f1_parameter_bounds()
        # pass

    def create_f1_variant(self, base_design):
        variant = base_design.copy()

        # Wing structure variations
        variant['total_span'] = self.vary_parameter('total_span', variant['total_span'], 0.1)
        variant['root_chord'] = self.vary_parameter('root_chord', variant['root_chord'], 0.15)
        variant['tip_chord'] = self.vary_parameter('tip_chord', variant['tip_chord'], 0.15)
        variant['sweep_angle'] = self.vary_parameter('sweep_angle', variant['sweep_angle'], 0.25)
        variant['dihedral_angle'] = self.vary_parameter('dihedral_angle', variant['dihedral_angle'], 0.2)
        
        # Airfoil variations
        variant['max_thickness_ratio'] = self.vary_parameter('max_thickness_ratio', variant['max_thickness_ratio'], 0.2)
        variant['camber_ratio'] = self.vary_parameter('camber_ratio', variant['camber_ratio'], 0.25)
        variant['camber_position'] = self.vary_parameter('camber_position', variant['camber_position'], 0.1)
        
        # Flap system variations (your specific arrays)
        for i in range(len(variant['flap_cambers'])):
            variant['flap_cambers'][i] = self.vary_parameter('flap_cambers', variant['flap_cambers'][i], 0.2)
            variant['flap_slot_gaps'][i] = self.vary_parameter('flap_slot_gaps', variant['flap_slot_gaps'][i], 0.2)
            variant['flap_vertical_offsets'][i] = self.vary_parameter('flap_vertical_offsets', variant['flap_vertical_offsets'][i], 0.15)
            variant['flap_horizontal_offsets'][i] = self.vary_parameter('flap_horizontal_offsets', variant['flap_horizontal_offsets'][i], 0.15)
        
        # Endplate variations
        variant['endplate_height'] = self.vary_parameter('endplate_height', variant['endplate_height'], 0.15)
        variant['endplate_max_width'] = self.vary_parameter('endplate_max_width', variant['endplate_max_width'], 0.2)
        variant['endplate_min_width'] = self.vary_parameter('endplate_min_width', variant['endplate_min_width'], 0.2)
        
        # Y250 region variations
        variant['y250_step_height'] = self.vary_parameter('y250_step_height', variant['y250_step_height'], 0.25)
        variant['y250_transition_length'] = self.vary_parameter('y250_transition_length', variant['y250_transition_length'], 0.2)
        variant['central_slot_width'] = self.vary_parameter('central_slot_width', variant['central_slot_width'], 0.2)
        
        #so these values are taken directly from some cacluation using the help of perplextity and susharsan
        return variant
    
    def define_f1_parameter_bounds(self):
        return {
            # Wing structure bounds (FIA 2024 regulations compliance)
            'total_span': (1600, 1800),          # FIA regulation: max 1800mm (Article 3.3.1)
            'root_chord': (250, 330),            # FIA regulation: max 330mm at centerline (Article 3.3.2)
            'tip_chord': (200, 300),             # Structural requirements + aero efficiency
            'sweep_angle': (2, 8),               # F1 typical range for stability
            'dihedral_angle': (1, 6),            # Ground effect optimization (typically 2-4°)
            
            # Airfoil bounds (aerodynamic performance + manufacturing)
            'max_thickness_ratio': (0.04, 0.20), # Thin sections for lower drag, thicker for strength
            'camber_ratio': (0.06, 0.15),        # High camber for downforce, regulated max ~15%
            'camber_position': (0.35, 0.50),     # Forward camber position for F1 wings
            
            # Endplate bounds (FIA Article 3.4 compliance)
            'endplate_height': (250, 325),       # FIA max height: 325mm (Article 3.4.1)
            'endplate_max_width': (100, 150),    # Aerodynamic effectiveness + regulations
            'endplate_min_width': (35, 70),      # Structural requirements at top
            
            # Y250 region bounds (FIA Article 3.3.6 - Y250 vortex regulations)
            'y250_step_height': (15, 25),        # Regulation compliance + vortex strength
            'y250_transition_length': (80, 120), # Smooth aerodynamic transition
            'central_slot_width': (0, 30),       # Updated: 0 for clean geometry, max 30mm if needed
            
            # Flap system bounds (realistic F1 multi-element configuration)
            'flap_cambers': (0.08, 0.16),        # High camber for rear elements (8-16%)
            'flap_slot_gaps': (8, 18),           # Optimal slot gap: 1-2% of chord (8-18mm typical)
            'flap_vertical_offsets': (20, 125),  # Progressive stacking (20-125mm range)
            'flap_horizontal_offsets': (25, 145), # Overlap for slot effect (25-145mm)

            # Weight bounds (FIA minimum weight + realistic carbon fiber construction)
            'weight_estimate': (2.8, 5.5),       # FIA minimum ~3kg, realistic 3-5kg
        }


    def create_initial_population(self):
        population = []

        base_dict = asdict(self.base_params)
        population.append(base_dict)

        for i in range(self.population_size - 1):
            individual = self.create_f1_variant(base_dict)
            population.append(individual)
        
        return population
    

    def vary_parameter(self, param_name, base_value, variation_factor):
        #so here from param name we getthe values from the dict
        #base-value is the iniitla value we are taking
        #variation_factor how much shuld we change or populate accordingly

        if param_name in self.param_bounds:
            min_val, max_val = self.param_bounds[param_name]
            variation = base_value * variation_factor * (random.random() - 0.5) * 2
 #some noise to train more, inspired from diffusion models (ddpm)

            new_value = base_value + variation
            return np.clip(new_value, min_val, max_val)
        else:
            variation = base_value * variation_factor * (random.random() - 0.5) * 2

            return max(0.001, base_value + variation)
        

