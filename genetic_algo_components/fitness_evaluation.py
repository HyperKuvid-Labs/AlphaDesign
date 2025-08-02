import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from formula_constraints import F1FrontWingAnalyzer, F1FrontWingParams
from cfd_analysis import STLWingAnalyzer
from wing_generator import UltraRealisticF1FrontWingGenerator
import os
import tempfile
import signal

class FitnessEval:
    def __init__(self, weight_constraints=0.3, weight_perf=0.4, weight_cfd=0.3):
        self.weight_constraints = weight_constraints
        self.weight_perf = weight_perf
        self.weight_cfd = weight_cfd
        self.temp_dir = tempfile.mkdtemp()

    def should_skip_cfd(self, constraint_score):
        return constraint_score['constraint_compliance'] < 0.4 #skipping the obvious failures

    def evaluate_pop_with_progress(self, population, pbar: tqdm):
        fitness_scores = []
        cfd_skipped_count = 0

        for i, individual in enumerate(population):
            pbar.set_description(f"📊 Evaluating Individual {i + 1}")
            
            try:
                constraint_score = self.evaluate_formula_constratins(individual)

                if self.should_skip_cfd(constraint_score):
                    cfd_skipped_count += 1
                    pbar.set_postfix({'Status': 'Constraints Failed - CFD Skipped'})
                    
                    #this sets the penalty scores
                    fitness = {
                        'total_fitness': constraint_score['constraint_compliance'] * 30,  
                        'constraint_fitness': constraint_score['constraint_compliance'] * 100,
                        'performance_fitness': 0,
                        'cfd_fitness': 0,
                        'constraint_compliance': constraint_score['constraint_compliance'],
                        'computed_downforce': constraint_score.get('computed_downforce', 0),
                        'cfd_downforce': 0,
                        'cfd_efficiency': 0,
                        'valid': False,
                        'cfd_skipped': True
                    }
                elif constraint_score['constraint_compliance'] > 0.6:
                    pbar.set_postfix({'Status': 'CFD Analysis'})
                    cfd_score = self.evaluate_cfd_perf(individual, i)
                    fitness = self.combine_scores(constraint_score, cfd_score)
                    fitness['cfd_skipped'] = False
                else:
                    pbar.set_postfix({'Status': 'Constraints Marginal'})
                    fitness = self.combine_scores(constraint_score, {'cfd_valid': False})
                    fitness['cfd_skipped'] = True

                fitness_scores.append(fitness)
                
                if isinstance(fitness, dict):
                    pbar.set_postfix({
                        'Status': 'Complete', 
                        'Fitness': f"{fitness.get('total_fitness', 0):.1f}",
                        'Skipped': cfd_skipped_count
                    })
                else:
                    pbar.set_postfix({'Status': 'Complete', 'Fitness': f"{fitness:.1f}"})
                
            except Exception as e:
                pbar.set_postfix({'Status': 'Error'})
                fitness_scores.append({
                    'total_fitness': -1000,
                    'constraint_fitness': 0,
                    'performance_fitness': 0,
                    'cfd_fitness': 0,
                    'constraint_compliance': 0,
                    'valid': False,
                    'cfd_skipped': True,
                    'error': str(e)
                })
            
            pbar.update(1)

        total_pop = len(population)
        cfd_run_count = total_pop - cfd_skipped_count
        print(f"📈 CFD Efficiency: {cfd_run_count}/{total_pop} analyses run ({cfd_skipped_count} skipped)")

        return fitness_scores

    def evaluate_pop(self, population):
        fitness_scores = []

        for i, individual in enumerate(population):
            print(f"Evaluating individual {i + 1} of {len(population)}...")

            try:
                constraint_score = self.evaluate_formula_constratins(individual)

                if self.should_skip_cfd(constraint_score):
                    print(f"Individual {i + 1} constraints too low - skipping CFD")
                    fitness = {
                        'total_fitness': constraint_score['constraint_compliance'] * 30,
                        'constraint_fitness': constraint_score['constraint_compliance'] * 100,
                        'performance_fitness': 0,
                        'cfd_fitness': 0,
                        'valid': False,
                        'cfd_skipped': True
                    }
                elif constraint_score['constraint_compliance'] > 0.6:
                    cfd_score = self.evaluate_cfd_perf(individual, i)
                    fitness = self.combine_scores(constraint_score, cfd_score)
                    fitness['cfd_skipped'] = False
                else:
                    print(f"Individual {i + 1} marginal constraints - skipping CFD")
                    fitness = self.combine_scores(constraint_score, {'cfd_valid': False})
                    fitness['cfd_skipped'] = True

                fitness_scores.append(fitness)
                
            except Exception as e:
                print(f"Error evaluating individual {i}: {e}")
                fitness_scores.append({
                    'total_fitness': -1000,
                    'constraint_fitness': 0,
                    'performance_fitness': 0,
                    'cfd_fitness': 0,
                    'valid': False,
                    'cfd_skipped': True,
                    'error': str(e)
                })

        return fitness_scores
    
    def evaluate_formula_constratins(self, individual):
        try:
            params = F1FrontWingParams(**individual)
            analyzer = F1FrontWingAnalyzer(params)
            results = analyzer.run_complete_analysis()

            compliance_bonus = 0
            if results.get('flap_gap_optimal', False):
                compliance_bonus += 0.1
            if results.get('flap_attachment', False):
                compliance_bonus += 0.1
            if results.get('downforce_target_met', False):
                compliance_bonus += 0.15
                
            adjusted_compliance = min(1.0, results['overall_compliance'] + compliance_bonus)

            return {
                'constraint_compliance': adjusted_compliance,
                'constraint_percentage': results['compliance_percentage'],
                'computed_downforce': results['computed_values']['total_downforce'],
                'computed_drag': results['computed_values']['total_drag'],
                'computed_efficiency': results['computed_values']['efficiency_computed'],
                'safety_factor': results['computed_values']['safety_factor'],
                'constraint_valid': True
            }
        except Exception as e:
            print(f"Error in constraint evaluation: {e}")
            return {
                'constraint_compliance': 0,
                'constraint_percentage': 0,
                'computed_downforce': 0,
                'computed_drag': 1000,
                'computed_efficiency': 0,
                'safety_factor': 0,
                'constraint_valid': False,
                'error': str(e)
            }

    def evaluate_cfd_perf(self, individual: Dict, individual_idx: int):
        try:
            from cfd_analysis import STLWingAnalyzer
            
            cfd_dir = "cfd_temp_files"
            os.makedirs(cfd_dir, exist_ok=True)
            
            stl_filename = f"individual_{individual_idx}_wing.stl"
            stl_path = os.path.join(cfd_dir, stl_filename)
            
            wing_generator = UltraRealisticF1FrontWingGenerator(**individual)
            wing_mesh = wing_generator.generate_complete_wing(stl_filename)
            
            expected_path = os.path.join("f1_wing_output", stl_filename)
            if os.path.exists(expected_path):
                import shutil
                shutil.copy2(expected_path, stl_path)
            else:
                return self.get_default_cfd_score()
            
            if os.path.exists(stl_path):
                analyzer = STLWingAnalyzer(stl_path)
                
                speed_ms = analyzer.convert_speed_to_ms(330)  # 200 km/h test speed
                cfd_result = analyzer.multi_element_analysis(speed_ms, 75, 0)
                
                return {
                    'cfd_downforce': float(cfd_result['total_downforce']),
                    'cfd_drag': float(cfd_result['total_drag']),
                    'cfd_efficiency': float(cfd_result['efficiency_ratio']),
                    'cfd_valid': True,
                    'flow_attachment': cfd_result['flow_characteristics']['flow_attachment']
                }
            else:
                return self.get_default_cfd_score()
                
        except Exception as e:
            print(f"❌ CFD evaluation failed: {str(e)}")
            return self.get_default_cfd_score()

    def get_default_cfd_score(self):
        """Return conservative CFD score when analysis fails"""
        return {
            'cfd_downforce': 1000,  # Conservative estimate
            'cfd_drag': 100,
            'cfd_efficiency': 10.0,
            'cfd_valid': False,
            'flow_attachment': 'Unknown'
        }
     
    def combine_scores(self, constraint_score, cfd_score):
        constraint_fitness = constraint_score['constraint_compliance'] * 100

        computed_downforce = constraint_score.get('computed_downforce', 0)
        computed_efficiency = constraint_score.get('computed_efficiency', 0)
        performance_fitness = min(100, (computed_downforce / 1000) * 50 + computed_efficiency * 50)

        if cfd_score.get('cfd_valid', False):
            cfd_downforce = cfd_score['downforce']
            cfd_efficiency = cfd_score['efficiency']
            cfd_fitness = min(100, (cfd_downforce / 1500) * 60 + cfd_efficiency * 40)
        else:
            cfd_fitness = 0

        total_fitness = (
            self.weight_constraints * constraint_fitness +
            self.weight_perf * performance_fitness +
            self.weight_cfd * cfd_fitness
        )

        return {
            'total_fitness': total_fitness,
            'constraint_fitness': constraint_fitness,
            'performance_fitness': performance_fitness,
            'cfd_fitness': cfd_fitness,
            'constraint_compliance': constraint_score['constraint_compliance'],
            'computed_downforce': constraint_score.get('computed_downforce', 0),
            'cfd_downforce': cfd_score.get('downforce', 0),
            'cfd_efficiency': cfd_score.get('efficiency', 0),
            'valid': constraint_score.get('constraint_valid', False) and cfd_score.get('cfd_valid', False)
        }




