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
        """Skip CFD analysis for designs that clearly don't meet basic constraints"""
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

            return {
                'constraint_compliance': results['overall_compliance'],
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

    def evaluate_cfd_perf(self, individual, id):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                wing_generator = UltraRealisticF1FrontWingGenerator(**individual)
                stl_filename = f"individual_{id}_wing.stl"
                stl_path = os.path.join(self.temp_dir, stl_filename)

                wing_mesh = wing_generator.generate_complete_wing(stl_filename)
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("CFD analysis timeout")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300) 
                
                try:
                    import shutil
                    generated_path = os.path.join("f1_wing_output", stl_filename)
                    os.makedirs(os.path.dirname(generated_path), exist_ok=True)
                    shutil.move(stl_path, generated_path)

                    analyzer = STLWingAnalyzer(wing_mesh)
                    results = analyzer.run_comprehensive_analysis()
                    
                    signal.alarm(0) 
                    
                    optimal_settings = results['optimal_settings']
                    speed_sweep = results['speed_sweep']

                    target_speed_data = next((d for d in speed_sweep if d['speed_kmh'] == 200), speed_sweep[3])

                    return {
                        'downforce': target_speed_data['downforce_N'],
                        'drag': target_speed_data['drag_N'],
                        'efficiency': target_speed_data['efficiency'],
                        'max_efficiency_speed': optimal_settings['max_efficiency_speed'],
                        'optimal_ground_clearance': optimal_settings['optimal_ground_clearance'],
                        'cfd_valid': True
                    }
                    
                except TimeoutError:
                    signal.alarm(0)
                    retry_count += 1
                    print(f"CFD timeout for individual {id}, retry {retry_count}/{max_retries}")
                    continue
                    
            except Exception as e:
                retry_count += 1
                print(f"Error in CFD evaluation (attempt {retry_count}): {e}")
                if retry_count >= max_retries:
                    return {
                        'downforce': 0, 'drag': 0, 'efficiency': 0,
                        'max_efficiency_speed': 0, 'optimal_ground_clearance': 0,
                        'cfd_valid': False, 'error': str(e)
                    }
        
        return {'cfd_valid': False, 'error': 'Max retries exceeded'}
        
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




