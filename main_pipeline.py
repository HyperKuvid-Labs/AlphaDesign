import gc
import os
import json
import time
import torch
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from tqdm import tqdm

from formula_constraints import F1FrontWingParams, F1FrontWingAnalyzer
from genetic_algo_components.initialize_population import F1PopulInit
from genetic_algo_components.fitness_evaluation import FitnessEval
from genetic_algo_components.crossover_ops import CrossoverOps
from genetic_algo_components.mutation_strategy import F1MutationOperator
from neural_network_components.forward_pass import NeuralNetworkForwardPass
from neural_network_components.network_initialization import NetworkInitializer
from neural_network_components.optimizer_integration import OptimizerManager
from neural_network_components.loss_calculation import AlphaDesignLoss
from neural_network_components.parameter_tweaking import ParamterTweaker
from wing_generator import UltraRealisticF1FrontWingGenerator

class AlphaDesignPipeline:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        self.current_generation = 0
        self.best_designs_history = []
        self.training_metrics = []
        self.neural_network = None
        self.optimizer_manager = None
        
        #as of now not using early stopping but will be used in future
        from early_stopping import EarlyStoppingManager  # You need to create this
        self.early_stopping = EarlyStoppingManager(
            patience=self.config['early_stopping']['patience'],
            min_delta=self.config['early_stopping']['min_delta'],
            restore_best_weights=True
        )

        #so this time took help of perplexity to write better logs and print statements
        print("🚀 AlphaDesign Pipeline Initialized")
        print(f"📊 Max Generations: {self.config['max_generations']}")
        print(f"🧬 Population Size: {self.config['population_size']}")
        print(f"🧠 Neural Network Enabled: {self.config['neural_network_enabled']}")
    
    def load_config(self, config_path: str):
        default_config = {
            "max_generations": 50,
            "population_size": 20,
            "neural_network_enabled": True,
            "save_frequency": 5,
            "max_runtime_hours": 24,
            "neural_network": {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "training_frequency": 3
            },
            "cfd_analysis": {
                "enabled": True,
                "parallel_processes": 2
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_logging(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/alphadesign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AlphaDesign")
    
    def setup_directories(self):
        self.output_dirs = {
            'checkpoints': 'checkpoints',
            'best_designs': 'best_designs',
            'neural_networks': 'neural_networks',
            'generation_data': 'generation_data',
            'stl_outputs': 'stl_outputs'
        }
        
        for dir_name, dir_path in self.output_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
    
    def run_complete_pipeline(self, base_params: F1FrontWingParams):
        
        self.logger.info("🏁 Starting AlphaDesign Complete Pipeline")
        start_time = time.time()
        
        try:
            # phase 1: Initialize components
            results = self.initialize_pipeline_components(base_params)
            
            # phase 2: Main optimization loop
            results.update(self.run_optimization_loop(base_params))
            
            # phase 3: Final analysis and cleanup
            results.update(self.finalize_pipeline())
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Pipeline completed in {total_time/3600:.2f} hours")
            
            return results
            
        except Exception as e:
            self.logger.error(f"💥 Pipeline failed: {str(e)}")
            raise
    
    def initialize_pipeline_components(self, base_params: F1FrontWingParams):
        
        self.logger.info("🔧 Initializing Pipeline Components")
        
        # 1. genetic algo components
        self.population_init = F1PopulInit(base_params, self.config['population_size'])
        self.fitness_eval = FitnessEval()
        self.crossover_ops = CrossoverOps()
        self.mutation_ops = F1MutationOperator()
        
        # 2. neural network components (if enabled)
        if self.config['neural_network_enabled']:
            self.setup_neural_network(base_params)
        
        # 3. initialize population
        self.current_population = self.population_init.create_initial_population()
        
        self.logger.info(f"✅ Components initialized. Population size: {len(self.current_population)}")
        
        return {"initialization": "success", "population_size": len(self.current_population)}
    
    def setup_neural_network(self, base_params: F1FrontWingParams):
        param_dict = asdict(base_params)
        scalar_params = sum(1 for v in param_dict.values() if isinstance(v, (int, float)))
        list_params = sum(len(v) for v in param_dict.values() if isinstance(v, list))
        param_count = scalar_params + list_params
        
        self.neural_network, total_params = NetworkInitializer.setup_network(
            param_count, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.optimizer_manager = OptimizerManager(
            self.neural_network,
            learning_rate=self.config['neural_network']['learning_rate']
        )
        
        self.loss_calculator = AlphaDesignLoss()
        self.param_tweaker = ParamterTweaker()
        
        self.logger.info(f"🧠 Neural Network initialized: {total_params} parameters")
    
    def run_optimization_loop(self, base_params: F1FrontWingParams):
        self.logger.info("🔄 Starting Optimization Loop")
        
        generation_results = []
        max_runtime = self.config['max_runtime_hours'] * 3600
        start_time = time.time()
        
        # Main progress bar for generations
        generation_pbar = tqdm(
            total=self.config['max_generations'],
            desc="🧬 Generations",
            unit="gen",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for generation in range(self.config['max_generations']):
            self.current_generation = generation
            
            if time.time() - start_time > max_runtime:
                self.logger.info(f"⏰ Runtime limit reached ({self.config['max_runtime_hours']} hours)")
                break
            
            generation_pbar.set_description(f"🧬 Gen {generation + 1}")
            
            gen_result = self.run_single_generation(base_params)
            generation_results.append(gen_result)
            
            # Update progress bar with current metrics
            generation_pbar.set_postfix({
                'Best': f"{gen_result['best_fitness']:.2f}",
                'Avg': f"{gen_result['average_fitness']:.2f}",
                'Valid': f"{gen_result['valid_individuals']}"
            })
            generation_pbar.update(1)
            
            if generation % self.config['save_frequency'] == 0:
                self.save_checkpoint(generation, gen_result)
            
            if self.early_stopping.should_stop(gen_result['best_fitness']):
                self.logger.info(f"🛑 Early stopping triggered at generation {generation}")
                break
            
            if self.config['neural_network_enabled'] and generation % self.config['neural_network']['training_frequency'] == 0:
                self.train_neural_network_extended(generation_results[-3:])
        
        generation_pbar.close()
        
        return {
            "total_generations": len(generation_results),
            "generation_results": generation_results,
        }
    
    def run_single_generation(self, base_params: F1FrontWingParams):
        generation_start = time.time()
        
        # Progress bar for population evaluation
        eval_pbar = tqdm(
            total=len(self.current_population),
            desc="📊 Evaluating Population",
            unit="individual",
            position=1,
            leave=False
        )
        
        # Modified fitness evaluation with progress
        fitness_scores = self.fitness_eval.evaluate_pop_with_progress(
            self.current_population, eval_pbar
        )
        eval_pbar.close()
        
        # 2. find the best individual, so we can save and see how the design evolves
        valid_scores = [score for score in fitness_scores if isinstance(score, dict) and score.get('valid', False)]
        
        if valid_scores:
            best_idx = max(range(len(valid_scores)), key=lambda i: valid_scores[i]['total_fitness'])
            best_individual = self.current_population[best_idx]
            best_fitness = valid_scores[best_idx]['total_fitness']
            
            # 3. save the best design STL
            self.save_best_design_stl(best_individual, self.current_generation)
            
        else:
            self.logger.warning("⚠️ No valid individuals in population")
            best_individual = self.current_population[0]
            best_fitness = -1000
        
        gen_pbar = tqdm(
            total=len(self.current_population),
            desc="🔄 Creating Next Generation",
            unit="individual",
            position=1,
            leave=False
        )
        
        new_population = self.generate_next_population_with_progress(gen_pbar)
        gen_pbar.close()
        
        # 5. neural network guidance, to be specific by the policy network and its output
        if self.config['neural_network_enabled']:
            nn_pbar = tqdm(
                total=len(new_population),
                desc="🧠 Applying Neural Guidance",
                unit="individual",
                position=1,
                leave=False
            )
            new_population = self.apply_neural_guidance_with_progress(new_population, nn_pbar)
            nn_pbar.close()
        
        self.current_population = new_population
        
        generation_time = time.time() - generation_start
        
        result = {
            "generation": self.current_generation,
            "best_fitness": best_fitness,
            "best_individual": best_individual,
            "average_fitness": sum(score.get('total_fitness', -1000) for score in fitness_scores) / len(fitness_scores),
            "valid_individuals": len(valid_scores),
            "generation_time": generation_time
        }
        
        self.logger.info(f"✅ Generation {self.current_generation}: Best={best_fitness:.2f}, Avg={result['average_fitness']:.2f}, Time={generation_time:.1f}s")
        
        # Add memory cleanup at end of generation
        import gc
        import torch
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        return result
    
    def save_best_design_stl(self, individual: Dict[str, Any], generation: int):
        try:
            params = F1FrontWingParams(**individual)
            
            wing_generator = UltraRealisticF1FrontWingGenerator(**individual)
            stl_filename = f"generation_{generation:03d}_best_design.stl"
            stl_path = os.path.join(self.output_dirs['stl_outputs'], stl_filename)
            
            wing_mesh = wing_generator.generate_complete_wing(stl_filename)
            
            import shutil
            generated_path = os.path.join("f1_wing_output", stl_filename)
            if os.path.exists(generated_path):
                shutil.move(generated_path, stl_path)
                self.logger.info(f"💾 Best design saved: {stl_path}")
            
            json_path = stl_path.replace('.stl', '_params.json')
            with open(json_path, 'w') as f:
                json.dump(individual, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save STL: {str(e)}")
    
    def generate_next_population_with_progress(self, pbar: tqdm) -> List[Dict[str, Any]]:
        new_population = []
        
        # Elite selection
        elite_count = max(1, len(self.current_population) // 5)
        fitness_scores = self.fitness_eval.evaluate_pop(self.current_population)
        
        population_with_fitness = list(zip(self.current_population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1].get('total_fitness', -1000) if isinstance(x[1], dict) else -1000, reverse=True)
        
        # Add elites
        for i in range(elite_count):
            new_population.append(population_with_fitness[i][0])
            pbar.update(1)
        
        # Generate offspring
        while len(new_population) < len(self.current_population):
            parent1 = self.tournament_selection(population_with_fitness)
            parent2 = self.tournament_selection(population_with_fitness)
            
            child1, child2 = self.crossover_ops.f1_aero_crossover(parent1, parent2)
            child1 = self.mutation_ops.f1_wing_mutation(child1)
            child2 = self.mutation_ops.f1_wing_mutation(child2)
            
            new_population.extend([child1, child2])
            pbar.update(min(2, len(self.current_population) - len(new_population) + 2))
        
        return new_population[:len(self.current_population)]
    
    def tournament_selection(self, population_with_fitness: List, tournament_size: int = 3):
        import random
        
        tournament = random.sample(population_with_fitness, min(tournament_size, len(population_with_fitness)))
        winner = max(tournament, key=lambda x: x[1].get('total_fitness', -1000) if isinstance(x[1], dict) else -1000)
        return winner[0]
    
    def apply_neural_guidance_with_progress(self, population: List[Dict[str, Any]], pbar: tqdm):
        if self.neural_network is None:
            pbar.update(len(population))
            return population
        
        try:
            guided_population = []
            
            for individual in population:
                # Convert dict to ordered parameter tensor
                param_keys = sorted(individual.keys())  # Ensure consistent ordering
                param_values = [individual[key] if isinstance(individual[key], (int, float)) 
                              else sum(individual[key]) if isinstance(individual[key], list) 
                              else 0 for key in param_keys]
                
                param_tensor = torch.tensor([param_values], dtype=torch.float32)
                
                # Get neural network predictions
                with torch.no_grad():
                    policy_output, value_output = self.neural_network(param_tensor)
                
                # Apply parameter tweaks
                modified_params = self.param_tweaker.apply_neural_tweaks(
                    param_tensor, policy_output, exploration=True
                )
                
                # Convert back to individual dict with proper mapping
                guided_individual = individual.copy()
                modified_values = modified_params.squeeze().tolist()
                
                # Map back to parameters (simplified approach)
                for i, key in enumerate(param_keys):
                    if isinstance(individual[key], (int, float)) and i < len(modified_values):
                        guided_individual[key] = modified_values[i]
                
                guided_population.append(guided_individual)
                pbar.update(1)
            
            return guided_population
            
        except Exception as e:
            self.logger.warning(f"⚠️ Neural guidance failed: {str(e)}")
            pbar.update(len(population))
            return population
        
    def cleanup_generation(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        import matplotlib.pyplot as plt
        plt.close('all')
        
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            self.logger.warning(f"⚠️ Memory usage high: {memory_percent}%")
    
    def train_neural_network_extended(self, recent_generations: List[Dict[str, Any]], generation: int):
        
        if not recent_generations or self.neural_network is None:
            return
        
        try:
            self.logger.info(f"🧠 Extended neural network training - Generation {generation}")
            
            # Adaptive epochs based on generation
            if generation < 30:
                epochs = 40  # More training early on
            elif generation < 80:
                epochs = 25  # Standard training
            else:
                epochs = 15  # Fine-tuning later
            
            # Curriculum learning - focus on different aspects over time
            if generation < 40:
                # Early: Focus on constraint compliance
                weight_constraints = 0.6
                weight_performance = 0.4
                training_phase = "Constraint Focus"
            elif generation < 80:
                # Middle: Balance constraints and performance
                weight_constraints = 0.4
                weight_performance = 0.6
                training_phase = "Balanced Training"
            else:
                # Late: Focus on optimization
                weight_constraints = 0.2
                weight_performance = 0.8
                training_phase = "Performance Focus"
            
            self.logger.info(f"📚 Curriculum Phase: {training_phase} (Epochs: {epochs})")
            
            # Progress bar for extended training
            train_pbar = tqdm(
                total=epochs,
                desc=f"🧠 Extended Training ({training_phase})",
                unit="epoch",
                position=1,
                leave=False
            )
            
            # Training loop with curriculum weights
            optimizer = self.optimizer_manager.get_optimizer()
            
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                
                for gen_data in recent_generations:
                    # Convert to tensor
                    param_tensor = self.param_tweaker.genetic_to_neural_params(
                        [list(gen_data['best_individual'].values())]
                    )
                    
                    # Forward pass
                    policy_output, value_output = self.neural_network(param_tensor)
                    
                    # Curriculum-weighted loss
                    constraint_loss = self.calculate_constraint_loss(gen_data, value_output)
                    performance_loss = self.calculate_performance_loss(gen_data, value_output)
                    
                    total_loss = (weight_constraints * constraint_loss + 
                                 weight_performance * performance_loss)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    batch_count += 1
                
                avg_epoch_loss = epoch_loss / max(batch_count, 1)
                
                train_pbar.set_postfix({
                    'Loss': f"{avg_epoch_loss:.4f}",
                    'Phase': training_phase[:8]
                })
                train_pbar.update(1)
                
                if epoch % 10 == 0:
                    self.logger.info(f"🧠 Epoch {epoch}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
                
                if avg_epoch_loss < 1e-6:
                    self.logger.info(f"🛑 Training converged early at epoch {epoch}")
                    break
            
            train_pbar.close()
            self.logger.info(f"✅ Extended neural network training completed - {epochs} epochs")
            
            self.save_training_metrics(generation, epoch_loss, training_phase)
            
            self.cleanup_generation()
            
        except Exception as e:
            self.logger.error(f"❌ Extended neural network training failed: {str(e)}")
    
    def calculate_constraint_loss(self, gen_data: Dict[str, Any], value_output: torch.Tensor):
        try:
            best_individual = gen_data['best_individual']

            params = F1FrontWingParams(**best_individual)
            analyzer = F1FrontWingAnalyzer(params)
            constraint_results = analyzer.run_complete_analysis()
            
            target_compliance = torch.tensor([constraint_results['overall_compliance']], dtype=torch.float32)
            
            constraint_loss = torch.nn.functional.mse_loss(value_output, target_compliance)
            
            return constraint_loss
            
        except Exception as e:
            self.logger.warning(f"⚠️ Constraint loss calculation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    def calculate_performance_loss(self, gen_data: Dict[str, Any], value_output: torch.Tensor):
        try:
            fitness_score = gen_data['best_fitness']
            
            normalized_fitness = min(1.0, max(0.0, fitness_score / 100.0))
            target_performance = torch.tensor([normalized_fitness], dtype=torch.float32)
            
            performance_loss = torch.nn.functional.mse_loss(value_output, target_performance)
            
            return performance_loss
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance loss calculation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    def save_training_metrics(self, generation: int, final_loss: float, training_phase: str):
        try:
            metrics = {
                'generation': generation,
                'final_loss': final_loss,
                'training_phase': training_phase,
                'timestamp': datetime.now().isoformat()
            }
            
            metrics_path = os.path.join(self.output_dirs['neural_networks'], f'training_metrics_gen_{generation}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save training metrics: {e}")
    
    def train_neural_network(self, recent_generations: List[Dict[str, Any]]):
        if not recent_generations or self.neural_network is None:
            return
        
        try:
            self.logger.info("🧠 Training neural network...")
            
            training_data = []
            for gen_result in recent_generations:
                training_data.append({
                    'individual': gen_result['best_individual'],
                    'fitness': gen_result['best_fitness']
                })
            
            optimizer = self.optimizer_manager.get_optimizer()
            
            train_pbar = tqdm(
                total=10 * len(training_data),
                desc="🧠 Neural Network Training",
                unit="batch",
                position=1,
                leave=False
            )
            
            for epoch in range(10):
                for data in training_data:
                    param_tensor = self.param_tweaker.genetic_to_neural_params([list(data['individual'].values())])
                    
                    #forward pass
                    policy_output, value_output = self.neural_network(param_tensor)
                    
                    #calculare the loss
                    target_value = torch.tensor([data['fitness']], dtype=torch.float32)
                    loss = self.loss_calculator.cfd_reward_loss(value_output, target_value)
                    
                    #backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                    train_pbar.update(1)
            
            train_pbar.close()
            self.logger.info("✅ Neural network training completed")

            self.cleanup_generation()
            
        except Exception as e:
            self.logger.error(f"❌ Neural network training failed: {str(e)}")
    
    def run_optimization_loop(self, base_params: F1FrontWingParams):
        self.logger.info("🔄 Starting Optimization Loop")
        
        generation_results = []
        max_runtime = self.config['max_runtime_hours'] * 3600
        start_time = time.time()
        
        # Main progress bar for generations
        generation_pbar = tqdm(
            total=self.config['max_generations'],
            desc="🧬 Generations",
            unit="gen",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for generation in range(self.config['max_generations']):
            self.current_generation = generation
            
            if time.time() - start_time > max_runtime:
                self.logger.info(f"⏰ Runtime limit reached ({self.config['max_runtime_hours']} hours)")
                break
            
            generation_pbar.set_description(f"🧬 Gen {generation + 1}")
            
            gen_result = self.run_single_generation(base_params)
            generation_results.append(gen_result)
            
            generation_pbar.set_postfix({
                'Best': f"{gen_result['best_fitness']:.2f}",
                'Avg': f"{gen_result['average_fitness']:.2f}",
                'Valid': f"{gen_result['valid_individuals']}"
            })
            generation_pbar.update(1)
            
            if generation % self.config['save_frequency'] == 0:
                self.save_checkpoint(generation, gen_result)
                self.cleanup_generation()
            
            if self.early_stopping.should_stop(gen_result['best_fitness']):
                self.logger.info(f"🛑 Early stopping triggered at generation {generation}")
                break
            
            # Use extended training instead of basic training
            if self.config['neural_network_enabled'] and generation % self.config['neural_network']['training_frequency'] == 0:
                # Use extended training with curriculum learning
                self.train_neural_network_extended(generation_results[-5:], generation)  # Use more recent generations
        
        generation_pbar.close()
        
        # Final cleanup after optimization loop
        self.cleanup_generation()
        
        return {
            "total_generations": len(generation_results),
            "generation_results": generation_results,
        }
    
    def finalize_pipeline(self) -> Dict[str, Any]:
        
        self.logger.info("🏁 Finalizing pipeline...")
        
        if self.neural_network is not None:
            final_nn_path = os.path.join(self.output_dirs['neural_networks'], 'final_network.pth')
            torch.save(self.neural_network.state_dict(), final_nn_path)
        
        summary = self.generate_summary_report()
        
        return {
            "finalization": "success",
            "summary": summary
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        
        stl_files = [f for f in os.listdir(self.output_dirs['stl_outputs']) if f.endswith('.stl')]
        
        summary = {
            "total_generations": self.current_generation,
            "total_designs_generated": len(stl_files),
            "output_directories": self.output_dirs,
            "best_designs_stl": stl_files,
            "final_population_size": len(self.current_population) if hasattr(self, 'current_population') else 0
        }
        
        summary_path = os.path.join(self.output_dirs['checkpoints'], 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Summary report saved: {summary_path}")
        
        return summary
