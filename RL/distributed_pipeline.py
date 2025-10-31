import gc
import os
import json
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from tqdm import tqdm

from alphadesign import load_base_parameters
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
    def __init__(self, config_path: str = "config.json", rank: int = -1, world_size: int = 1):
        self.config = self.load_config(config_path)
        
        # distributed training parameters
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = rank >= 0
        self.is_main_process = rank <= 0  # rank -1 (non-distributed) or rank 0 (main process)
        
        self.setup_logging()
        self.setup_directories()
        
        self.current_generation = 0
        self.best_designs_history = []
        self.training_metrics = []
        self.neural_network = None
        self.optimizer_manager = None
        self.generation_results = []
        
        # #as of now not using early stopping but will be used in future
        # from early_stopping import EarlyStoppingManager  # You need to create this
        # self.early_stopping = EarlyStoppingManager(
        #     patience=self.config['early_stopping']['patience'],
        #     min_delta=self.config['early_stopping']['min_delta'],
        #     restore_best_weights=True
        # )

        #so this time took help of perplexity to write better logs and print statements
        if self.is_main_process:
            print("🚀 AlphaDesign Pipeline Initialized")
            print(f"📊 Max Generations: {self.config['max_generations']}")
            print(f"🧬 Population Size: {self.config['population_size']}")
            print(f"🧠 Neural Network Enabled: {self.config['neural_network_enabled']}")
            if self.is_distributed:
                print(f"🌐 Distributed Training: {self.world_size} processes")
    
    @staticmethod
    def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
        """
        initialize distributed process group for multi-gpu training
        
        args:
            rank: unique identifier for each process (0 to world_size-1)
            world_size: total number of processes participating in training
            backend: communication backend ('nccl' for gpu, 'gloo' for cpu)
        """
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        # initialize the process group
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        # set device for this process
        torch.cuda.set_device(rank)
        
        print(f"✅ Process {rank}/{world_size} initialized on GPU {rank}")
    
    @staticmethod
    def cleanup_distributed():
        """
        cleanup distributed process group after training
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            print("🧹 Distributed process group destroyed")
    
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
        
        # Create handlers with UTF-8 encoding to support emoji characters
        file_handler = logging.FileHandler(
            f"{log_dir}/alphadesign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        stream_handler = logging.StreamHandler()
        
        # Set UTF-8 encoding for console output on Windows
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except Exception:
                # If reconfigure fails, we'll use ASCII-safe logging
                pass
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[file_handler, stream_handler]
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
            # Verify directory was created
            if not os.path.exists(dir_path):
                self.logger.error(f"Failed to create directory: {dir_path}")
                raise RuntimeError(f"Cannot create required directory: {dir_path}")
            else:
                self.logger.info(f"✅ Directory ready: {dir_path}")
    
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
        
        # Initialize fitness evaluator with CFD results directory from config
        cfd_config = self.config.get('cfd_analysis', {})
        cfd_results_dir = cfd_config.get('results_output_dir', 'cfd_results')
        self.fitness_eval = FitnessEval(cfd_results_dir=cfd_results_dir)
        
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
        
        # determine device based on distributed setup
        if self.is_distributed:
            device = torch.device(f'cuda:{self.rank}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.neural_network, total_params = NetworkInitializer.setup_network(
            param_count,
            device=device,
            hidden_dim=min(512, max(256, param_count * 4)),
            depth=3
        )
        
        # wrap model with ddp for distributed training
        if self.is_distributed:
            self.neural_network = DDP(
                self.neural_network,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True  # useful for complex models(need to research on this more)
            )
            if self.is_main_process:
                self.logger.info(f"🌐 Model wrapped with DistributedDataParallel")
        
        # initialize optimizer with the wrapped model
        model_for_optimizer = self.neural_network.module if self.is_distributed else self.neural_network
        self.optimizer_manager = OptimizerManager(
            model_for_optimizer,
            learning_rate=self.config['neural_network']['learning_rate']
        )

        self.optimizer_manager.use_adamw_cosine(
            t0=10, 
            t_mult=2,
            lr=2e-4,  #lower learning rate for stability
            weight_decay=1e-3
        )
        
        self.loss_calculator = AlphaDesignLoss()
        self.param_tweaker = ParamterTweaker()
        
        if self.is_main_process:
            self.logger.info(f"🧠 Neural Network initialized: {total_params} parameters")
            if self.is_distributed:
                self.logger.info(f"🌐 Distributed training on {self.world_size} GPUs")
    
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
    
    def run_single_generation(self, base_params: F1FrontWingParams):
        generation_start = time.time()
        
        # Set generation number for CFD result tracking
        self.fitness_eval.set_generation(self.current_generation)
        
        # only show progress bar on main process
        show_progress = self.is_main_process
        
        # Progress bar for population evaluation
        eval_pbar = None
        if show_progress:
            eval_pbar = tqdm(
                total=len(self.current_population),
                desc="📊 Evaluating Population",
                unit="individual",
                position=1,
                leave=False
            )
        
        # Modified fitness evaluation with progress
        if show_progress:
            fitness_scores = self.fitness_eval.evaluate_pop_with_progress(
                self.current_population, eval_pbar
            )
            eval_pbar.close()
        else:
            fitness_scores = self.fitness_eval.evaluate_pop(self.current_population)
        
        # synchronize fitness scores across all processes if distributed
        if self.is_distributed:
            # gather fitness scores from all processes to main process
            gathered_scores = [None] * self.world_size
            dist.all_gather_object(gathered_scores, fitness_scores)
            if self.is_main_process:
                # flatten the gathered scores
                fitness_scores = [score for scores in gathered_scores for score in scores]
        
        # 2. find the best individual, so we can save and see how the design evolves
        valid_scores = [score for score in fitness_scores if isinstance(score, dict) and score.get('valid', False)]
        
        # Fix: Handle case when no valid individuals exist
        if not valid_scores:
            if self.is_main_process:
                self.logger.warning("⚠️ No valid individuals in population - creating recovery population")
            
            # Create recovery individuals with looser constraints
            recovery_population = []
            from genetic_algo_components.initialize_population import F1PopulInit
            
            pop_init = F1PopulInit(base_params, min(5, len(self.current_population)))
            recovery_individuals = pop_init.create_initial_population()
            
            # Add some variation to recovery individuals
            for individual in recovery_individuals:
                # Apply conservative mutations to ensure validity
                individual['max_thickness_ratio'] = max(0.08, min(0.25, individual.get('max_thickness_ratio', 0.15)))
                individual['camber_ratio'] = max(0.04, min(0.18, individual.get('camber_ratio', 0.08)))
                individual['total_span'] = max(1400, min(1800, individual.get('total_span', 1600)))
                recovery_population.append(individual)
            
            # Replace worst individuals with recovery individuals
            num_to_replace = min(len(recovery_population), len(self.current_population))
            self.current_population[-num_to_replace:] = recovery_population[:num_to_replace]
            
            # Re-evaluate with recovery population
            if show_progress:
                eval_pbar = tqdm(
                    total=len(self.current_population),
                    desc="📊 Re-evaluating Population",
                    unit="individual",
                    position=1,
                    leave=False
                )
                fitness_scores = self.fitness_eval.evaluate_pop_with_progress(
                    self.current_population, eval_pbar
                )
                eval_pbar.close()
            else:
                fitness_scores = self.fitness_eval.evaluate_pop(self.current_population)
            
            valid_scores = [score for score in fitness_scores if isinstance(score, dict) and score.get('valid', False)]
            
            if not valid_scores:
                # If still no valid individuals, use constraint compliance as fallback
                valid_scores = [score for score in fitness_scores if isinstance(score, dict) and score.get('constraint_compliance', 0) > 0.3]
        
        if valid_scores:
            best_score = max(valid_scores, key=lambda x: x.get('total_fitness', -1000))
            best_fitness = best_score['total_fitness']
            best_individual = self.current_population[fitness_scores.index(best_score)]
            
            # Save best design (only on main process)
            if self.is_main_process and self.current_generation % self.config['save_frequency'] == 0:
                self.save_best_design_stl(best_individual, self.current_generation)
                
            self.best_designs_history.append({
                'generation': self.current_generation,
                'fitness': best_fitness,
                'individual': best_individual.copy()
            })
        else:
            # Absolute fallback - use best constraint compliance
            best_score = max(fitness_scores, key=lambda x: x.get('constraint_compliance', 0) if isinstance(x, dict) else 0)
            best_fitness = best_score.get('constraint_compliance', 0) * 50  # Scale up constraint compliance
            best_individual = self.current_population[fitness_scores.index(best_score)]
            
            if self.is_main_process:
                self.logger.warning(f"⚠️ Using constraint compliance fallback: {best_fitness:.2f}")
        
        gen_pbar = None
        if show_progress:
            gen_pbar = tqdm(
                total=len(self.current_population),
                desc="🔄 Creating Next Generation",
                unit="individual",
                position=1,
                leave=False
            )
        
        new_population = self.generate_next_population_with_progress(gen_pbar)
        if show_progress:
            gen_pbar.close()
        
        # 5. neural network guidance, to be specific by the policy network and its output
        if self.config['neural_network_enabled']:
            nn_pbar = None
            if show_progress:
                nn_pbar = tqdm(
                    total=len(new_population),
                    desc="🧠 Neural Network Guidance",
                    unit="individual",
                    position=1,
                    leave=False
                )
            new_population = self.apply_neural_guidance_with_progress(new_population, nn_pbar)
            if show_progress:
                nn_pbar.close()
        
        # synchronize new population across all processes
        if self.is_distributed:
            # broadcast new population from main process to all other processes
            new_population = self.broadcast_population(new_population)
        
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
        
        if self.is_main_process:
            self.logger.info(f"✅ Generation {self.current_generation}: Best={best_fitness:.2f}, Avg={result['average_fitness']:.2f}, Time={generation_time:.1f}s")
        
        # Add memory cleanup at end of generation
        import gc
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        return result
    
    def broadcast_population(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        broadcast population from main process to all other processes
        """
        if not self.is_distributed:
            return population
        
        # broadcast population size first
        pop_list = [population] if self.is_main_process else [None]
        dist.broadcast_object_list(pop_list, src=0)
        
        return pop_list[0]
    
    def save_best_design_stl(self, individual: Dict[str, Any], generation: int):
        try:
            required_params = ['total_span', 'root_chord', 'tip_chord', 'flap_count']
            for param in required_params:
                if param not in individual:
                    self.logger.error(f"❌ Missing required parameter: {param}")
                    return
            
            params = F1FrontWingParams(**individual)
            
            wing_generator = UltraRealisticF1FrontWingGenerator(**individual)
            stl_filename = f"generation_{generation:03d}_best_design.stl"
            stl_path = os.path.join(self.output_dirs['stl_outputs'], stl_filename)
            
            wing_mesh = wing_generator.generate_complete_wing(stl_filename)
            
            import shutil
            generated_path = os.path.join("f1_wing_output", stl_filename)
            if os.path.exists(generated_path):
                shutil.copy2(generated_path, stl_path)
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

    def individual_to_tensor(self, individual: Dict[str, Any]):
        values = []
    
        scalar_params = [
            'total_span', 'root_chord', 'tip_chord', 'chord_taper_ratio',
            'sweep_angle', 'dihedral_angle', 'max_thickness_ratio', 
            'camber_ratio', 'camber_position', 'leading_edge_radius',
            'trailing_edge_thickness', 'upper_surface_radius', 'lower_surface_radius',
            'endplate_height', 'endplate_max_width', 'endplate_min_width',
            'endplate_thickness_base', 'endplate_forward_lean', 'endplate_rearward_sweep',
            'endplate_outboard_wrap', 'footplate_extension', 'footplate_height',
            'arch_radius', 'footplate_thickness', 'primary_strake_count',
            'y250_width', 'y250_step_height', 'y250_transition_length',
            'central_slot_width', 'pylon_count', 'pylon_spacing',
            'pylon_major_axis', 'pylon_minor_axis', 'pylon_length',
            'primary_cascade_span', 'primary_cascade_chord',
            'secondary_cascade_span', 'secondary_cascade_chord',
            'wall_thickness_structural', 'wall_thickness_aerodynamic',
            'wall_thickness_details', 'minimum_radius', 'mesh_resolution_aero',
            'mesh_resolution_structural', 'resolution_span', 'resolution_chord',
            'mesh_density', 'density', 'weight_estimate',
            'target_downforce', 'target_drag', 'efficiency_factor'
        ]
        
        for param in scalar_params:
            if param in individual:
                values.append(float(individual[param]))
        
        list_params = [
            'twist_distribution_range', 'flap_spans', 'flap_root_chords',
            'flap_tip_chords', 'flap_cambers', 'flap_slot_gaps',
            'flap_vertical_offsets', 'flap_horizontal_offsets', 'strake_heights'
        ]
        
        for param in list_params:
            if param in individual and isinstance(individual[param], list):
                values.extend([float(x) for x in individual[param]])
        
        return torch.tensor(values, dtype=torch.float32)
    
    def tensor_to_individual(self, tensor: torch.Tensor, template: Dict[str, Any]):
        individual = template.copy()
        tensor_values = tensor.cpu().numpy().tolist()
        
        idx = 0
        
        scalar_params = [
            'total_span', 'root_chord', 'tip_chord', 'chord_taper_ratio',
            'sweep_angle', 'dihedral_angle', 'max_thickness_ratio', 
            'camber_ratio', 'camber_position', 'leading_edge_radius',
            'trailing_edge_thickness', 'upper_surface_radius', 'lower_surface_radius',
            'endplate_height', 'endplate_max_width', 'endplate_min_width',
            'endplate_thickness_base', 'endplate_forward_lean', 'endplate_rearward_sweep',
            'endplate_outboard_wrap', 'footplate_extension', 'footplate_height',
            'arch_radius', 'footplate_thickness', 'primary_strake_count',
            'y250_width', 'y250_step_height', 'y250_transition_length',
            'central_slot_width', 'pylon_count', 'pylon_spacing',
            'pylon_major_axis', 'pylon_minor_axis', 'pylon_length',
            'primary_cascade_span', 'primary_cascade_chord',
            'secondary_cascade_span', 'secondary_cascade_chord',
            'wall_thickness_structural', 'wall_thickness_aerodynamic',
            'wall_thickness_details', 'minimum_radius', 'mesh_resolution_aero',
            'mesh_resolution_structural', 'resolution_span', 'resolution_chord',
            'mesh_density', 'density', 'weight_estimate',
            'target_downforce', 'target_drag', 'efficiency_factor'
        ]
        
        for param in scalar_params:
            if param in individual and idx < len(tensor_values):
                individual[param] = tensor_values[idx]
                idx += 1
        
        list_params = [
            'twist_distribution_range', 'flap_spans', 'flap_root_chords',
            'flap_tip_chords', 'flap_cambers', 'flap_slot_gaps',
            'flap_vertical_offsets', 'flap_horizontal_offsets', 'strake_heights'
        ]
        
        for param in list_params:
            if param in individual and isinstance(individual[param], list):
                param_length = len(individual[param])
                if idx + param_length <= len(tensor_values):
                    individual[param] = tensor_values[idx:idx + param_length]
                    idx += param_length
        
        return individual
    
    def apply_neural_guidance_with_progress(self, population: List[Dict[str, Any]], pbar: Optional[tqdm]):
        if self.neural_network is None:
            if pbar:
                pbar.update(len(population))
            return population
        
        model = self.neural_network.module if self.is_distributed else self.neural_network
        device = next(model.parameters()).device

        try:
            guided_population = []
            
            for individual in population:
                param_tensor = self.individual_to_tensor(individual)
                param_tensor = param_tensor.to(device) 

                if param_tensor.shape[0] != model.param_count:
                    if pbar:
                        pbar.set_postfix({'Status': f'Shape mismatch: {param_tensor.shape[0]} vs {model.param_count}'})
                    guided_population.append(individual) 
                    if pbar:
                        pbar.update(1)
                    continue

                if param_tensor.dim() == 1:
                    param_tensor = param_tensor.unsqueeze(0)
                
                # Get neural network predictions
                with torch.no_grad():
                    policy_output, value_output = self.neural_network(param_tensor)
                
                # Apply parameter tweaks
                guided_tensor = self.param_tweaker.apply_neural_tweaks(
                    param_tensor, policy_output, exploration=True
                )
                
                # Convert back to individual dict with proper mapping
                guided_individual = self.tensor_to_individual(guided_tensor.squeeze(), individual)
                
                guided_population.append(guided_individual)
                if pbar:
                    pbar.update(1)
            
            return guided_population
            
        except Exception as e:
            if self.is_main_process:
                self.logger.warning(f"⚠️ Neural guidance failed: {str(e)}")
            if pbar:
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

    def cleanup_old_checkpoints(self, keep_count: int = 10):
        
        try:
            checkpoint_dir = self.output_dirs['checkpoints']
            
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) 
                if f.startswith('checkpoint_gen_') and f.endswith('.json')
            ]
            
            if len(checkpoint_files) <= keep_count:
                return
            
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            
            files_to_remove = checkpoint_files[:-keep_count]
            
            for filename in files_to_remove:
                file_path = os.path.join(checkpoint_dir, filename)
                os.remove(file_path)
                
                gen_num = filename.split('_')[2].split('.')[0]
                
                summary_path = os.path.join(checkpoint_dir, f'summary_gen_{gen_num}.json')
                if os.path.exists(summary_path):
                    os.remove(summary_path)
                    
                nn_path = os.path.join(self.output_dirs['neural_networks'], f'network_gen_{gen_num}.pth')
                if os.path.exists(nn_path):
                    os.remove(nn_path)
            
            if files_to_remove:
                self.logger.info(f"🧹 Cleaned up {len(files_to_remove)} old checkpoints")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Checkpoint cleanup failed: {str(e)}")

    def save_checkpoint(self, generation: int, gen_result: Dict[str, Any]):
        # only save on main process
        if not self.is_main_process:
            return
        
        try:
            self.logger.info(f"💾 Saving checkpoint for generation {generation}...")
            
            checkpoint = {
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'current_population': self.current_population,
                'best_designs_history': self.best_designs_history,
                'training_metrics': self.training_metrics,
                'generation_result': gen_result,
                'generation_results' : self.generation_results,
                'pipeline_state': {
                    'current_generation': self.current_generation,
                    'total_runtime': time.time() - self.pipeline_start_time if hasattr(self, 'pipeline_start_time') else 0
                },
                'distributed_config': {
                    'world_size': self.world_size,
                    'is_distributed': self.is_distributed
                }
            }
            
            if self.neural_network is not None:
                nn_checkpoint_path = os.path.join(
                    self.output_dirs['neural_networks'], 
                    f'network_gen_{generation:03d}.pth'
                )
                
                # save the unwrapped model state dict
                model_to_save = self.neural_network.module if self.is_distributed else self.neural_network
                
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer_manager.get_optimizer().state_dict() if self.optimizer_manager else None,
                    'generation': generation,
                    'total_params': sum(p.numel() for p in model_to_save.parameters())
                }, nn_checkpoint_path)
                
                checkpoint['neural_network_checkpoint'] = nn_checkpoint_path
                self.logger.info(f"🧠 Neural network saved: {nn_checkpoint_path}")
            
            checkpoint_filename = f'checkpoint_gen_{generation:03d}.json'
            checkpoint_path = os.path.join(self.output_dirs['checkpoints'], checkpoint_filename)
            
            def json_serializer(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                return str(obj)
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=json_serializer)
            
            summary_checkpoint = {
                'generation': generation,
                'timestamp': checkpoint['timestamp'],
                'best_fitness': gen_result.get('best_fitness', -1000),
                'average_fitness': gen_result.get('average_fitness', -1000),
                'valid_individuals': gen_result.get('valid_individuals', 0),
                'population_size': len(self.current_population),
                'neural_network_enabled': self.config['neural_network_enabled']
            }
            
            summary_path = os.path.join(self.output_dirs['checkpoints'], f'summary_gen_{generation:03d}.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_checkpoint, f, indent=2)
            
            self.cleanup_old_checkpoints()
            
            if gen_result.get('best_fitness', -1000) > -1000:
                self.best_designs_history.append({
                    'generation': generation,
                    'fitness': gen_result['best_fitness'],
                    'individual': gen_result.get('best_individual', {}),
                    'timestamp': datetime.now().isoformat()
                })
            
            self.logger.info(f"✅ Checkpoint saved successfully: {checkpoint_path}")
            
            checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            self.logger.info(f"📊 Checkpoint size: {checkpoint_size:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {str(e)}")
        
    def train_neural_network_extended(self, recent_generations: List[Dict[str, Any]], generation: int):
        
        if not recent_generations or self.neural_network is None:
            return
        
        try:
            # get the actual model for parameter access
            model = self.neural_network.module if self.is_distributed else self.neural_network
            
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
            
            if self.is_main_process:
                self.logger.info(f"📚 Curriculum Phase: {training_phase} (Epochs: {epochs})")
            
            # Progress bar for extended training (only on main process)
            train_pbar = None
            if self.is_main_process:
                train_pbar = tqdm(
                    total=epochs,
                    desc=f"🧠 Extended Training ({training_phase})",
                    unit="epoch",
                    position=1,
                    leave=False
                )
            
            # Training loop with curriculum weights
            optimizer = self.optimizer_manager.get_optimizer()
            device = next(model.parameters()).device
            
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                
                # set model to training mode
                self.neural_network.train()
                
                for gen_data in recent_generations:
                    param_tensor = self.param_tweaker.genetic_to_neural_params(
                        [list(gen_data['best_individual'].values())]
                    )
                    
                    param_tensor = param_tensor.to(device)
                    
                    policy_output, value_output = self.neural_network(param_tensor)
                    
                    constraint_loss = self.calculate_constraint_loss(gen_data, value_output)
                    performance_loss = self.calculate_performance_loss(gen_data, value_output)
                    
                    total_loss = (weight_constraints * constraint_loss + 
                                 weight_performance * performance_loss)
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    batch_count += 1
                
                # synchronize loss across all processes if distributed
                if self.is_distributed:
                    epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
                    dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
                    epoch_loss = epoch_loss_tensor.item() / self.world_size
                
                avg_epoch_loss = epoch_loss / max(batch_count, 1)
                
                if train_pbar:
                    train_pbar.set_postfix({
                        'Loss': f"{avg_epoch_loss:.4f}",
                        'Phase': training_phase[:8]
                    })
                    train_pbar.update(1)
                
                if epoch % 10 == 0 and self.is_main_process:
                    self.logger.info(f"🧠 Epoch {epoch}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
                
                if avg_epoch_loss < 1e-6:
                    if self.is_main_process:
                        self.logger.info(f"🛑 Training converged early at epoch {epoch}")
                    break
            
            if train_pbar:
                train_pbar.close()
            
            if self.is_main_process:
                self.logger.info(f"✅ Extended neural network training completed - {epochs} epochs")
                self.save_training_metrics(generation, epoch_loss, training_phase)
            
            # synchronize all processes after training
            if self.is_distributed:
                dist.barrier()
            
            self.cleanup_generation()
            
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"❌ Extended neural network training failed: {str(e)}")
    
    def calculate_constraint_loss(self, gen_data: Dict[str, Any], value_output: torch.Tensor):
        try:
            best_individual = gen_data['best_individual']
            
            if isinstance(best_individual, list):
                best_individual = best_individual[0] if len(best_individual) > 0 else {}
            
            params = F1FrontWingParams(**best_individual)
            analyzer = F1FrontWingAnalyzer(params)
            constraint_results = analyzer.run_complete_analysis()
            
            compliance_value = float(constraint_results['overall_compliance'])
            target_compliance = torch.tensor([compliance_value], 
                                        dtype=torch.float32, 
                                        device=value_output.device)
            
            constraint_loss = torch.nn.functional.mse_loss(value_output, target_compliance)
            
            return constraint_loss
            
        except Exception as e:
            self.logger.warning(f"⚠️ Constraint loss calculation failed: {e}")
            return torch.tensor(0.01, dtype=torch.float32, requires_grad=True, device=value_output.device)

    def calculate_performance_loss(self, gen_data: Dict[str, Any], value_output: torch.Tensor):
        try:
            fitness_score = gen_data['best_fitness']
            
            if isinstance(fitness_score, list):
                fitness_score = fitness_score[0] if len(fitness_score) > 0 else 0.0
            
            fitness_score = float(fitness_score)
            normalized_fitness = min(1.0, max(0.0, fitness_score / 100.0))
            
            target_performance = torch.tensor([normalized_fitness], 
                                            dtype=torch.float32,
                                            device=value_output.device)
            
            performance_loss = torch.nn.functional.mse_loss(value_output, target_performance)
            
            return performance_loss
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance loss calculation failed: {e}")
            return torch.tensor(0.01, dtype=torch.float32, requires_grad=True, device=value_output.device)

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
            "final_population_size": len(self.current_population) if hasattr(self, 'current_population') else 0,
            "early_stopped": False,
            # "generated_results" : self.generation_results if hasattr(self, 'generation_results') else [],
        }
        
        summary_path = os.path.join(self.output_dirs['checkpoints'], 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Summary report saved: {summary_path}")
        
        return summary

def create_default_wing_params() -> F1FrontWingParams:
    """
    create default f1 front wing parameters with regulatory compliant values
    
    returns:
        f1frontwingparams: default wing configuration
    """
    return F1FrontWingParams(
        # basic wing dimensions
        total_span=1600.0,
        root_chord=400.0,
        tip_chord=200.0,
        chord_taper_ratio=0.5,
        sweep_angle=5.0,
        dihedral_angle=3.0,
        twist_distribution_range=[0.0, -2.0, -4.0],
        base_profile="NACA2412",
        
        # airfoil parameters
        max_thickness_ratio=0.15,
        camber_ratio=0.08,
        camber_position=0.4,
        leading_edge_radius=0.015,
        trailing_edge_thickness=0.002,
        upper_surface_radius=0.8,
        lower_surface_radius=0.6,
        
        # flap configuration
        flap_count=5,
        flap_spans=[200.0, 250.0, 300.0, 350.0, 500.0],
        flap_root_chords=[350.0, 320.0, 290.0, 260.0, 230.0],
        flap_tip_chords=[180.0, 170.0, 160.0, 150.0, 140.0],
        flap_cambers=[0.10, 0.09, 0.08, 0.07, 0.06],
        flap_slot_gaps=[8.0, 8.0, 8.0, 8.0, 8.0],
        flap_vertical_offsets=[0.0, -5.0, -10.0, -15.0, -20.0],
        flap_horizontal_offsets=[0.0, 10.0, 20.0, 30.0, 40.0],
        
        # endplate configuration
        endplate_height=300.0,
        endplate_max_width=80.0,
        endplate_min_width=40.0,
        endplate_thickness_base=4.0,
        endplate_forward_lean=15.0,
        endplate_rearward_sweep=10.0,
        endplate_outboard_wrap=5.0,
        
        # footplate configuration
        footplate_extension=50.0,
        footplate_height=20.0,
        arch_radius=15.0,
        footplate_thickness=3.0,
        
        # strakes and vortex generators
        primary_strake_count=3,
        strake_heights=[8.0, 10.0, 12.0],
        
        # y250 vortex region
        y250_width=30.0,
        y250_step_height=10.0,
        y250_transition_length=100.0,
        central_slot_width=50.0,
        
        # pylon configuration
        pylon_count=2,
        pylon_spacing=400.0,
        pylon_major_axis=25.0,
        pylon_minor_axis=15.0,
        pylon_length=150.0,
        
        # cascade elements
        cascade_enabled=True,
        primary_cascade_span=100.0,
        primary_cascade_chord=50.0,
        secondary_cascade_span=80.0,
        secondary_cascade_chord=40.0,
        
        # manufacturing parameters
        wall_thickness_structural=3.0,
        wall_thickness_aerodynamic=1.5,
        wall_thickness_details=1.0,
        minimum_radius=2.0,
        
        # mesh and resolution
        mesh_resolution_aero=0.5,
        mesh_resolution_structural=1.0,
        resolution_span=50,
        resolution_chord=30,
        mesh_density=1.0,
        surface_smoothing=True,
        
        # material properties
        material="carbon_fiber",
        density=1600.0,
        weight_estimate=15.0,
        
        # performance targets
        target_downforce=1500.0,
        target_drag=200.0,
        efficiency_factor=7.5
    )

def run_distributed_training(rank: int, world_size: int, config_path: str, base_params: F1FrontWingParams):
    """
    main function for each distributed process
    
    args:
        rank: process rank (0 to world_size-1)
        world_size: total number of processes
        config_path: path to configuration file
        base_params: base wing parameters
    """
    # setup distributed environment
    AlphaDesignPipeline.setup_distributed(rank, world_size)
    
    try:
        # create pipeline instance for this process
        pipeline = AlphaDesignPipeline(config_path, rank=rank, world_size=world_size)
        
        # run the complete pipeline
        results = pipeline.run_complete_pipeline(base_params)
        
        # only main process prints final results
        if rank == 0:
            print("✅ Distributed training completed successfully")
            print(f"📊 Total generations: {results.get('total_generations', 0)}")
    
    finally:
        # cleanup distributed environment
        AlphaDesignPipeline.cleanup_distributed()

def main_distributed(config_path: str = "config.json", world_size: int = None):
    """
    launcher for distributed training
    
    args:
        config_path: path to configuration file
        world_size: number of gpus to use (defaults to all available)
    """
    # determine world size
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size < 1:
        print("❌ No GPUs available for distributed training")
        return
    
    print(f"🚀 Starting distributed training with {world_size} GPUs")
    
    # create default base parameters (not from config file)
    base_params = create_default_wing_params()
    
    # spawn processes for distributed training
    mp.spawn(
        run_distributed_training,
        args=(world_size, config_path, base_params),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # for distributed multi-gpu training:
    main_distributed("config.json", world_size=2)
    pass

