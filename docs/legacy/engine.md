# AlphaDesign - Constrained Optimization Engine

**Genetic Algorithm System with an Optional Neural Value Model for F1 Wing Optimization**

## Overview

This directory contains the core constrained genetic-optimization engine for Formula 1 front-wing designs. The aerodynamic evaluator is a physics-informed empirical surrogate, not a high-fidelity CFD solver. The optional neural component currently trains a value estimate; policy guidance is disabled by default because no action/reward trajectories are recorded.

Checked-in checkpoints, logs, STL files, and CFD reports are legacy artifacts from earlier runs. They are retained for provenance only and must not be presented as validated paper results.

The supported package is now under `src/alphadesign`; this document preserves
the engine-level description while the repository layout is defined by the
root README.

## Architecture

### Core Components

- **Neural Network Pipeline** (`neural_network_components/`)
  - Shared feature extraction with value and retained policy heads
  - Value-head training from generation-level fitness labels
  - Policy guidance disabled by default pending action/reward data

- **Genetic Algorithm Engine** (`genetic_algo_components/`)
  - F1-specific population initialization
  - Adaptive mutation strategies for aerodynamic parameters
  - Intelligent crossover operations
  - Multi-objective fitness evaluation

- **Aerodynamic surrogate** (`cfd_analysis.py`)
  - Physics-informed empirical multi-element aerodynamic model
  - Performance metrics calculation (downforce, drag, L/D ratio)
  - Parallel evaluation capabilities
  - Smart result caching

- **Wing Generation** (`wing_generator.py`)
  - Parametric F1 wing geometry creation
  - F1 regulation compliance checking
  - STL file export for 3D visualization
  - Multi-element wing support

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv alphadesign-env
source alphadesign-env/bin/activate  # On Windows: alphadesign-env\Scripts\activate

# Install dependencies
python -m pip install -e .

# Verify installation
python examples/legacy/verify_setup.py
```

### Basic Usage

```bash
# Inspect the supported package command
python -m alphadesign --help

# Run with custom configuration
python -m alphadesign --config custom_configs/config.json

# Print the package scope without starting an optimization
python -m alphadesign --show-scope
```

## Configuration

The system is configured through `configs/config.json`:

```json
{
  "max_generations": 50,
  "population_size": 20,
  "neural_network_enabled": true,
  "cfd_analysis": {
    "enabled": true,
    "parallel_processes": 4
  }
}
```

### Key Parameters

- **max_generations**: Maximum optimization iterations
- **population_size**: Number of designs per generation
- **neural_network_enabled**: Enable RL guidance
- **parallel_processes**: CFD analysis parallelization

## System Requirements

- **Python 3.8+**
- **PyTorch** (CPU or CUDA)
- **NumPy, SciPy** for numerical computation
- **STL libraries** for 3D model export
- **4GB+ RAM** (8GB+ recommended)
- **Multi-core CPU** for parallel CFD

## Output

The system generates:

- **STL Files** (`f1_wing_output/`) - 3D models of optimized wings
- **Performance Reports** - Detailed analysis of design improvements
- **Training Logs** (`logs/`) - Optimization progress and metrics
- **Best Designs** - Top-performing configurations with parameters

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific component tests
python -m pytest tests/test_neural_network/ -v
python -m pytest tests/test_genetic_algo/ -v
```

### Code Structure

```
├── alphadesign.py           # Main entry point
├── main_pipeline.py         # Core optimization loop
├── wing_generator.py        # F1 wing geometry
├── cfd_analysis.py         # Fluid dynamics analysis
├── configs/config.json             # System configuration
├── genetic_algo_components/ # GA implementation
├── neural_network_components/ # NN implementation
├── tests/                  # Test suite
└── logs/                   # Runtime logs
```

## Advanced Features

### Neural Network Training

The optional value model learns from generation-level fitness labels:

- **Policy Network**: Retained for explicitly labelled frozen-policy ablations; disabled by default
- **Value Network**: Evaluates design quality without CFD
- **Recorded provenance**: Seeds, budgets, configuration, and commit metadata are written for each run

### Multi-Objective Optimization

Optimizes for multiple criteria:

- **Aerodynamic Performance**: Downforce and drag coefficients
- **Structural Integrity**: Safety factors and material stress
- **Manufacturing Feasibility**: Production constraints
- **Regulatory Compliance**: F1 technical regulations

### Performance Monitoring

- **Real-time Progress**: Live optimization metrics
- **Convergence Detection**: Automatic stopping when optimal
- **Resource Management**: Memory and CPU usage optimization
- **Distributed Computing**: Multi-machine scaling capabilities

## Research Applications

This system enables research in:

- **Hybrid AI Optimization**: Combining RL and evolutionary algorithms
- **Aerodynamic Design**: Advanced wing optimization techniques
- **Multi-Physics Simulation**: Coupling CFD with structural analysis
- **Automated Engineering**: AI-driven design exploration

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## Support

For technical issues:
1. Check the troubleshooting section in the main README
2. Review log files in `logs/`
3. Run `verify_setup.py` to check system configuration
4. Open an issue on GitHub with system details and error logs
