# Tests

This directory contains component, integration, and research-safety tests for AlphaDesign.

## Structure

- `test_genetic_algo/` - Tests for genetic algorithm components
- `test_neural_network/` - Tests for the optional neural value-model components
- `test_cfd_analysis/` - Tests for the empirical aerodynamic surrogate
- `test_integration/` - Integration tests for the complete system
- `test_paper_results_safety.py` - Unit, failure, provenance, and fitness safeguards
- `conftest.py` - Test configuration and utilities

## Running Tests

From the repository root:

```bash
python -m pip install -e .

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_genetic_algo/ -v
python -m pytest tests/test_neural_network/ -v
python -m pytest tests/test_cfd_analysis/ -v
python -m pytest tests/test_integration/ -v

# Run with coverage
python -m pytest tests/ --cov=alphadesign --cov-report=html
```

## Test Status

Some historical component tests predate the package reorganization and still encode obsolete assumptions. The focused safety suite is the required gate for unit handling, evaluation failures, fitness conventions, policy gating, and experiment metadata. Broader-suite failures must be reported separately rather than hidden.

### Adding New Tests

When implementing new functionality:

1. Add corresponding test files in the appropriate test directory
2. Use the existing test structure as a template
3. Replace placeholder tests with actual implementations
4. Ensure tests cover both success and failure cases
5. Keep performance benchmarks separate from correctness tests

## Test Configuration

Tests use a mock configuration (see `conftest.py`) that:
- Reduces population sizes and generation counts for faster execution
- Disables computationally expensive surrogate sweeps and neural training
- Uses temporary directories for test outputs
- Sets short timeouts to prevent hanging tests
