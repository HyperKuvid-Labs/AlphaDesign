# Test Requirements

This directory contains the test suite for the AlphaDesign RL components.

## Structure

- `test_genetic_algo/` - Tests for genetic algorithm components
- `test_neural_network/` - Tests for neural network components  
- `test_cfd_analysis/` - Tests for CFD analysis functionality
- `test_integration/` - Integration tests for the complete system
- `conftest.py` - Test configuration and utilities

## Running Tests

From the RL directory:

```bash
# Install test dependencies (if not already installed)
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_genetic_algo/ -v
python -m pytest tests/test_neural_network/ -v
python -m pytest tests/test_cfd_analysis/ -v
python -m pytest tests/test_integration/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Status

**Note**: The current test files contain placeholder tests since many components are still under development. As the project components are implemented, these tests should be updated with actual test cases.

### Current Test Coverage

- ✅ Test structure and framework setup
- ✅ Genetic algorithm component tests (placeholders)
- ✅ Neural network component tests (placeholders)
- ✅ CFD analysis tests (placeholders)
- ✅ Integration tests (placeholders)

### Adding New Tests

When implementing new functionality:

1. Add corresponding test files in the appropriate test directory
2. Use the existing test structure as a template
3. Replace placeholder tests with actual implementations
4. Ensure tests cover both success and failure cases
5. Add performance benchmarks for optimization algorithms

## Test Configuration

Tests use a mock configuration (see `conftest.py`) that:
- Reduces population sizes and generation counts for faster execution
- Disables computationally expensive operations (CFD, NN training)
- Uses temporary directories for test outputs
- Sets short timeouts to prevent hanging tests