# Tests

This directory contains the automated tests for the `mutrimcts` package, primarily using the `pytest` framework.

## Running Tests

From the project root:

```bash
pytest tests/ -v
```

## Test Coverage

- `test_muzero_mcts.py` - Core MuZero MCTS functionality tests
  - Basic MCTS execution
  - Different simulation counts
  - Policy normalization
  - Discount factor variations
