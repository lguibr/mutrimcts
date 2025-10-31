# `src/mutrimcts` - Python Package Source

This directory contains the core Python source code for the `mutrimcts` package.

## Files

- **`__init__.py`**: Package initialization and public API exports
- **`config.py`**: SearchConfiguration class (Pydantic model)
- **`mcts_wrapper.py`**: Python wrapper and MuZeroNetworkInterface Protocol
- **`py.typed`**: PEP 561 marker for type information

## C++ Extension

The `cpp/` subdirectory contains the C++ implementation compiled into `mutrimcts_cpp` module.

## Usage Flow

1. **Network Interface:** User implements `MuZeroNetworkInterface` Protocol
2. **Configuration:** Create `SearchConfiguration` instance
3. **Binding Layer:** Call `run_mcts()` which invokes the compiled C++ extension (`mutrimcts_cpp`)

See the main [README](../../README.md) for usage examples.
