# `src/mutrimcts/cpp` - C++ Core Implementation

This directory contains the C++ source code for the high-performance MuZero Monte Carlo Tree Search (MCTS) engine used by the `mutrimcts` package.

## Files

- **`CMakeLists.txt`**: CMake build script for configuring the build process, finding dependencies (Python, Pybind11), and defining the C++ extension module (`mutrimcts_cpp`)
- **`bindings.cpp`**: Pybind11 code that creates Python bindings, exposing C++ functions to Python
- **`config.h`**: C++ SearchConfig struct matching the Python SearchConfiguration
- **`python_interface.h`**: Helper functions and structs for MuZero network interface (initial/recurrent inference)
- **`mcts.h`**: Node class and MCTS function declarations
- **`mcts.cpp`**: Core MuZero MCTS algorithm implementation

## Build Process

The build is managed by CMake and integrated with Python's setuptools via `setup.py`:

1. CMake locates Python and Pybind11
2. Compiles C++ sources with optimizations
3. Creates shared library (`mutrimcts_cpp.*.so` or `.pyd`)
4. Python imports as `mutrimcts.mutrimcts_cpp`

## Architecture

```
┌─────────────────┐
│ Python Wrapper  │ (mcts_wrapper.py)
│ run_mcts()      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pybind11        │ (bindings.cpp)
│ Type Conversion │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MCTS Core       │ (mcts.cpp)
│ Latent Space    │ • Selection (PUCT)
│ Tree Search     │ • Expansion (recurrent inference)
└────────┬────────┘ • Backpropagation (discounted)
         │
         ▼
┌─────────────────┐
│ Network Calls   │ (python_interface.h)
│ via Python      │ • initial_inference()
└─────────────────┘ • recurrent_inference()
```

The binding layer in `bindings.cpp` allows seamless calls between the Python wrapper in `src/mutrimcts/` and this C++ core.

## MuZero Algorithm

The implementation follows the MuZero algorithm:

1. **Root Initialization**: Call `initial_inference()` to get initial hidden state
2. **MCTS Simulations**: For each simulation:
   - **Selection**: Traverse tree using PUCT until reaching a leaf
   - **Expansion**: Call `recurrent_inference(hidden_state, action)` to predict next state, reward, policy, value
   - **Backpropagation**: Update visit counts and values with discount factor
3. **Result Generation**: Return visit counts, root value, and MCTS policy

Key innovation: Operates entirely in learned latent space - no game engine needed during search!
