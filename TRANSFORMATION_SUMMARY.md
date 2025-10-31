# MuZero MCTS Transformation Summary

## Overview
Successfully transformed `trimcts` (AlphaZero MCTS) → `mutrimcts` (MuZero MCTS in learned latent space)

**Date:** October 30, 2025  
**Version:** 0.1.0  
**Status:** ✅ Complete and Tested

---

## What Changed

### 1. Core Architecture
- **Before:** AlphaZero MCTS operating on real game states
- **After:** MuZero MCTS operating in learned latent space
- **Impact:** No longer requires game engine interaction during search

### 2. Network Interface
#### Removed: `AlphaZeroNetworkInterface`
```python
def evaluate_state(state) -> (policy, value)
def evaluate_batch(states) -> [(policy, value), ...]
```

#### Added: `MuZeroNetworkInterface`
```python
def initial_inference(observation) -> (hidden_state, policy, value)
def recurrent_inference(hidden_state, action) -> (next_hidden_state, reward, policy, value)
```

### 3. API Changes
#### Old `run_mcts()` signature:
```python
run_mcts(
    root_state: GameState,
    network_interface: AlphaZeroNetworkInterface,
    config: SearchConfiguration,
    previous_tree_handle: Optional[Handle] = None,
    last_action: int = -1
) -> (visit_counts, tree_handle, avg_depth)
```

#### New `run_mcts()` signature:
```python
run_mcts(
    initial_observation: GameState,
    network_interface: MuZeroNetworkInterface,
    config: SearchConfiguration
) -> (visit_counts, root_value, mcts_policy)
```

**Key differences:**
- No tree reuse (removed for correctness/simplicity)
- Returns root value and MCTS policy instead of tree handle
- Operates on observations, not game states

---

## Implementation Details

### C++ Core Changes

#### Node Structure (mcts.h)
**Removed:**
- `materialized_state_` - no longer stores game states
- `get_state()`, `is_terminal()` - no game state access
- Lazy state materialization logic

**Added:**
- `py::object hidden_state_` - learned latent representation
- `float reward_` - transition reward for this node

#### MCTS Algorithm (mcts.cpp)
1. **Root Initialization:**
   - Calls `initial_inference(observation)` to get initial hidden state
   - Expands root with network policy
   - Adds Dirichlet noise for exploration

2. **Simulation Loop:**
   - **Selection:** PUCT-based tree traversal (unchanged)
   - **Expansion:** Uses `recurrent_inference(hidden_state, action)` for new nodes
   - **Backpropagation:** Discounted value propagation with `reward + discount * value`

3. **Result Generation:**
   - Visit counts from root children
   - Root value (averaged accumulated value)
   - MCTS policy (normalized visit counts)

#### Network Interface (python_interface.h)
```cpp
struct InitialInferenceResult {
    py::object hidden_state;
    PolicyMap policy;
    Value value;
};

struct RecurrentInferenceResult {
    py::object next_hidden_state;
    Value reward;
    PolicyMap policy;
    Value value;
};
```

### Removed Components
- ✓ `mcts_manager.h` / `mcts_manager.cpp` - tree reuse infrastructure
- ✓ All game state interaction functions
- ✓ Capsule-based tree lifetime management
- ✓ AlphaZero network interface

---

## Test Results

### All Tests Passing ✅

```
============================= test session starts ==============================
platform darwin -- Python 3.10.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/lg/lab/mutrimcts
configfile: pyproject.toml
collected 4 items

tests/test_muzero_mcts.py ....                                           [100%]

============================== 4 passed in 0.14s ===============================
```

### Test Coverage
1. ✓ `test_mcts_basic_run` - Basic MCTS execution with mock network
2. ✓ `test_mcts_different_simulations` - Varying simulation counts (5, 10, 20)
3. ✓ `test_mcts_policy_normalization` - Policy sums to 1.0, consistent with visits
4. ✓ `test_mcts_with_discount` - Different discount factors (0.9, 0.99, 1.0)

### Integration Test Results
```
Visit counts: {0: 7, 1: 10, 2: 3, 3: 0}
Root value: 0.440
MCTS policy: {0: 0.35, 1: 0.50, 2: 0.15, 3: 0.0}
Total visits: 20 (matches max_simulations)
Policy sum: 1.000000 (perfectly normalized)
```

---

## File Changes Summary

### Renamed Files
- `src/trimcts/` → `src/mutrimcts/`

### Modified Files
- `pyproject.toml` - Updated package name, version, dependencies
- `setup.py` - Updated module name and CMake variables
- `src/mutrimcts/__init__.py` - Removed AlphaZero exports
- `src/mutrimcts/mcts_wrapper.py` - **Complete rewrite** for MuZero
- `src/mutrimcts/cpp/mcts.h` - Node restructure for latent space
- `src/mutrimcts/cpp/mcts.cpp` - **Complete rewrite** of MCTS algorithm
- `src/mutrimcts/cpp/python_interface.h` - New MuZero network interface
- `src/mutrimcts/cpp/bindings.cpp` - Simplified bindings
- `src/mutrimcts/cpp/CMakeLists.txt` - Updated project name, removed manager

### Deleted Files
- ✗ `src/mutrimcts/cpp/mcts_manager.h`
- ✗ `src/mutrimcts/cpp/mcts_manager.cpp`
- ✗ `tests/test_alpha_wrapper.py`

### New Files
- ✓ `tests/test_muzero_mcts.py` - Comprehensive MuZero tests

---

## Build Information

### Build Status: ✅ Success
```
[ 33%] Building CXX object CMakeFiles/mutrimcts_cpp.dir/bindings.cpp.o
[ 66%] Building CXX object CMakeFiles/mutrimcts_cpp.dir/mcts.cpp.o
[100%] Linking CXX shared module mutrimcts_cpp.cpython-310-darwin.so
[100%] Built target mutrimcts_cpp
```

### Build Artifacts
- `build/mutrimcts_cpp.cpython-310-darwin.so` - C++ extension module
- Copied to: `src/mutrimcts/mutrimcts_cpp.cpython-310-darwin.so`

### Requirements
- Python >= 3.9
- pybind11 >= 2.10
- CMake >= 3.14
- numpy >= 1.20.0
- pydantic >= 2.0.0

---

## Usage Example

```python
import mutrimcts
import numpy as np

# Implement your MuZero network
class MyMuZeroNetwork(mutrimcts.MuZeroNetworkInterface):
    def initial_inference(self, observation):
        # observation → (hidden_state, policy, value)
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def recurrent_inference(self, hidden_state, action):
        # (hidden_state, action) → (next_hidden_state, reward, policy, value)
        next_hidden, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden)
        return next_hidden, reward, policy, value

# Configure search
config = mutrimcts.SearchConfiguration(
    max_simulations=50,
    max_depth=10,
    cpuct=1.25,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    discount=0.997  # Important for MuZero!
)

# Run MCTS
network = MyMuZeroNetwork()
observation = get_current_observation()
visit_counts, root_value, mcts_policy = mutrimcts.run_mcts(
    observation, network, config
)

# Use results for training
# - visit_counts: target for policy loss
# - root_value: used in value bootstrapping
# - mcts_policy: for action selection
```

---

## Key Differences from AlphaZero

| Feature | AlphaZero (trimcts) | MuZero (mutrimcts) |
|---------|---------------------|-------------------|
| **Search Space** | Real game states | Learned latent states |
| **Game Interaction** | Required during search | Only at root |
| **Network Calls** | `evaluate_state()` | `initial_inference()` + `recurrent_inference()` |
| **State Representation** | Actual game state | Hidden state tensor |
| **Rewards** | Only at terminal | Predicted per transition |
| **Tree Reuse** | Supported | Removed (for simplicity) |
| **Returns** | (visits, handle, depth) | (visits, value, policy) |

---

## Performance Characteristics

### Network Calls
- **Initial inference:** 1 call per search
- **Recurrent inference:** 1 call per simulation (≈ max_simulations)
- **Total:** ~(1 + max_simulations) network calls

### Memory
- No game state copies
- Only stores hidden state tensors in tree
- Significantly reduced memory footprint for complex games

### Speed
- Build time: ~2.4s (CMake configuration)
- Compilation: ~3s (2 C++ files)
- Test execution: ~0.14s (4 tests)

---

## Next Steps

### For MuTriangle Integration
1. Implement actual MuZero network (representation, dynamics, prediction)
2. Connect to game environment for observations
3. Set up training loop with replay buffer
4. Tune hyperparameters (cpuct, discount, Dirichlet, simulations)

### Potential Improvements
- Add batched MCTS for parallel game inference
- Implement virtual loss for multi-threaded search
- Add temperature-based action selection
- Implement tree reuse (if needed for performance)

---

## Verification Commands

```bash
# Build from source
cd /Users/lg/lab/mutrimcts
mkdir -p build && cd build
cmake ../src/mutrimcts/cpp
cmake --build . --config Release
cp mutrimcts_cpp.cpython-310-darwin.so ../src/mutrimcts/

# Run tests
cd ..
PYTHONPATH=src python3 -m pytest tests/test_muzero_mcts.py -v

# Import test
python3 -c "import sys; sys.path.insert(0, 'src'); import mutrimcts; print(mutrimcts.__version__)"
```

---

## Success Metrics

✅ **All transformation phases completed:**
1. Phase 0: Namespace renaming
2. Phase 1: Python API redefinition
3. Phase 2: C++ network interface
4. Phase 3: Node restructuring
5. Phase 4: MCTS algorithm rewrite
6. Phase 5: Tree reuse removal
7. Phase 6: Bindings simplification
8. Phase 7: Testing

✅ **Build successful** (C++17, optimized)  
✅ **All tests passing** (4/4)  
✅ **Integration verified**  
✅ **API documented**  

---

## Conclusion

The `mutrimcts` library is now a complete, independent MuZero MCTS engine operating in learned latent space. It's ready for integration with the MuTriangle project.

**Repository:** https://github.com/lguibr/mutrimcts  
**License:** MIT  
**Author:** Luis Guilherme P. M.

