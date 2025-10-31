# MuTriMCTS

[![CI](https://github.com/lguibr/mutrimcts/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/mutrimcts/actions)
[![PyPI](https://img.shields.io/pypi/v/mutrimcts.svg)](https://pypi.org/project/mutrimcts/)

---

<img src="bitmap.png" alt="MuTriMCTS Logo" width="300"/>

---

**MuTriMCTS** is a high-performance Python package providing C++ bindings for MuZero Monte Carlo Tree Search (MCTS) operating in learned latent space.

## Features

- 🚀 **High Performance**: C++ core with Python bindings via pybind11
- 🧠 **MuZero Algorithm**: MCTS in learned latent space (no game engine required during search)
- 🎯 **Clean API**: Simple Protocol-based network interface
- 📦 **Easy Installation**: Available via PyPI
- ✅ **Well Tested**: Comprehensive test suite
- 🔧 **Configurable**: Flexible search parameters (simulations, CPUCT, discount, Dirichlet noise)

## Installation

### From PyPI (when published)

```bash
pip install mutrimcts
```

### From Source

```bash
git clone https://github.com/lguibr/mutrimcts.git
cd mutrimcts
pip install -e .
```

### Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/lguibr/mutrimcts.git
cd mutrimcts
pip install -e ".[dev]"

# Run tests
pytest tests/

# Clean build artifacts (if needed)
# rm -rf build/ src/mutrimcts.egg-info/ dist/ src/mutrimcts/mutrimcts_cpp.*.so
```

## Quick Start

```python
import mutrimcts
import numpy as np

# Implement your MuZero network
class MyMuZeroNetwork(mutrimcts.MuZeroNetworkInterface):
    def initial_inference(self, observation):
        """
        observation → (hidden_state, policy, value)
        """
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def recurrent_inference(self, hidden_state, action):
        """
        (hidden_state, action) → (next_hidden_state, reward, policy, value)
        """
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

# Use results for training and action selection
# - visit_counts: target for policy loss
# - root_value: used in value bootstrapping  
# - mcts_policy: for action selection (proportional to visits)
```

## API Reference

### Network Interface

```python
class MuZeroNetworkInterface(Protocol):
    def initial_inference(self, observation: Any) -> tuple[Any, dict[int, float], float]:
        """Returns: (hidden_state, policy_dict, value)"""
        ...
    
    def recurrent_inference(self, hidden_state: Any, action: int) -> tuple[Any, float, dict[int, float], float]:
        """Returns: (next_hidden_state, reward, policy_dict, value)"""
        ...
```

### Search Configuration

```python
config = SearchConfiguration(
    max_simulations=50,      # Number of MCTS simulations
    max_depth=10,            # Maximum search depth
    cpuct=1.25,              # PUCT exploration constant
    dirichlet_alpha=0.3,     # Dirichlet noise alpha
    dirichlet_epsilon=0.25,  # Dirichlet noise weight
    discount=0.997,          # Discount factor (gamma)
    mcts_batch_size=1        # Batch size for network calls
)
```

### MCTS Function

```python
def run_mcts(
    initial_observation: Any,
    network_interface: MuZeroNetworkInterface,
    config: SearchConfiguration
) -> tuple[dict[int, int], float, dict[int, float]]:
    """
    Returns:
        - visit_counts: dict[int, int] - Visit counts per action
        - root_value: float - Root node value estimate
        - mcts_policy: dict[int, float] - Normalized MCTS policy
    """
```

## Project Structure

```
mutrimcts/
├── src/mutrimcts/              # Python package source
│   ├── __init__.py             # Package exports
│   ├── config.py               # SearchConfiguration
│   ├── mcts_wrapper.py         # Python entry point
│   └── cpp/                    # C++ source code
│       ├── bindings.cpp        # pybind11 bindings
│       ├── mcts.h/.cpp         # MCTS algorithm
│       ├── python_interface.h  # Network interface
│       ├── config.h            # Config struct
│       └── CMakeLists.txt      # Build configuration
├── tests/                      # Test suite
│   └── test_muzero_mcts.py
├── pyproject.toml              # Package metadata
├── setup.py                    # Build script
└── README.md                   # This file
```

## How It Works

MuTriMCTS implements the MuZero algorithm:

1. **Initial Inference**: Converts raw observation to latent state
2. **Tree Search**: MCTS in latent space using learned dynamics
3. **Recurrent Inference**: Predicts next state, reward, policy, value
4. **Backpropagation**: Discounted value accumulation
5. **Result**: Visit counts and improved policy for training

### Key Differences from AlphaZero

| Feature | AlphaZero | MuZero (MuTriMCTS) |
|---------|-----------|-------------------|
| Search Space | Real game states | Learned latent states |
| Game Engine | Required during search | Only at root |
| State Representation | Actual game state | Hidden state tensor |
| Rewards | Only at terminal | Predicted per transition |
| Network Calls | `evaluate_state()` | `initial_inference()` + `recurrent_inference()` |

## Development

### Building from Source

```bash
# Install dependencies
pip install pybind11>=2.10 cmake>=3.14

# Build C++ extension
mkdir build && cd build
cmake ../src/mutrimcts/cpp
cmake --build . --config Release

# Copy to package
cp mutrimcts_cpp.*.so ../src/mutrimcts/
```

### Running Tests

```bash
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

If you use MuTriMCTS in your research, please cite:

```bibtex
@software{mutrimcts2025,
  author = {Luis Guilherme P. M.},
  title = {MuTriMCTS: MuZero MCTS in Learned Latent Space},
  year = {2025},
  url = {https://github.com/lguibr/mutrimcts}
}
```

## Links

- **Repository**: https://github.com/lguibr/mutrimcts
- **Issues**: https://github.com/lguibr/mutrimcts/issues
- **PyPI**: https://pypi.org/project/mutrimcts/

## Acknowledgments

Based on the MuZero algorithm by DeepMind. Optimized for research and experimentation.
