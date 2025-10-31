# File: mutrimcts/src/mutrimcts/mcts_wrapper.py
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .config import SearchConfiguration

logger = logging.getLogger(__name__)

# Type hint for the game state object expected by the network interfaces
GameState = Any

# --- Conditional Import for MyPy ---
if TYPE_CHECKING:
    from . import mutrimcts_cpp as mutrimcts_cpp_stub

    mutrimcts_cpp: type[mutrimcts_cpp_stub]
# --- End Conditional Import ---


@runtime_checkable
class MuZeroNetworkInterface(Protocol):
    """Protocol for MuZero network that operates in learned latent space."""

    def initial_inference(
        self, observation: GameState
    ) -> tuple[Any, dict[int, float], float]:
        """
        Takes a raw game observation and produces the initial hidden state, policy, and value.
        
        Args:
            observation: The raw game observation/state
            
        Returns:
            A tuple containing:
                - hidden_state: The learned latent representation (can be any type, typically a tensor)
                - policy_dict: Action probabilities as {action: probability}
                - value: The value estimate for this state
        """
        ...

    def recurrent_inference(
        self, hidden_state: Any, action: int
    ) -> tuple[Any, float, dict[int, float], float]:
        """
        Takes a hidden state and action, and predicts the next state, reward, policy, and value.
        
        Args:
            hidden_state: The current latent representation
            action: The action to take
            
        Returns:
            A tuple containing:
                - next_hidden_state: The predicted next latent representation
                - reward: The predicted immediate reward for this transition
                - policy_dict: Action probabilities from the next state as {action: probability}
                - value: The value estimate for the next state
        """
        ...


def run_mcts(
    initial_observation: GameState,
    network_interface: MuZeroNetworkInterface,
    config: SearchConfiguration,
) -> tuple[dict[int, int], float, dict[int, float]]:
    """
    Python entry point for MuZero MCTS in learned latent space.
    
    Args:
        initial_observation: The current game observation (raw features).
        network_interface: The MuZero network evaluation interface.
        config: The MCTS search configuration.
        
    Returns:
        A tuple containing:
            - visit_counts (dict[int, int]): Visit counts for actions from the root.
            - root_value (float): The final root value estimate after search.
            - mcts_policy (dict[int, float]): The MCTS-improved policy for the root (normalized visit counts).
    """
    # Validate config
    if not isinstance(config, SearchConfiguration):
        raise TypeError("config must be an instance of SearchConfiguration")

    # Network interface type check
    if not isinstance(network_interface, MuZeroNetworkInterface):
        raise TypeError(
            "network_interface must implement MuZeroNetworkInterface"
        )

    # Import the C++ extension
    try:
        import mutrimcts.mutrimcts_cpp as cpp_module
    except ImportError as e:
        raise ImportError(
            "MuTriMCTS C++ extension module ('mutrimcts.mutrimcts_cpp') not found or failed to import. "
            "Ensure the package was built correctly (`pip install -e .`). "
            f"Original error: {e}"
        ) from e

    # Ensure expected function exists
    if not hasattr(cpp_module, "run_mcts_cpp"):
        raise RuntimeError(
            "Loaded module missing 'run_mcts_cpp'. Build might be incomplete or corrupted."
        )

    # Call into C++ - it now returns a tuple (visit_counts, root_value, mcts_policy)
    try:
        result_tuple = cpp_module.run_mcts_cpp(
            initial_observation,
            network_interface,
            config,
        )
    except Exception as cpp_err:
        logger.error(f"Error during C++ MCTS execution: {cpp_err}", exc_info=True)
        raise

    # Validate and unpack the returned tuple
    if not isinstance(result_tuple, tuple) or len(result_tuple) != 3:
        logger.error(
            f"C++ MCTS returned unexpected type or length: {type(result_tuple)}"
        )
        raise RuntimeError("C++ MCTS returned invalid result structure")

    visit_counts_raw, root_value_raw, mcts_policy_raw = result_tuple

    # Validate visit counts
    validated_visit_counts: dict[int, int] = {}
    if not isinstance(visit_counts_raw, dict):
        logger.error(
            f"C++ MCTS returned unexpected type for visit counts: {type(visit_counts_raw)}"
        )
        raise TypeError("Visit counts must be a dictionary")
    else:
        # Filter and validate keys/values
        for k, v in visit_counts_raw.items():
            if isinstance(k, int) and isinstance(v, int):
                validated_visit_counts[k] = v
            else:
                logger.warning(
                    f"Skipping invalid visit count entry: ({k!r}:{type(k)}, {v!r}:{type(v)})"
                )

    # Validate root value
    if not isinstance(root_value_raw, (float, int)):
        logger.error(
            f"C++ MCTS returned unexpected type for root value: {type(root_value_raw)}"
        )
        raise TypeError("Root value must be a float or int")
    root_value: float = float(root_value_raw)

    # Validate MCTS policy
    validated_mcts_policy: dict[int, float] = {}
    if not isinstance(mcts_policy_raw, dict):
        logger.error(
            f"C++ MCTS returned unexpected type for MCTS policy: {type(mcts_policy_raw)}"
        )
        raise TypeError("MCTS policy must be a dictionary")
    else:
        # Filter and validate keys/values
        for k, v in mcts_policy_raw.items():
            if isinstance(k, int) and isinstance(v, (float, int)):
                validated_mcts_policy[k] = float(v)
            else:
                logger.warning(
                    f"Skipping invalid MCTS policy entry: ({k!r}:{type(k)}, {v!r}:{type(v)})"
                )

    return validated_visit_counts, root_value, validated_mcts_policy
