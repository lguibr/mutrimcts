# File: tests/test_muzero_mcts.py
"""Tests for MuZero MCTS implementation."""

import pytest
import numpy as np
from typing import Any

from mutrimcts import MuZeroNetworkInterface, SearchConfiguration, run_mcts


class MockMuZeroNetwork(MuZeroNetworkInterface):
    """A mock MuZero network for testing."""

    def __init__(self, num_actions: int = 4, hidden_dim: int = 8):
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.initial_inference_count = 0
        self.recurrent_inference_count = 0

    def initial_inference(
        self, observation: Any
    ) -> tuple[Any, dict[int, float], float]:
        """
        Returns a dummy hidden state, uniform policy, and zero value.
        """
        self.initial_inference_count += 1
        
        # Create a dummy hidden state (numpy array)
        hidden_state = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Uniform policy over all actions
        uniform_prob = 1.0 / self.num_actions
        policy = {i: uniform_prob for i in range(self.num_actions)}
        
        # Zero value
        value = 0.0
        
        return hidden_state, policy, value

    def recurrent_inference(
        self, hidden_state: Any, action: int
    ) -> tuple[Any, float, dict[int, float], float]:
        """
        Returns a dummy next hidden state, zero reward, uniform policy, and zero value.
        """
        self.recurrent_inference_count += 1
        
        # Create a dummy next hidden state (modify the input slightly)
        if isinstance(hidden_state, np.ndarray):
            next_hidden_state = hidden_state + 0.1 * action
        else:
            next_hidden_state = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Zero reward
        reward = 0.0
        
        # Uniform policy
        uniform_prob = 1.0 / self.num_actions
        policy = {i: uniform_prob for i in range(self.num_actions)}
        
        # Zero value
        value = 0.0
        
        return next_hidden_state, reward, policy, value


@pytest.fixture
def mock_observation() -> dict[str, Any]:
    """Provides a dummy observation."""
    return {"state": np.array([1, 2, 3, 4], dtype=np.float32)}


@pytest.fixture
def mock_network() -> MockMuZeroNetwork:
    """Provides a mock MuZero network."""
    return MockMuZeroNetwork(num_actions=4, hidden_dim=8)


@pytest.fixture
def search_config() -> SearchConfiguration:
    """Provides a default search configuration."""
    return SearchConfiguration(
        max_simulations=10,
        max_depth=5,
        cpuct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=0.99,
        mcts_batch_size=1,
    )


def test_mcts_basic_run(
    mock_observation: dict[str, Any],
    mock_network: MockMuZeroNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test basic MCTS run with MuZero interface."""
    print("\n--- Starting Basic MuZero MCTS Test ---")
    
    visit_counts, root_value, mcts_policy = run_mcts(
        mock_observation, mock_network, search_config
    )
    
    print(f"Visit Counts: {visit_counts}")
    print(f"Root Value: {root_value}")
    print(f"MCTS Policy: {mcts_policy}")
    print(f"Initial inference calls: {mock_network.initial_inference_count}")
    print(f"Recurrent inference calls: {mock_network.recurrent_inference_count}")
    
    # Validate return types
    assert isinstance(visit_counts, dict), "visit_counts should be a dict"
    assert isinstance(root_value, float), "root_value should be a float"
    assert isinstance(mcts_policy, dict), "mcts_policy should be a dict"
    
    # Validate visit counts
    assert len(visit_counts) > 0, "Should have visit counts for some actions"
    for action, count in visit_counts.items():
        assert isinstance(action, int), "Actions should be integers"
        assert isinstance(count, int), "Visit counts should be integers"
        assert count > 0, "Visit counts should be positive"
    
    # Visit counts should sum to approximately max_simulations
    total_visits = sum(visit_counts.values())
    assert total_visits <= search_config.max_simulations, (
        f"Total visits ({total_visits}) should not exceed max_simulations ({search_config.max_simulations})"
    )
    
    # Validate MCTS policy
    assert len(mcts_policy) == len(visit_counts), (
        "MCTS policy should have same actions as visit counts"
    )
    for action, prob in mcts_policy.items():
        assert isinstance(action, int), "Actions should be integers"
        assert isinstance(prob, float), "Probabilities should be floats"
        assert 0.0 <= prob <= 1.0, "Probabilities should be in [0, 1]"
    
    # MCTS policy should sum to approximately 1.0
    policy_sum = sum(mcts_policy.values())
    assert abs(policy_sum - 1.0) < 1e-5, (
        f"MCTS policy should sum to 1.0, got {policy_sum}"
    )
    
    # Should have called initial_inference exactly once
    assert mock_network.initial_inference_count == 1, (
        "Should call initial_inference exactly once"
    )
    
    # Should have called recurrent_inference multiple times (at least once per simulation)
    assert mock_network.recurrent_inference_count >= 1, (
        "Should call recurrent_inference at least once"
    )
    
    print("--- Test Passed ---")


def test_mcts_different_simulations(
    mock_observation: dict[str, Any],
    mock_network: MockMuZeroNetwork,
) -> None:
    """Test MCTS with different simulation counts."""
    print("\n--- Testing Different Simulation Counts ---")
    
    for num_sims in [5, 10, 20]:
        config = SearchConfiguration(
            max_simulations=num_sims,
            max_depth=10,
            cpuct=1.25,
            dirichlet_alpha=0.0,  # Disable noise for determinism
            dirichlet_epsilon=0.0,
            discount=1.0,
        )
        
        visit_counts, root_value, mcts_policy = run_mcts(
            mock_observation, mock_network, config
        )
        
        total_visits = sum(visit_counts.values())
        print(f"Simulations: {num_sims}, Total visits: {total_visits}")
        
        # More simulations should generally lead to more total visits
        assert total_visits <= num_sims, (
            f"Total visits ({total_visits}) should not exceed simulations ({num_sims})"
        )
        assert total_visits > 0, "Should have at least some visits"


def test_mcts_policy_normalization(
    mock_observation: dict[str, Any],
    mock_network: MockMuZeroNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test that MCTS policy is properly normalized."""
    print("\n--- Testing MCTS Policy Normalization ---")
    
    visit_counts, root_value, mcts_policy = run_mcts(
        mock_observation, mock_network, search_config
    )
    
    # Check normalization
    policy_sum = sum(mcts_policy.values())
    assert abs(policy_sum - 1.0) < 1e-5, (
        f"MCTS policy should sum to 1.0, got {policy_sum}"
    )
    
    # Check consistency between visit counts and policy
    total_visits = sum(visit_counts.values())
    for action in visit_counts.keys():
        expected_prob = visit_counts[action] / total_visits
        actual_prob = mcts_policy[action]
        assert abs(expected_prob - actual_prob) < 1e-5, (
            f"Policy probability for action {action} should match normalized visits"
        )
    
    print("--- Test Passed ---")


def test_mcts_with_discount(
    mock_observation: dict[str, Any],
    mock_network: MockMuZeroNetwork,
) -> None:
    """Test MCTS with different discount factors."""
    print("\n--- Testing Different Discount Factors ---")
    
    for discount in [0.9, 0.99, 1.0]:
        config = SearchConfiguration(
            max_simulations=10,
            max_depth=5,
            cpuct=1.25,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            discount=discount,
        )
        
        visit_counts, root_value, mcts_policy = run_mcts(
            mock_observation, mock_network, config
        )
        
        print(f"Discount: {discount}, Root value: {root_value}, Visits: {sum(visit_counts.values())}")
        
        # Should complete successfully
        assert len(visit_counts) > 0
        assert isinstance(root_value, float)
        assert len(mcts_policy) > 0
    
    print("--- Test Passed ---")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

