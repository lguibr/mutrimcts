#pragma once

#include <pybind11/pybind11.h> // Include pybind11 first
#include <vector>
#include <map>
#include <memory> // For std::unique_ptr
#include <random>
#include <utility>  // For std::pair
#include <tuple>    // For std::tuple

#include "config.h"
#include "python_interface.h" // For types and Python interaction helpers

namespace py = pybind11;

namespace mutrimcts
{

  class Node
  {
  public:
    // Constructor for nodes
    Node(Node *parent, Action action, float prior = 0.0);
    ~Node() = default; // Use default destructor

    // Disable copy constructor and assignment operator
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // Enable move constructor and assignment operator
    Node(Node &&) = default;
    Node &operator=(Node &&) = default;

    // --- Core MCTS Methods ---
    bool is_expanded() const;
    float get_value_estimate() const;
    Node *select_child(const SearchConfig &config);
    void expand(const PolicyMap &policy_map);
    void backpropagate(float value, const SearchConfig &config);
    void add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng);

    // --- Tree Structure ---
    std::unique_ptr<Node> detach_child(Action action);
    void set_parent(Node *new_parent);
    Node *get_parent() const;
    Action get_action_taken() const; // Action that led *to* this node

    // --- Public Members ---
    std::map<Action, std::unique_ptr<Node>> children_;

    int visit_count_ = 0;
    double total_action_value_ = 0.0; // Use double for accumulation
    float prior_probability_ = 0.0;

    // MuZero-specific: latent state and reward
    py::object hidden_state_;  // The learned latent representation
    float reward_ = 0.0f;      // Reward for transitioning TO this node

  private:
    Node *parent_;        // Pointer to parent node
    Action action_taken_; // Action taken by parent to reach this node

    float calculate_puct(const SearchConfig &config) const;
  };

  // Main MCTS function signature for MuZero
  PYBIND11_EXPORT std::tuple<VisitMap, Value, PolicyMap> run_mcts_cpp_internal(
      py::object initial_observation,
      py::object network_interface,
      const SearchConfig &config);

} // namespace mutrimcts
