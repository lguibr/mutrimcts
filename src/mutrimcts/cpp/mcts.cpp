// File: mutrimcts/src/mutrimcts/cpp/mcts.cpp
#include "mcts.h"
#include "python_interface.h" // For Python interaction
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility> // For std::move
#include <tuple>   // For std::tuple return type

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mutrimcts
{

  // --- Node Implementation ---

  // Constructor for nodes
  Node::Node(Node *parent, Action action, float prior)
      : parent_(parent), action_taken_(action), prior_probability_(prior),
        hidden_state_(py::none()), reward_(0.0f) {}

  Node *Node::get_parent() const
  {
    return parent_;
  }

  Action Node::get_action_taken() const
  {
    return action_taken_;
  }

  bool Node::is_expanded() const
  {
    return !children_.empty();
  }

  float Node::get_value_estimate() const
  {
    if (visit_count_ == 0)
      return 0.0f;
    return static_cast<float>(total_action_value_ / visit_count_);
  }

  float Node::calculate_puct(const SearchConfig &config) const
  {
    if (!parent_)
      return std::numeric_limits<float>::infinity();

    float q_value = get_value_estimate();
    double parent_visits_sqrt = std::sqrt(static_cast<double>(std::max(1, parent_->visit_count_)));
    double exploration_term = config.cpuct * prior_probability_ * (parent_visits_sqrt / (1.0 + visit_count_));

    return q_value + static_cast<float>(exploration_term);
  }

  Node *Node::select_child(const SearchConfig &config)
  {
    if (children_.empty())
      return nullptr;

    Node *best_child = nullptr;
    float max_score = -std::numeric_limits<float>::infinity();

    for (auto const &[action, child_ptr] : children_)
    {
      float score = child_ptr->calculate_puct(config);
      if (score > max_score)
      {
        max_score = score;
        best_child = child_ptr.get();
      }
    }

    if (best_child == nullptr && !children_.empty())
    {
      std::cerr << "Warning: select_child failed to find a best child despite having children." << std::endl;
    }

    return best_child;
  }

  // Simplified expansion for MuZero - just create child nodes with priors
  void Node::expand(const PolicyMap &policy_map)
  {
    if (is_expanded())
      return;

    // Create child nodes for each action in the policy
    for (const auto &[action, prior] : policy_map)
    {
      children_[action] = std::make_unique<Node>(this, action, prior);
    }
  }

  // Backpropagate with discount factor
  void Node::backpropagate(float leaf_value, const SearchConfig &config)
  {
    float value = leaf_value;
    Node *current = this;

    while (current != nullptr)
    {
      current->visit_count_++;
      current->total_action_value_ += value;
      // Discount the value as we go up the tree
      value = current->reward_ + config.discount * value;
      current = current->parent_;
    }
  }

  void Node::set_parent(Node *new_parent)
  {
    parent_ = new_parent;
  }

  std::unique_ptr<Node> Node::detach_child(Action action)
  {
    auto it = children_.find(action);
    if (it == children_.end())
    {
      return nullptr;
    }
    std::unique_ptr<Node> child_ptr = std::move(it->second);
    children_.erase(it);
    return child_ptr;
  }

  // Dirichlet noise sampling helper
  void sample_dirichlet_simple(double alpha, size_t k, std::vector<double> &output, std::mt19937 &rng)
  {
    output.resize(k);
    std::gamma_distribution<double> dist(alpha, 1.0);
    double sum = 0.0;

    for (size_t i = 0; i < k; ++i)
    {
      output[i] = dist(rng);
      if (output[i] < 1e-9)
        output[i] = 1e-9;
      sum += output[i];
    }

    if (sum > 1e-9)
    {
      for (size_t i = 0; i < k; ++i)
        output[i] /= sum;
    }
    else
    {
      if (k > 0)
      {
        for (size_t i = 0; i < k; ++i)
          output[i] = 1.0 / k;
      }
    }
  }

  void Node::add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng)
  {
    if (children_.empty() || config.dirichlet_alpha <= 0 || config.dirichlet_epsilon <= 0)
      return;

    size_t num_children = children_.size();
    std::vector<double> noise;
    sample_dirichlet_simple(config.dirichlet_alpha, num_children, noise, rng);

    size_t i = 0;
    double total_prior = 0.0;

    for (auto &[action, child_ptr] : children_)
    {
      float current_prior = child_ptr->prior_probability_;
      if (!std::isfinite(current_prior))
      {
        std::cerr << "Warning: Non-finite prior probability encountered before adding noise for action " << action << ". Resetting to 0." << std::endl;
        current_prior = 0.0f;
      }
      child_ptr->prior_probability_ = (1.0f - static_cast<float>(config.dirichlet_epsilon)) * current_prior + static_cast<float>(config.dirichlet_epsilon * noise[i]);
      total_prior += child_ptr->prior_probability_;
      i++;
    }

    // Renormalize if needed
    if (std::abs(total_prior - 1.0) > 1e-6 && total_prior > 1e-9)
    {
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ /= static_cast<float>(total_prior);
      }
    }
    else if (total_prior <= 1e-9 && num_children > 0)
    {
      float uniform_prior = 1.0f / static_cast<float>(num_children);
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ = uniform_prior;
      }
      std::cerr << "Warning: Total prior probability near zero after adding noise. Resetting to uniform." << std::endl;
    }
  }

  // --- Helper function to calculate MCTS policy from visit counts ---
  PolicyMap calculate_mcts_policy(const VisitMap &visits)
  {
    PolicyMap policy;
    int total_visits = 0;

    for (const auto &[action, count] : visits)
    {
      total_visits += count;
    }

    if (total_visits > 0)
    {
      for (const auto &[action, count] : visits)
      {
        policy[action] = static_cast<float>(count) / total_visits;
      }
    }

    return policy;
  }

  // --- Main MuZero MCTS Logic ---

  PYBIND11_EXPORT std::tuple<VisitMap, Value, PolicyMap> run_mcts_cpp_internal(
      py::object initial_observation,
      py::object network_interface,
      const SearchConfig &config)
  {
    // Initialize random number generator
    std::mt19937 rng(std::random_device{}());

    // --- Step 1: Initial Inference ---
    InitialInferenceResult initial_result;
    try
    {
      initial_result = initial_inference(network_interface, initial_observation);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during initial inference: " << e.what() << std::endl;
      return {VisitMap{}, 0.0f, PolicyMap{}};
    }

    // --- Step 2: Create Root Node ---
    auto root = std::make_unique<Node>(nullptr, -1, 1.0f);
    root->hidden_state_ = initial_result.hidden_state;
    root->reward_ = 0.0f; // Root has no reward

    // Expand root with initial policy
    root->expand(initial_result.policy);

    // Add Dirichlet noise to root for exploration
    if (root->is_expanded())
    {
      root->add_dirichlet_noise(config, rng);
    }

    // Initialize root with value from network
    root->backpropagate(initial_result.value, config);

    // --- Step 3: MCTS Simulation Loop ---
    for (uint32_t sim = 0; sim < config.max_simulations; ++sim)
    {
      Node *current_node = root.get();
      std::vector<Node *> path; // Track the path for backpropagation
      path.push_back(current_node);

      // --- Selection Phase ---
      // Traverse the tree using PUCT until we reach a leaf node
      while (current_node->is_expanded())
      {
        Node *selected_child = current_node->select_child(config);
        if (!selected_child)
        {
          std::cerr << "Warning: Selection failed. Breaking simulation." << std::endl;
          break;
        }
        current_node = selected_child;
        path.push_back(current_node);
      }

      // --- Expansion Phase ---
      // We've reached a leaf node - need to expand it
      if (current_node->visit_count_ == 0)
      {
        // This is a new leaf - use the parent's hidden state and this node's action
        if (current_node->get_parent() == nullptr)
        {
          // This shouldn't happen (root should be expanded)
          std::cerr << "Warning: Reached unexpanded root during simulation." << std::endl;
          continue;
        }

        Node *parent = current_node->get_parent();
        Action action = current_node->get_action_taken();

        // Perform recurrent inference
        RecurrentInferenceResult recurrent_result;
        try
        {
          recurrent_result = recurrent_inference(network_interface, parent->hidden_state_, action);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Error during recurrent inference: " << e.what() << std::endl;
          continue;
        }

        // Store the results in the current node
        current_node->hidden_state_ = recurrent_result.next_hidden_state;
        current_node->reward_ = recurrent_result.reward;

        // Expand the node with the predicted policy
        current_node->expand(recurrent_result.policy);

        // Backpropagate the value
        current_node->backpropagate(recurrent_result.value, config);
      }
      else
      {
        // Node has been visited before but not expanded (shouldn't happen often)
        // Just backpropagate its current value estimate
        current_node->backpropagate(current_node->get_value_estimate(), config);
      }
    }

    // --- Step 4: Collect Results ---
    VisitMap visit_counts;
    for (const auto &[action, child_ptr] : root->children_)
    {
      visit_counts[action] = child_ptr->visit_count_;
    }

    // Calculate root value (average of accumulated values)
    Value root_value = root->get_value_estimate();

    // Calculate MCTS-improved policy (normalized visit counts)
    PolicyMap mcts_policy = calculate_mcts_policy(visit_counts);

    return {visit_counts, root_value, mcts_policy};
  }

} // namespace mutrimcts
