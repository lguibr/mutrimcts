// File: src/mutrimcts/cpp/python_interface.h
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic vector/map conversions
#include <map>       // Added for std::map
#include <stdexcept> // For std::runtime_error
#include <string>

namespace py = pybind11;

namespace mutrimcts
{

  // Define basic types used across C++/Python
  using Action = int;
  using Value = float;
  using PolicyMap = std::map<Action, float>;
  using VisitMap = std::map<Action, int>;

  // --- Helper functions to interact with Python objects ---

  inline py::object call_python_method(py::handle obj, const char *method_name)
  {
    try
    {
      return obj.attr(method_name)();
    }
    catch (py::error_already_set &e)
    {
      throw std::runtime_error("Python error in method '" + std::string(method_name) + "': " + e.what());
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error("C++ error calling method '" + std::string(method_name) + "': " + e.what());
    }
  }

  template <typename Arg>
  inline py::object call_python_method(py::handle obj, const char *method_name, Arg &&arg)
  {
    try
    {
      return obj.attr(method_name)(std::forward<Arg>(arg));
    }
    catch (py::error_already_set &e)
    {
      throw std::runtime_error("Python error in method '" + std::string(method_name) + "': " + e.what());
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error("C++ error calling method '" + std::string(method_name) + "': " + e.what());
    }
  }

  template <typename Arg1, typename Arg2>
  inline py::object call_python_method(py::handle obj, const char *method_name, Arg1 &&arg1, Arg2 &&arg2)
  {
    try
    {
      return obj.attr(method_name)(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
    }
    catch (py::error_already_set &e)
    {
      throw std::runtime_error("Python error in method '" + std::string(method_name) + "': " + e.what());
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error("C++ error calling method '" + std::string(method_name) + "': " + e.what());
    }
  }

  // --- MuZero Network Interface Structs ---

  struct InitialInferenceResult
  {
    py::object hidden_state;
    PolicyMap policy;
    Value value;
  };

  struct RecurrentInferenceResult
  {
    py::object next_hidden_state;
    Value reward;
    PolicyMap policy;
    Value value;
  };

  // --- MuZero Network Interface Functions ---

  inline InitialInferenceResult initial_inference(py::handle network, py::handle observation)
  {
    // Call network.initial_inference(observation)
    // Returns: (hidden_state, policy_dict, value)
    py::tuple result = call_python_method(network, "initial_inference", observation).cast<py::tuple>();
    if (result.size() != 3)
      throw std::runtime_error("Python 'initial_inference' must return (hidden_state, policy_dict, value).");

    py::object hidden_state = result[0];
    PolicyMap policy = result[1].cast<PolicyMap>();
    Value value = result[2].cast<Value>();

    return {hidden_state, policy, value};
  }

  inline RecurrentInferenceResult recurrent_inference(py::handle network, py::handle hidden_state, Action action)
  {
    // Call network.recurrent_inference(hidden_state, action)
    // Returns: (next_hidden_state, reward, policy_dict, value)
    py::tuple result = call_python_method(network, "recurrent_inference", hidden_state, action).cast<py::tuple>();
    if (result.size() != 4)
      throw std::runtime_error("Python 'recurrent_inference' must return (next_hidden_state, reward, policy_dict, value).");

    py::object next_hidden_state = result[0];
    Value reward = result[1].cast<Value>();
    PolicyMap policy = result[2].cast<PolicyMap>();
    Value value = result[3].cast<Value>();

    return {next_hidden_state, reward, policy, value};
  }

} // namespace mutrimcts
