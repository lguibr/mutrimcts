// File: mutrimcts/src/mutrimcts/cpp/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For map/vector/pair conversions
#include <pybind11/pytypes.h> // For py::object, py::handle

#include "mcts.h"             // C++ MCTS logic header
#include "config.h"           // C++ SearchConfig struct
#include "python_interface.h" // For types
#include <string>    // std::string
#include <stdexcept> // std::runtime_error
#include <tuple>     // For std::tuple

namespace py = pybind11;
namespace tc = mutrimcts; // Alias for your C++ namespace

// Helper function to transfer config from Python Pydantic model to C++ struct
static tc::SearchConfig python_to_cpp_config(const py::object &py_config)
{
  tc::SearchConfig cpp_config;
  try
  {
    cpp_config.max_simulations = py_config.attr("max_simulations").cast<uint32_t>();
    cpp_config.max_depth = py_config.attr("max_depth").cast<uint32_t>();
    cpp_config.cpuct = py_config.attr("cpuct").cast<double>();
    cpp_config.dirichlet_alpha = py_config.attr("dirichlet_alpha").cast<double>();
    cpp_config.dirichlet_epsilon = py_config.attr("dirichlet_epsilon").cast<double>();
    cpp_config.discount = py_config.attr("discount").cast<double>();
    cpp_config.mcts_batch_size = py_config.attr("mcts_batch_size").cast<uint32_t>();
  }
  catch (const py::error_already_set &e)
  {
    throw std::runtime_error(
        std::string("Error accessing SearchConfiguration attributes: ") + e.what());
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error(
        std::string("Error converting SearchConfiguration: ") + e.what());
  }
  return cpp_config;
}

// Wrapper function exposed to Python
// Returns a tuple (VisitMap, Value, PolicyMap)
std::tuple<tc::VisitMap, tc::Value, tc::PolicyMap> run_mcts_cpp_wrapper(
    py::object initial_observation,
    py::object network_interface,
    const py::object &config_py)
{
  tc::SearchConfig config_cpp = python_to_cpp_config(config_py);

  try
  {
    // Call the internal C++ MCTS implementation
    return tc::run_mcts_cpp_internal(
        initial_observation,
        network_interface,
        config_cpp);
  }
  catch (const py::error_already_set &)
  {
    throw; // Re-throw Python exceptions
  }
  catch (const std::exception &e)
  {
    throw py::value_error(std::string("Error in C++ MCTS execution: ") + e.what());
  }
}

PYBIND11_MODULE(mutrimcts_cpp, m)
{
  m.doc() = "C++ core for MuTriMCTS (MuZero MCTS in latent space)";

  // Expose the MCTS function
  m.def("run_mcts_cpp",
        &run_mcts_cpp_wrapper,
        py::arg("initial_observation"),
        py::arg("network_interface"),
        py::arg("config"),
        R"pbdoc(
            Runs MuZero MCTS simulations in learned latent space.

            Args:
                initial_observation: The raw game observation/state.
                network_interface: Python object implementing the MuZero network interface.
                config: Python SearchConfiguration object.

            Returns:
                A tuple containing:
                    - VisitMap (dict[int, int]): Visit counts for actions from the root.
                    - Value (float): The final root value estimate after search.
                    - PolicyMap (dict[int, float]): The MCTS-improved policy (normalized visit counts).
        )pbdoc",
        py::return_value_policy::move);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
