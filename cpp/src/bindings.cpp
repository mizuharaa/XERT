#include "reach/env.hpp"
#include "reach/ik.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;
namespace {
template <typename T, size_t N>
py::array_t<T> array(const std::array<T, N>& value) {
    py::array_t<T> result(N);
    std::copy(value.begin(), value.end(), result.mutable_data());
    return result;
}
}

PYBIND11_MODULE(_reach_core, module) {
    py::module_::import("mujoco");
    py::class_<reach::SiteTarget>(module, "SiteTarget")
        .def(py::init<int, std::array<double, 3>, std::vector<double>, double>());
    module.def("solve_ik", [](py::object model, py::object data, const std::vector<int>& joints,
        const std::vector<reach::SiteTarget>& targets, const std::vector<std::array<double, 2>>& limits,
        int iterations, double damping) {
        auto mujoco = py::module_::import("mujoco");
        if (!py::isinstance(model, mujoco.attr("MjModel")) || !py::isinstance(data, mujoco.attr("MjData")) ||
                data.attr("model").ptr() != model.ptr()) throw py::value_error("Expected matching MjModel and MjData");
        return reach::solve_ik(reinterpret_cast<mjModel*>(model.attr("_address").cast<std::uintptr_t>()),
                              reinterpret_cast<mjData*>(data.attr("_address").cast<std::uintptr_t>()),
                              joints, targets, limits, iterations, damping);
    });
    py::class_<reach::Env>(module, "Env")
        .def(py::init([](py::object model, py::object data, py::object ik_data, int horizon) {
            // Hold the Python owners, never free or duplicate their MuJoCo allocations.
            auto mujoco = py::module_::import("mujoco");
            if (!py::isinstance(model, mujoco.attr("MjModel")) ||
                !py::isinstance(data, mujoco.attr("MjData")) || !py::isinstance(ik_data, mujoco.attr("MjData")) ||
                data.attr("model").ptr() != model.ptr() || ik_data.attr("model").ptr() != model.ptr())
                throw py::value_error("Expected MjModel and its MjData owners");
            return reach::Env(reinterpret_cast<mjModel*>(model.attr("_address").cast<std::uintptr_t>()),
                              reinterpret_cast<mjData*>(data.attr("_address").cast<std::uintptr_t>()),
                              reinterpret_cast<mjData*>(ik_data.attr("_address").cast<std::uintptr_t>()), horizon);
        }), py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("reset_state", [](reach::Env& env, const std::array<double, 2>& xy, int delay) {
            return array(env.reset_state(xy, delay));
        })
        .def("observation", [](const reach::Env& env) { return array(env.observation()); })
        .def("expert_action", [](reach::Env& env) { return array(env.expert_action()); })
        .def("step", [](reach::Env& env, py::array_t<float, py::array::c_style | py::array::forcecast> action) {
            if (action.ndim() != 1 || action.shape(0) != 6)
                throw py::value_error("action must contain six finite normalized joint targets in [-1,1]");
            reach::Action values{};
            std::copy_n(action.data(), 6, values.begin());
            const auto result = env.step(values);
            py::dict info;
            info["success"] = result.success;
            info["failure_reason"] = result.failure_reason.empty() ? py::object(py::none()) : py::object(py::str(result.failure_reason));
            info["lifted"] = result.lifted;
            info["steps"] = result.steps;
            info["object_position"] = result.object_position;
            info["object_in_robot_contact"] = result.held;
            return py::make_tuple(array(result.observation), result.reward, result.finished, info);
        })
        .def_readonly("steps", &reach::Env::steps)
        .def_readonly("stable_steps", &reach::Env::stable_steps)
        .def_readonly("lifted", &reach::Env::lifted)
        .def_readonly("finished", &reach::Env::finished)
        .def_property_readonly("initial_block", [](const reach::Env& env) { return array(env.initial_block); })
        .def_property_readonly("pending", [](const reach::Env& env) {
            py::list result;
            for (const auto& item : env.pending) result.append(array(item));
            return result;
        });
}
