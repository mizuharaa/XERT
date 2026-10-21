#pragma once
#include "reach/env.hpp"

namespace reach {
Action expert_action(Env& env);

struct SiteTarget {
    int site;
    std::array<double, 3> position;
    std::vector<double> quaternion;  // Empty means position-only; otherwise wxyz.
    double weight;
};
std::vector<double> solve_ik(const mjModel* model, const mjData* state,
    const std::vector<int>& joints, const std::vector<SiteTarget>& targets,
    const std::vector<std::array<double, 2>>& limits, int iterations, double damping);
}
