#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <deque>
#include <string>

namespace reach {
using Action = std::array<double, 6>;
// The baseline observation has 26 entries (including its seven-value block pose).
using Observation = std::array<float, 26>;

struct StepResult {
    Observation observation;
    double reward;
    bool finished, success, lifted, held;
    int steps;
    std::array<double, 3> object_position;
    std::string failure_reason;
};

class Env {
public:
    // Borrowed pointers: their owners must outlive this core.
    Env(mjModel* model, mjData* data, mjData* ik_data, int horizon = 600);
    Observation reset_state(const std::array<double, 2>& block_xy, int delay_steps);
    StepResult step(const Action& action);
    Observation observation() const;
    Action expert_action();

    mjModel* model;
    mjData* data;
    mjData* ik_data;
    std::array<int, 6> qids{}, dids{};
    std::array<std::array<double, 2>, 6> limits{};
    int site, block, block_q, block_d, tray, horizon;
    int steps = 0, stable_steps = 0;
    bool lifted = false, finished = false;
    std::array<double, 3> initial_block{};
    std::deque<Action> pending;
};
}
