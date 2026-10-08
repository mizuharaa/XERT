#include "reach/env.hpp"
#include "reach/ik.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace reach {
Env::Env(mjModel* model, mjData* data, mjData* ik_data, int horizon)
    : model(model), data(data), ik_data(ik_data), horizon(horizon) {
    // The core reads mjModel/mjData fields directly, so the loaded DLL must match the build headers.
    if (mj_version() != mjVERSION_HEADER)
        throw std::runtime_error("mujoco.dll version differs from the headers this core was built with; rebuild");
    if (!model || !data || !ik_data || model->nu != 6 || horizon < 1 || horizon > 2000)
        throw std::invalid_argument("Expected bench model/data and horizon in [1, 2000]");
    auto id = [model](mjtObj kind, const char* name) {
        int result = mj_name2id(model, kind, name);
        if (result < 0) throw std::invalid_argument(std::string("Missing bench element: ") + name);
        return result;
    };
    const char* names[] = {"Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L"};
    for (int i = 0; i < 6; ++i) {
        int joint = id(mjOBJ_JOINT, names[i]);
        if (model->jnt_type[joint] != mjJNT_HINGE ||
            model->actuator_trnid[2*i] != joint || model->actuator_trntype[i] != mjTRN_JOINT)
            throw std::invalid_argument("Bench joint/actuator layout mismatch");
        qids[i] = model->jnt_qposadr[joint];
        dids[i] = model->jnt_dofadr[joint];
        limits[i] = {model->actuator_ctrlrange[2*i], model->actuator_ctrlrange[2*i+1]};
        if (!(limits[i][0] < limits[i][1])) throw std::invalid_argument("Invalid actuator limits");
    }
    site = id(mjOBJ_SITE, "grasp");
    block = id(mjOBJ_BODY, "block");
    tray = id(mjOBJ_BODY, "tray");
    int joint = id(mjOBJ_JOINT, "block_joint");
    if (model->jnt_type[joint] != mjJNT_FREE) throw std::invalid_argument("Block needs a free joint");
    block_q = model->jnt_qposadr[joint];
    block_d = model->jnt_dofadr[joint];
}

Observation Env::reset_state(const std::array<double, 2>& xy, int delay_steps) {
    if (delay_steps < 0 || delay_steps > 20 || !std::isfinite(xy[0]) || !std::isfinite(xy[1]) ||
        std::abs(xy[0]) > .4 || std::abs(xy[1]) > .4)
        throw std::invalid_argument("Invalid block position or delay_steps");
    mj_resetData(model, data);
    data->qpos[block_q] = xy[0];
    data->qpos[block_q+1] = xy[1];
    data->qpos[block_q+2] = .096;
    const Action initial = {0, 1.0, 1.2, -.6, 0, 1.0};
    for (int i = 0; i < 6; ++i) data->ctrl[i] = data->qpos[qids[i]] = initial[i];
    mj_forward(model, data);
    steps = stable_steps = 0;
    lifted = finished = false;
    std::copy_n(data->xpos + 3*block, 3, initial_block.begin());
    pending.assign(delay_steps, initial);
    return observation();
}

Observation Env::observation() const {
    Observation obs{};
    for (int i = 0; i < 6; ++i) {
        obs[i] = static_cast<float>(data->qpos[qids[i]]);
        obs[6+i] = static_cast<float>(data->qvel[dids[i]]);
    }
    for (int i = 0; i < 7; ++i) obs[12+i] = static_cast<float>(data->qpos[block_q+i]);
    for (int i = 0; i < 3; ++i) {
        obs[19+i] = static_cast<float>(data->site_xpos[3*site+i]);
        obs[22+i] = static_cast<float>(model->body_pos[3*tray+i]);
    }
    obs[25] = static_cast<float>(static_cast<double>(steps) / horizon);
    return obs;
}

StepResult Env::step(const Action& action) {
    Action target{};
    for (int i = 0; i < 6; ++i) {
        float a = static_cast<float>(action[i]);
        if (!std::isfinite(a) || std::abs(a) > 1.0f + 1e-6f)
            throw std::invalid_argument("action must contain six finite normalized joint targets in [-1,1]");
        // NumPy adds and halves in float32 BEFORE multiplying by the float64 range.
        float fraction = (std::clamp(a, -1.0f, 1.0f) + 1.0f) * .5f;
        target[i] = limits[i][0] + fraction * (limits[i][1] - limits[i][0]);
    }
    if (finished) throw std::runtime_error("episode finished; reset before stepping");
    pending.push_back(target);
    std::copy(pending.front().begin(), pending.front().end(), data->ctrl);
    pending.pop_front();
    for (int i = 0; i < 20; ++i) mj_step(model, data);
    for (int i = 0; i < model->nq; ++i)
        if (!std::isfinite(data->qpos[i])) throw std::runtime_error("Nonfinite simulation state");
    for (int i = 0; i < model->nv; ++i)
        if (!std::isfinite(data->qvel[i])) throw std::runtime_error("Nonfinite simulation state");
    mj_forward(model, data);
    ++steps;
    const auto* pos = data->xpos + 3*block;
    const auto* tray_pos = model->body_pos + 3*tray;
    lifted |= pos[2] > .145;
    bool held = false;
    for (int i = 0; i < data->ncon; ++i) {
        const auto& contact = data->contact[i];
        int b1 = model->geom_bodyid[contact.geom1], b2 = model->geom_bodyid[contact.geom2];
        int other = b1 == block ? b2 : b1;
        if ((b1 == block || b2 == block) && other != 0 && other != tray) held = true;
    }
    const auto* velocity = data->qvel + block_d;
    double speed = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]);
    bool settled = lifted && std::abs(pos[0]-tray_pos[0]) < .029 && std::abs(pos[1]-tray_pos[1]) < .029 &&
                   .095 < pos[2] && pos[2] < .115 && speed < .025 && !held;
    stable_steps = settled ? stable_steps + 1 : 0;
    bool success = stable_steps >= 12;
    std::string failure = pos[2] < .04 ? "fell_off_table" : "";
    finished = success || !failure.empty() || steps >= horizon;
    if (finished && !success && failure.empty()) failure = lifted ? "timeout_after_lift" : "timeout_no_lift";
    return {observation(), static_cast<double>(success), finished, success, lifted, held, steps,
            {pos[0], pos[1], pos[2]}, failure};
}

Action Env::expert_action() { return reach::expert_action(*this); }
}
