#include "reach/ik.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace reach {
namespace {
// Six equations only; partial pivoting avoids a matrix dependency.
std::array<double, 6> solve(double (&a)[6][6], std::array<double, 6> b) {
    for (int k = 0; k < 6; ++k) {
        int pivot = k;
        for (int i = k+1; i < 6; ++i)
            if (std::abs(a[i][k]) > std::abs(a[pivot][k])) pivot = i;
        if (!std::isfinite(a[pivot][k]) || a[pivot][k] == 0)
            throw std::runtime_error("Singular IK system");
        for (int j = 0; j < 6; ++j) std::swap(a[k][j], a[pivot][j]);
        std::swap(b[k], b[pivot]);
        for (int i = k+1; i < 6; ++i) {
            double factor = a[i][k] / a[k][k];
            for (int j = k+1; j < 6; ++j) a[i][j] -= factor * a[k][j];
            b[i] -= factor * b[k];
        }
    }
    std::array<double, 6> x{};
    for (int i = 5; i >= 0; --i) {
        double residual = b[i];
        for (int j = i+1; j < 6; ++j) residual -= a[i][j] * x[j];
        x[i] = residual / a[i][i];
    }
    return x;
}
}

Action expert_action(Env& env) {
    const int phase = env.steps;
    const auto& start = env.initial_block;
    const auto* tray = env.model->body_pos + 3*env.tray;
    std::array<double, 3> target{};
    double grip;
    if (phase < 65) { target = start; target[2] += .06; grip = 1.0; }
    else if (phase < 140) { target = start; target[2] += .06*(1-(phase-65)/75.0); grip = 1.0; }
    else if (phase < 185) { target = start; grip = -.1; }
    else if (phase < 255) { target = start; target[2] += .075*((phase-185)/70.0); grip = -.1; }
    else if (phase < 340) {
        double alpha = (phase-255)/85.0;
        for (int i = 0; i < 3; ++i)
            target[i] = (1-alpha)*(start[i] + (i == 2 ? .075 : 0)) + alpha*(tray[i] + (i == 2 ? .10 : 0));
        grip = -.1;
    } else {
        std::copy_n(tray, 3, target.begin());
        if (phase < 410) { target[2] += .10 - .06*(phase-340)/70; grip = -.1; }
        else if (phase < 465) { target[2] += .04; grip = 1.0; }
        else { target[2] += .10; grip = 1.0; }
    }
    const auto* model = env.model;
    auto* ik = env.ik_data;
    std::copy_n(env.data->qpos, model->nq, ik->qpos);
    ik->qpos[env.qids[4]] = 0;
    std::vector<double> jp(3*model->nv), jr(3*model->nv);
    for (int iteration = 0; iteration < 35; ++iteration) {
        mj_kinematics(model, ik);
        mj_comPos(model, ik);
        mj_jacSite(model, ik, jp.data(), jr.data(), env.site);
        const auto* matrix = ik->site_xmat + 9*env.site;
        double axis[] = {matrix[1], matrix[4], matrix[7]};
        std::array<double, 6> error{};
        for (int i = 0; i < 3; ++i) error[i] = target[i] - ik->site_xpos[3*env.site+i];
        error[3] = .1*axis[1]; error[4] = -.1*axis[0];
        double jac[6][4]{};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                jac[r][c] = jp[r*model->nv+env.dids[c]];
                for (int k = 0; k < 3; ++k)
                    jac[r+3][c] += (.1*((r == k ? 1.0 : 0.0) - axis[r]*axis[k])) * jr[k*model->nv+env.dids[c]];
            }
        }
        double system[6][6]{};
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                for (int k = 0; k < 4; ++k) system[r][c] += jac[r][k]*jac[c][k];
                if (r == c) system[r][c] += .0001;
            }
        }
        const auto solution = solve(system, error);
        for (int c = 0; c < 4; ++c) {
            double delta = 0;
            for (int r = 0; r < 6; ++r) delta += jac[r][c]*solution[r];
            ik->qpos[env.qids[c]] = std::clamp(ik->qpos[env.qids[c]] + std::clamp(delta, -.15, .15),
                                             env.limits[c][0], env.limits[c][1]);
        }
    }
    Action action{};
    for (int i = 0; i < 6; ++i) {
        double value = i == 5 ? grip : env.data->ctrl[i] + std::clamp(ik->qpos[env.qids[i]] - env.data->ctrl[i], -.05, .05);
        action[i] = std::clamp(2*(value-env.limits[i][0])/(env.limits[i][1]-env.limits[i][0])-1, -1.0, 1.0);
    }
    return action;
}
}
