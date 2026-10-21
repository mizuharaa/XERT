#include "reach/ik.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <memory>
#include <set>

namespace reach {
std::vector<double> solve_ik(const mjModel* model, const mjData* state,
    const std::vector<int>& joints, const std::vector<SiteTarget>& targets,
    const std::vector<std::array<double, 2>>& limits, int iterations, double damping) {
    const int n = static_cast<int>(joints.size());
    if (!n || limits.size() != joints.size() || targets.empty() || iterations < 1 ||
            iterations > 10000 || !std::isfinite(damping) || damping <= 0 ||
            std::set<int>(joints.begin(), joints.end()).size() != joints.size())
        throw std::invalid_argument("Invalid IK dimensions, iterations or damping");
    for (int i = 0; i < n; ++i) {
        int j = joints[i];
        if (j < 0 || j >= model->njnt || (model->jnt_type[j] != mjJNT_HINGE && model->jnt_type[j] != mjJNT_SLIDE) ||
                !std::isfinite(limits[i][0]) || !std::isfinite(limits[i][1]) || limits[i][0] >= limits[i][1])
            throw std::invalid_argument("IK needs valid scalar joints and finite ordered limits");
        if (model->jnt_limited[j] && (limits[i][0] < model->jnt_range[2*j] - 1e-8 ||
                limits[i][1] > model->jnt_range[2*j+1] + 1e-8))
            throw std::invalid_argument("IK limits exceed model limits");
    }
    for (const auto& target : targets) {
        if (target.site < 0 || target.site >= model->nsite || !std::isfinite(target.weight) || target.weight <= 0 ||
                (!target.quaternion.empty() && target.quaternion.size() != 4))
            throw std::invalid_argument("Invalid IK site, weight or quaternion");
        for (double v : target.position) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite IK target");
        double norm = 0;
        for (double v : target.quaternion) { if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite quaternion"); norm += v*v; }
        if (!target.quaternion.empty() && std::abs(norm - 1.0) > 1e-6)
            throw std::invalid_argument("Quaternion must have unit length");
    }
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model), mj_deleteData);
    if (!data) throw std::runtime_error("Cannot allocate IK state");
    for (int i = 0; i < model->nq; ++i) {
        if (!std::isfinite(state->qpos[i])) throw std::invalid_argument("Nonfinite IK state");
        data->qpos[i] = state->qpos[i];
    }
    std::vector<double> jp(3*model->nv), jr(3*model->nv), system(n*n), rhs(n), row(n), delta(n);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        mj_kinematics(model, data.get()); mj_comPos(model, data.get());
        std::fill(system.begin(), system.end(), 0); std::fill(rhs.begin(), rhs.end(), 0);
        for (int c = 0; c < n; ++c) system[c*n+c] = damping*damping;
        for (const auto& target : targets) {
            mj_jacSite(model, data.get(), jp.data(), jr.data(), target.site);
            double error[6]{};
            for (int r = 0; r < 3; ++r) error[r] = target.position[r] - data->site_xpos[3*target.site+r];
            if (!target.quaternion.empty()) {
                double current[4], inverse[4], relative[4];
                mju_mat2Quat(current, data->site_xmat + 9*target.site);
                mju_negQuat(inverse, current);
                mju_mulQuat(relative, target.quaternion.data(), inverse);
                mju_quat2Vel(error+3, relative, 1.0);
            }
            for (int r = 0; r < (target.quaternion.empty() ? 3 : 6); ++r) {
                for (int c = 0; c < n; ++c)
                    row[c] = (r < 3 ? jp[r*model->nv+model->jnt_dofadr[joints[c]]] : jr[(r-3)*model->nv+model->jnt_dofadr[joints[c]]]);
                for (int a = 0; a < n; ++a) {
                    rhs[a] += target.weight * row[a] * error[r];
                    for (int b = 0; b < n; ++b) system[a*n+b] += target.weight * row[a] * row[b];
                }
            }
        }
        // Positive damping makes this normal system positive definite; partial pivoting handles scale.
        for (int k = 0; k < n; ++k) {
            int pivot = k;
            for (int i = k+1; i < n; ++i) if (std::abs(system[i*n+k]) > std::abs(system[pivot*n+k])) pivot = i;
            if (!std::isfinite(system[pivot*n+k]) || system[pivot*n+k] == 0) throw std::runtime_error("Singular IK system");
            for (int j = k; j < n; ++j) std::swap(system[k*n+j], system[pivot*n+j]);
            std::swap(rhs[k], rhs[pivot]);
            for (int i = k+1; i < n; ++i) {
                double factor = system[i*n+k]/system[k*n+k];
                for (int j = k+1; j < n; ++j) system[i*n+j] -= factor * system[k*n+j];
                rhs[i] -= factor * rhs[k];
            }
        }
        for (int i = n-1; i >= 0; --i) {
            double residual = rhs[i];
            for (int j = i+1; j < n; ++j) residual -= system[i*n+j] * delta[j];
            delta[i] = residual/system[i*n+i];
            if (!std::isfinite(delta[i])) throw std::runtime_error("Nonfinite IK update");
        }
        for (int c = 0; c < n; ++c) {
            auto& q = data->qpos[model->jnt_qposadr[joints[c]]];
            q = std::clamp(q + std::clamp(delta[c], -.15, .15), limits[c][0], limits[c][1]);
        }
    }
    std::vector<double> result(n);
    for (int c = 0; c < n; ++c) result[c] = data->qpos[model->jnt_qposadr[joints[c]]];
    return result;
}

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
