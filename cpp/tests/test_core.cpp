#include "reach/env.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

// Explicit checks remain active in Release (unlike assert under NDEBUG).
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "Expected exported bench and positions");
        char error[2048]{};
        std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
        check(model != nullptr, error);
        for (int i = 0; i < model->ngeom; ++i) model->geom_friction[3*i] = 1.0;
        std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
        std::unique_ptr<mjData, decltype(&mj_deleteData)> ik(mj_makeData(model.get()), mj_deleteData);
        reach::Env env(model.get(), data.get(), ik.get());
        std::array<double, 2> xy{};
        char comma;
        std::ifstream positions(argv[2]);
        check(bool(positions >> xy[0] >> comma >> xy[1]) && comma == ',', "Missing seed 0 position");
        auto obs = env.reset_state(xy, 3);
        check(obs.size() == 26 && obs[25] == 0, "Observation dimension/time changed");
        const reach::Action initial = {0, 1, 1.2, -.6, 0, 1};
        for (int i = 0; i < 6; ++i) {
            check(obs[i] == static_cast<float>(initial[i]), "Reset joint layout");
            check(obs[6+i] == 0, "Reset velocity layout");
        }
        check(obs[12] == static_cast<float>(xy[0]) && obs[13] == static_cast<float>(xy[1]) &&
              obs[14] == static_cast<float>(.096) && obs[15] == 1 && obs[16] == 0 && obs[17] == 0 && obs[18] == 0,
              "Reset block pose layout");
        for (int i = 0; i < 3; ++i) {
            check(obs[19+i] == static_cast<float>(data->site_xpos[3*env.site+i]), "Reset site layout");
            check(obs[22+i] == static_cast<float>(model->body_pos[3*env.tray+i]), "Reset tray layout");
        }
        for (double invalid : {2.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            reach::Action action{}; action[0] = invalid;
            bool rejected = false;
            try { env.step(action); } catch (const std::invalid_argument&) { rejected = true; }
            check(rejected && env.steps == 0 && env.pending.size() == 3, "Invalid action changed state");
        }
        reach::Action action{}; action.fill(.123456789);
        for (int t = 0; t < 3; ++t) {
            env.step(action);
            for (int i = 0; i < 6; ++i) check(data->ctrl[i] == initial[i], "Delay queue applied too early");
        }
        env.step(action);
        for (int i = 0; i < 6; ++i) {
            float fraction = (static_cast<float>(action[i])+1.0f)*.5f;
            check(data->ctrl[i] == env.limits[i][0] + fraction*(env.limits[i][1]-env.limits[i][0]), "Delay/float32 target mismatch");
        }
        env.reset_state(xy, 0);
        check(env.steps == 0 && !env.finished && !env.lifted && env.stable_steps == 0 && env.pending.empty(), "Reset did not clear episode state");
        bool contact_while_lifted = false;
        reach::StepResult result{};
        while (!env.finished) {
            result = env.step(env.expert_action());
            contact_while_lifted |= result.held && result.object_position[2] > .145;
            for (float value : result.observation) check(std::isfinite(value), "Nonfinite observation");
        }
        check(contact_while_lifted, "Scripted trial did not lift through contact");
        check(result.success && result.reward == 1 && !result.held && result.failure_reason.empty(), "Scripted trial must succeed after release");
        bool rejected = false;
        try { env.step(action); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "Finished episode accepted a step");
        std::cout << "PASS reset layout, action validation, delay/float32, contact lift/release, terminal guard; scripted success at " << result.steps << " steps\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
