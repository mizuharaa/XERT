#include "reach/env.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        std::string model_path, positions_path;
        double friction = 1;
        int delay = 0, horizon = 600;
        for (int i = 1; i < argc; i += 2) {
            if (i+1 == argc) throw std::invalid_argument("Missing option value");
            std::string option = argv[i], value = argv[i+1];
            if (option == "--model") model_path = value;
            else if (option == "--positions") positions_path = value;
            else if (option == "--friction" || option == "--delay" || option == "--horizon") {
                size_t consumed;
                double number = std::stod(value, &consumed);
                if (consumed != value.size() || !std::isfinite(number)) throw std::invalid_argument("Invalid numeric option");
                if (option == "--friction") {
                    if (number < .1 || number > 2) throw std::invalid_argument("friction must be in [.1, 2]");
                    friction = number;
                } else {
                    int lower = option == "--delay" ? 0 : 1, upper = option == "--delay" ? 20 : 2000;
                    if (number != std::floor(number) || number < lower || number > upper)
                        throw std::invalid_argument("Invalid integer option range");
                    (option == "--delay" ? delay : horizon) = static_cast<int>(number);
                }
            } else throw std::invalid_argument("Unknown option: " + option);
        }
        if (model_path.empty() || positions_path.empty())
            throw std::invalid_argument("Usage: reach_trials --model bench.xml --positions positions.csv [--friction F] [--delay N] [--horizon H]");
        std::ifstream positions(positions_path);
        if (!positions) throw std::runtime_error("Cannot open positions CSV");
        char error[2048]{};
        std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error)), mj_deleteModel);
        if (!model) throw std::runtime_error(error);
        for (int i = 0; i < model->ngeom; ++i) model->geom_friction[3*i] = friction;
        std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
        std::unique_ptr<mjData, decltype(&mj_deleteData)> ik(mj_makeData(model.get()), mj_deleteData);
        reach::Env env(model.get(), data.get(), ik.get(), horizon);
        int trials = 0, successes = 0;
        std::string line;
        std::cout << std::setprecision(17) << std::boolalpha;
        while (std::getline(positions, line)) {
            std::istringstream row(line);
            std::array<double, 2> xy{};
            char comma;
            if (!(row >> xy[0] >> comma >> xy[1]) || comma != ',' || !(row >> std::ws).eof())
                throw std::invalid_argument("Expected exactly x,y on each CSV line");
            env.reset_state(xy, delay);
            reach::StepResult result{};
            while (!env.finished) result = env.step(env.expert_action());
            successes += result.success;
            std::cout << "{\"trial\":" << trials++ << ",\"position\":[" << xy[0] << ',' << xy[1]
                      << "],\"success\":" << result.success << ",\"steps\":" << result.steps << ",\"failure_reason\":";
            if (result.failure_reason.empty()) std::cout << "null";
            else std::cout << '"' << result.failure_reason << '"';
            std::cout << "}\n";
        }
        if (positions.bad() || trials == 0) throw std::runtime_error("Empty or unreadable positions CSV");
        std::cout << "{\"trials\":" << trials << ",\"successes\":" << successes
                  << ",\"friction\":" << friction << ",\"delay_steps\":" << delay << ",\"horizon\":" << horizon << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
