#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <iterator>
#include <cstring>

#include "config.hpp"
#include "picojson.h"


namespace {
config_t config;
std::random_device seed_gen;
std::default_random_engine engine(seed_gen());
}

void init_config(const char *exp_path, int device_id) {
    char config_fname[100];
    sprintf(config_fname, "%s/config.json", exp_path);

    std::ifstream ifs(config_fname);
    if (ifs.fail()) {
        fprintf(stderr, "CONFIG  cannot open file \"%s\"\n", config_fname);
        exit(-1);
    }
    const std::string json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    picojson::value v;
    const std::string err = picojson::parse(v, json);
    if (err.empty() == false) {
        std::cerr << err << std::endl;
        exit(-1);
    }

    picojson::object& obj = v.get<picojson::object>();

    config.tau = (float)obj["tau"].get<double>();
    config.c_puct = (float)obj["c_puct"].get<double>();
    config.e_frac = (float)obj["e_frac"].get<double>();
    config.d_alpha = (float)obj["d_alpha"].get<double>();
    config.e_step = (int)obj["e_step"].get<double>();
    // printf("CONFIG  tau=%f c_puct=%f e_frac=%f d_alpha=%f e_step=%d\n", config.tau, config.c_puct, config.e_frac, config.d_alpha, config.e_step);

    config.board_size = (int)obj["board_size"].get<double>();
    config.n_action = (int)obj["n_action"].get<double>();
    config.n_res_block = (int)obj["n_res_block"].get<double>();
    config.res_filter = (int)obj["res_filter"].get<double>();
    config.head_filter = (int)obj["head_filter"].get<double>();
    config.value_hidden = (int)obj["value_hidden"].get<double>();
    // printf("CONFIG  board_size=%d n_action=%d n_res_block=%d res_filter=%d head_filter=%d value_hidden=%d\n", config.board_size, config.n_action, config.n_res_block, config.res_filter, config.head_filter, config.value_hidden);

    config.total_game = (int)obj["total_game"].get<double>();
    config.n_thread = (int)obj["n_thread"].get<double>();
    config.n_simulation = (int)obj["n_simulation"].get<double>();
    // printf("CONFIG  total_game=%d n_thread=%d n_simulation=%d\n", config.total_game, config.n_thread, config.n_simulation);

    config.device_id = device_id;
    config.generation = -1;
    sprintf(config.model_jit_dname, "%s/model_jit", exp_path);
    // printf("CONFIG  model_jit_dname=%s\n", config.model_jit_dname);
}

const config_t& get_config() {
    return config;
}

void set_config(int n_thread, int n_simulation, int generation) {
    config.n_thread = n_thread;
    config.n_simulation = n_simulation;
    config.generation = generation;
}
