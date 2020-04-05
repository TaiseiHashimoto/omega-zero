typedef struct {
    float tau;
    float c_puct;
    float e_frac;
    float d_alpha;
    int e_step;

    int board_size;
    int n_action;
    int n_res_block;
    int res_filter;
    int head_filter;
    int value_hidden;

    int total_game;
    int n_thread;
    int n_simulation;
    int device_id;
    int generation;
    char model_dname[100];
} config_t;

void init_config(const char *exp_path, int device_id);
const config_t& get_config();
void set_config(int n_thread, int n_simulation, int generation);
