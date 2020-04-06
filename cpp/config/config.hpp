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
    int policy_filter;
    int value_filter;
    int value_hidden;

    int n_game;
    int n_thread;
    int n_simulation;
    int device_id;
    char model_fname[100];
} config_t;

void init_config(const char *exp_path, int generation, int device_id);
const config_t& get_config();
void set_config(int n_thread, int n_simulation, float e_frac);
