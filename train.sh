#!/bin/bash

n_thread=50
n_game=50
n_simulation=100
# n_thread=1
# n_game=1
# n_simulation=1

init_model_cmd="python python/init_model.py"
echo $init_model_cmd
eval $init_model_cmd

# for generation in {0..99}; do
for generation in {1..99}; do
# for generation in {0..1}; do
    self_play_cmd="./cpp/build/main ${generation} ${n_thread} ${n_game} ${n_simulation}"
    echo -e "\n${self_play_cmd}"
    eval $self_play_cmd || {
        break
    }

    model_train_cmd="python python/train.py ${generation}"
    echo -e "\n${model_train_cmd}"
    eval $model_train_cmd || {
        break
    }
done
