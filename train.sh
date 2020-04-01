#!/bin/bash

n_thread=50
n_game=2500
n_simulation=100
n_iter=10000
# n_thread=1
# n_game=1
# n_simulation=1
# n_iter=1


if [ -z "$(ls model)" ]; then
    init_model_cmd="python python/init_model.py"
    echo $init_model_cmd
    eval $init_model_cmd    
else
    echo "model/ is not empty"
fi

for generation in {0..99}; do
# for generation in {0..6}; do
    if [ ! -e "mldata/${generation}.dat" ]; then
        self_play_cmd="./cpp/build/main ${generation} ${n_thread} ${n_game} ${n_simulation}"
        echo -e "\n${self_play_cmd}"
        eval $self_play_cmd || {
            break
        }

        cat_cmd="cat mldata/${generation}_*.dat > mldata/${generation}.dat && rm mldata/${generation}_*.dat"
        echo -e "\n${cat_cmd}"
        eval $cat_cmd || {
            break
        }
    else
        echo "mldata/${generation}.dat already exists"
    fi

    if [ ! -e "model/model_$((generation+1)).pt" ]; then
        model_train_cmd="python python/train.py ${generation} --n-iter ${n_iter}"
        echo -e "\n${model_train_cmd}"
        eval $model_train_cmd || {
            break
        }
    else
        echo "model/model_$((generation+1)).pt already exists"
    fi

    hrs=$(( SECONDS/3600 )); mins=$(( (SECONDS-hrs*3600)/60)); secs=$(( SECONDS-hrs*3600-mins*60 ))
    printf '\nTime spent: %02d:%02d:%02d\n' $hrs $mins $secs
done
