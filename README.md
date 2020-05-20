# OmegaZero
Othello AI with AlphaZero algorithm

## Environment
- c++ 17
- python 3.6.7
- libtorch 1.4.0
- pytorch 1.4.0

## Build
In cpp directory  
`mkdir build && cd build`  
`cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..`  
`cmake --build .`

## Play games
In build directory
`./play <experiment id> <generation>`

## Features
- Self-play
    - c++ (libtorch)
    - multi-thread data generation
    - efficient computation by client / server system  
      (clients request state evaluations & server responds with neural network outputs)

- Model training
    - python (pytorch)
    - providing player's board and opponent's board as input  
      (instead of providing black board, white board and color board)
    - data augmentation (8x, flip & rotation)
    - remove duplication
