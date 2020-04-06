# OmegaZero
Othello AI using AlphaZero algorithm

# Environment
- c++ 17
- python 3.6.7
- libtorch 1.4.0
- pytorch 1.4.0

# Features
- Self-play
    - c++ (with libtorch)
    - multi-thread data generation
    - efficient computation by client / server system  
      (clients request state evaluations & server responds with neural network outputs)

- Model training
    - python (with pytorch)

- Modification from AlphaZero
    - Instead of providing the neural network with side (black / white) channel, I swapped the boards according to the side.
    - much less training (due to computational power limit and simplicity of the game)
    - some hyper parameters changed
