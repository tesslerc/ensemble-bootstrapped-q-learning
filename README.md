# ensemble-bootstrapped-q-learning
Code accompanying the ICML paper "Ensemble Boostrapped Q Learning"

Training the agent:
``python3.6 main.py --agent [dqn|ddqn|ebql|ensm-dqn|maxmin-dqn|rainbow'] --game [game] --enable-cudnn --seed [seed] --id [save name]``

Special thanks to Kai Arulkumaran for providing an open source repository for training Q-learning agents.
Our code has heavily relied on his implementation - https://github.com/Kaixhin/Rainbow
