# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import DQNNet


class Agent:
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.agent = args.agent
        self.ddqn = args.agent == 'ddqn' or args.agent == 'ebql'
        self.ensemble_size = args.ensemble_size
        self.use_target = not args.no_target

        self.online_net = DQNNet(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQNNet(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
        self.optimiser = optim.RMSprop(
            self.online_net.parameters(),
            lr=args.learning_rate,
            alpha=0.95,
            eps=0.01
        )

    def reset_noise(self):
        pass

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return self.online_net(state.unsqueeze(0))[0].argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        ensemble_q = self.online_net(states)[1]  # q(s_t, ·; θonline)

        with torch.no_grad():
            # Calculate target values
            if self.ddqn:
                ensemble_online_qn = self.online_net(next_states)[1]  # q(s_t+n, ·; θonline)
            if self.use_target:
                ensemble_target_qn = self.target_net(next_states)[1]  # q(s_t+n, ·; θtarget)
            else:
                ensemble_target_qn = ensemble_online_qn  # q(s_t+n, ·; θonline)

        loss = 0
        for i in range(self.ensemble_size):
            q_a = ensemble_q[range(self.batch_size), i, actions]

            if self.ensemble_size > 1 and (self.ddqn or self.agent == 'maxmin-dqn'):
                if self.ddqn:
                    qn = torch.mean(torch.cat([ensemble_target_qn[:, :i, :], ensemble_target_qn[:, i+1:, :]], dim=1), dim=1)
                elif self.agent == 'maxmin-dqn':
                    qn = ensemble_target_qn.min(1)[0]
                else:
                    raise NotImplementedError
            else:
                qn = ensemble_target_qn.mean(1)
            if self.ddqn:
                argmax_indices_ns = ensemble_online_qn[:, i, :].argmax(1)  # Perform argmax action selection using online network: argmax_a[q(s_t+n, a; θonline)]
                qn_a = qn[range(self.batch_size), argmax_indices_ns]  # Double-Q q(s_t+n, argmax_a[q(s_t+n, a; θonline)]; θtarget)
            else:
                qn_a = qn.max(1)[0]

            q_target = returns + self.discount * nonterminals.squeeze(1) * qn_a

            loss += torch.nn.functional.smooth_l1_loss(q_a, q_target, reduction='none')

        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return self.online_net(state.unsqueeze(0))[0].max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
