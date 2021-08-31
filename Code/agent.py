import numpy as np
import torch as th
from torch import nn as nn
import torch.nn.functional as F
from torch import tensor
import os
    
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action


FOOD_VALUE = 255
SELF_HEAD_VALUE = 50
SELF_BODY_VALUE = 51
ENEMY_HEAD_VALUE = 10
ENEMY_BODY_VALUE = 11


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.policy1 = nn.Linear(231, 64)
        self.policy2 = nn.Linear(64, 64)
        self.action = nn.Linear(64, 4)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = F.tanh(self.policy1(x))
        x = F.tanh(self.policy2(x))
        x = self.action(x)
        x = x.argmax()
        return x

class SACNet(nn.Module):
    def __init__(self):
        super(SACNet, self).__init__()
        self.policy1 = nn.Linear(231, 256)
        self.policy2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, 2)
        self.log_std = nn.Linear(256, 2)

        self.action_scale = 1
        self.action_bias = 0



    def forward(self, x):
        x = nn.Flatten()(x)
        x = F.relu(self.policy1(x))
        x = F.relu(self.policy2(x))
        #mean = th.tanh(self.mu(x)) * self.action_scale + self.action_bias
        mean = self.mu(x)
        logstd = self.log_std(x)
        #x = x.argmax()
        return mean



def get_geese_observation(state):
    """
    Given a particular geese, does some processing and returns a geese specific observation. 
    """

    if type(state) is list:
        state = state[0].observation
    else:

        state = Observation(state)

    
    agent = state.index
    rows = 7
    columns = 11
    game_board_self = np.zeros(rows*columns, None)
    game_board_enemy = np.zeros(rows*columns, None)
    game_board_food = np.zeros(rows*columns, None)


    for i, geese in enumerate(state.geese):
        identify=0
        if i==agent:
            identify=100
            for j, cell in enumerate(geese):
                if j == 0:
                    game_board_self[cell] = SELF_HEAD_VALUE
                else:
                    game_board_self[cell] = SELF_BODY_VALUE
        else:
            identify=-100
            for j, cell in enumerate(geese):
                if j == 0:
                    game_board_enemy[cell] = ENEMY_BODY_VALUE
                else:
                    game_board_enemy[cell] = ENEMY_HEAD_VALUE

    for food in state.food:
        game_board_food[food] = FOOD_VALUE
    game_board_self = game_board_self.reshape([rows, columns])
    game_board_enemy = game_board_enemy.reshape([rows, columns])
    game_board_food = game_board_food.reshape([rows, columns])

    head = get_geese_coord(game_board_self)[0]


    game_board_self = np.roll(game_board_self, 5-head[1], axis=1)
    game_board_self = np.roll(game_board_self, 3-head[0], axis=0)
    game_board_enemy = np.roll(game_board_enemy, 5-head[1], axis=1)
    game_board_enemy = np.roll(game_board_enemy, 3-head[0], axis=0)
    game_board_food = np.roll(game_board_food, 5-head[1], axis=1)
    game_board_food = np.roll(game_board_food, 3-head[0], axis=0)

    #game_board = game_board.reshape((game_board.shape[0], game_board.shape[1], 1))
    game_board = np.dstack((game_board_self, game_board_enemy, game_board_food))
    return game_board

def get_geese_coord(board):
    return get_coord_from_np_grid(board, SELF_HEAD_VALUE)

def get_food_coord(board):
    return get_coord_from_np_grid(board, FOOD_VALUE)

def get_enemy_geese_head_coord(board):
    return get_coord_from_np_grid(board, ENEMY_HEAD_VALUE)


def get_coord_from_np_grid(grid, value):
    coords = []
    for i in range(0, len(np.where(grid==value)[0])):
        coords.append((np.where(grid==value)[0][i], np.where(grid==value)[1][i]))
    return coords

ACTIONS = ['NORTH','SOUTH','WEST','EAST', '']

def agent3(obs, config):
    model = Net()

    #print('pre state load')
    #print('some more output')
    
    state_dict = th.load('tmp/best_pytorch_model')
    # Depending on whether I'm loading an MLP or CNN: 
    state_dict = {
        'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
        'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],
        'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
        'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],
        'action.weight': state_dict['action_net.weight'],
        'action.bias': state_dict['action_net.bias'],
    }
    '''
    # CNN
    state_dict = {
        'policy1.weight': state_dict['actor.latent_pi.0.weight'],
        'policy1.bias': state_dict['actor.latent_pi.0.bias'],
        'policy2.weight': state_dict['actor.latent_pi.2.weight'],
        'policy2.bias': state_dict['actor.latent_pi.2.bias'],
        'mu.weight': state_dict['actor.mu.weight'],
        'mu.bias': state_dict['actor.mu.bias'],
        'log_std.weight': state_dict['actor.log_std.weight'],
        'log_std.bias': state_dict['actor.log_std.bias'],
    }
    '''

    model.load_state_dict(state_dict)
    model.eval()

    obs = tensor(get_geese_observation(obs)).reshape(1, 7, 11, 3).float()
    action_value = model(obs)

    return ACTIONS[int(action_value)]