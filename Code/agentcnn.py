import numpy as np
import torch as th
from torch import nn as nn
import torch.nn.functional as F
from torch import tensor
import os
    
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action

FOOD_VALUE = 100
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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.fc3 = nn.Linear(2880, 512)
        self.shared1 = nn.Linear(512, 512)
        self.policy1 = nn.Linear(512, 256)
        self.policy2 = nn.Linear(256, 32)
        self.action = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.shared1(x))
        x = F.relu(self.policy1(x))
        x = F.relu(self.policy2(x))
        x = self.action(x)
        x = x.argmax()
        return x

def get_geese_observation(state):
    """
    Given a particular geese, does some processing and returns a geese specific observation. 
    Unfortunately specific to the geese environment for now.
    Encoding as follows: 
    2: enemy snake head
    1: enemy snake body
    11: own head
    12: own body
    100: food
    """


    agent = state.index
    rows = 7
    columns = 11
    game_board_self = np.zeros(rows*columns, None)
    game_board_enemy = np.zeros(rows*columns, None)
    game_board_food = np.zeros(rows*columns, None)


    for i, geese in enumerate(state.geese):
        identify=0
        if i==agent:
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
    print('shape: ', game_board.shape)
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

def agentcnn(obs, config):
    model = ConvNet()
    path = os.getcwd()
    #print('pre state load')
    #print('some more output')
    

    state_dict = th.load('tmp/best_pytorch_model')
    print('model loaded')
    state_dict = {
        'conv1.weight': state_dict['features_extractor.conv1.weight'],
        'conv1.bias': state_dict['features_extractor.conv1.bias'],
        'conv2.weight': state_dict['features_extractor.conv2.weight'],
        'conv2.bias': state_dict['features_extractor.conv2.bias'],
        'fc3.weight': state_dict['features_extractor.fc3.weight'],
        'fc3.bias': state_dict['features_extractor.fc3.bias'],

        'shared1.weight': state_dict['mlp_extractor.shared_net.0.weight'],
        'shared1.bias': state_dict['mlp_extractor.shared_net.0.bias'],

        'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
        'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],
        'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
        'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],

        'action.weight': state_dict['action_net.weight'],
        'action.bias': state_dict['action_net.bias'],
    }
    print('test2')
    model.load_state_dict(state_dict)
    model.eval()
    print('test3')
    obs = tensor(get_geese_observation(obs)).reshape(1, 3, 7, 11).float()
    action = model(obs)
    return ACTIONS[int(action)]