from dataclasses import dataclass
from enum import StrEnum
from random import random, choice, choices
from matplotlib import pyplot as plt
import numpy as np
from colorama import Fore
from copy import deepcopy

class CellType(StrEnum):
    EMPTY = "."
    WALL = "W"
    START = "S"
    END = "G"
    EXPERIENCE = "X"

class Action(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT" 

class QVersion(StrEnum):
    SARSA = "sarsa"
    SARASAMAX = "sarsamax"

@dataclass
class State:
    x: int
    y: int
    type: CellType
    Q: dict
    pi: Action


class QModel:

    def __init__(self, discount, alpha, epsilon, episodes=1000, qversion=QVersion.SARASAMAX):
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.qversion = qversion

    def get_next_action(self, state: State, episode: int):
        cur_epsilon = self.epsilon - episode * (self.epsilon/self.episodes)
        if random() < cur_epsilon:
            return choice(list(Action))
        else:
            return max(state.Q, key=state.Q.get)
        
    def update_Q(self, state: State, action: Action, reward: int, next_state: State, episode: int):
        if self.qversion == QVersion.SARASAMAX:
            state.Q[action] = state.Q.get(action, 0) + self.alpha * (reward + self.discount * max(next_state.Q.values()) - state.Q.get(action, 0))
        elif self.qversion == QVersion.SARSA:
            next_action = self.get_next_action(next_state, episode)
            state.Q[action] = state.Q.get(action, 0) + self.alpha * ((reward + self.discount * next_state.Q[next_action]) - state.Q.get(action, 0))
        else:
            assert ValueError, "Invalid Q version"


class Maze:

    def __init__(self, grid, model):
        self.grid = grid
        self.model = model
        
    def get_start_cell_pos(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j].type == CellType.START:
                    return i, j
                
        return None, None
    
    def get_state_action_pos(self, state: State, action: Action):
        new_state_x = state.x
        new_state_y = state.y
        if action == Action.UP:
            new_state_y -= 1
        elif action == Action.DOWN:
            new_state_y += 1
        elif action == Action.LEFT:
            new_state_x -= 1
        elif action == Action.RIGHT:
            new_state_x += 1
        
        if new_state_x < 0 or new_state_x >= len(self.grid[0]) or new_state_y < 0 or new_state_y >= len(self.grid):
            return state.y, state.x

        return new_state_y, new_state_x
    

    def take_action(self, state, action):
        new_state_x = state.x
        new_state_y = state.y
        if action == Action.UP:
            new_state_y -= 1
        elif action == Action.DOWN:
            new_state_y += 1
        elif action == Action.LEFT:
            new_state_x -= 1
        elif action == Action.RIGHT:
            new_state_x += 1
        
        new_state = None
        reward = 0
        if new_state_x < 0 or new_state_x >= len(self.grid[0]) or new_state_y < 0 or new_state_y >= len(self.grid):
            new_state, reward = state, -10
        elif self.grid[new_state_y][new_state_x].type == CellType.WALL:
            new_state, reward = state, -10
        elif self.grid[new_state_y][new_state_x].type == CellType.END:
            new_state, reward = self.grid[new_state_y][new_state_x], 10
        elif self.grid[new_state_y][new_state_x].type == CellType.EXPERIENCE:
            new_state = self.grid[new_state_y][new_state_x]
            reward = -1 if self.marked[new_state_y][new_state_x] == 1 else 8
        elif self.marked[new_state_y][new_state_x] == 1:
            new_state, reward = self.grid[new_state_y][new_state_x], -3
        else:
            new_state, reward = self.grid[new_state_y][new_state_x], -1
        return new_state, reward

    
    def train(self):
        self.cum_rewards = []
        for i in range(self.model.episodes):
            cum_reward = 0
            self.marked = [[0] * len(grid[i]) for i in range(len(grid))]
            x, y = self.get_start_cell_pos()
            start_state = self.grid[x][y]
            current_state = start_state
            while current_state.type != CellType.END:
                action = self.model.get_next_action(current_state, i)
                new_pos_y, new_pos_x = self.get_state_action_pos(current_state, action)
                next_state, reward = self.take_action(current_state, action)
                cum_reward += reward
                self.model.update_Q(current_state, action, reward, next_state, i)
                self.marked[new_pos_y][new_pos_x] = 1
                current_state = next_state
            # print("Episode", i, "done")
            self.cum_rewards.append(cum_reward)

    def extract_policy(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                self.grid[i][j].pi = max(self.grid[i][j].Q, key=self.grid[i][j].Q.get)
    
    def visualize_grid(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                symbol = "." if self.grid[i][j].type == CellType.EMPTY else str(self.grid[i][j].type)[:1]
                print(symbol, end=" ")
            print()
        print()

    
    def visualize_q_and_pi(self):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                state = grid[i][j]
                formatted_str = "{:.1f}".format(max(state.Q.values()))
                print(Fore.BLACK + formatted_str, end="\t")
            print()
        print()

        colored_grid = [[0] * len(self.grid[i]) for i in range(len(self.grid))]
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                symbol = ""
                match self.grid[i][j].pi:
                    case Action.UP:
                        symbol = "^"
                    case Action.DOWN:
                        symbol = "v"
                    case Action.LEFT:
                        symbol = "<"
                    case Action.RIGHT:
                        symbol = ">"
                if grid[i][j].type in (CellType.WALL, CellType.END):
                    symbol = "#"
                colored_grid[i][j] = [symbol, 0]

        i, j = self.get_start_cell_pos()
        cur_state = grid[i][j]
        while cur_state.type != CellType.END:
            colored_grid[i][j][1] = 1
            match self.grid[i][j].pi:
                case Action.UP:
                    i -= 1
                case Action.DOWN:
                    i += 1
                case Action.LEFT:
                    j -= 1
                case Action.RIGHT:
                    j += 1
            cur_state = grid[i][j]

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if colored_grid[i][j][1]:
                    print(Fore.GREEN + colored_grid[i][j][0], end=" ")
                else:
                    print(Fore.BLACK + colored_grid[i][j][0], end=" ")
            print()
        print()

    def plot_cum_reward(self):
        x = range(len(self.cum_rewards))

        plt.bar(x, self.cum_rewards)

        plt.xlabel('Episode number')
        plt.ylabel('Cumulative reward')
        plt.title('Cumulative reward over episode')

        plt.show()



if __name__ == "__main__":
    grid_size = 7
    grid = [[State(x, y, CellType.EMPTY, { action: 0 for action in Action }, None) for x in range(grid_size)] for y in range(grid_size)]
    
    #‌ manual
    grid_cell_types = [
        ['.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.'],
        ['S', 'W', 'W', 'W', 'W', 'W', 'G'],
    ]
    for i in range(grid_size):
        for j in range(grid_size):
            grid[i][j].type = grid_cell_types[i][j]

    # random
    # available_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    # start_pos, end_pos = choices(available_cells, k=2)
    # grid[start_pos[0]][start_pos[1]].type = CellType.START
    # grid[end_pos[0]][end_pos[1]].type = CellType.END
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if (i, j) == start_pos or (i, j) == end_pos:
    #             continue
    #         rand = random()
    #         if rand < 0.2:
    #             grid[i][j].type = CellType.WALL
    #         elif rand < 0.23:
    #             grid[i][j].type = CellType.EXPERIENCE

    model = QModel(discount=1, alpha=0.1, epsilon=0.9, episodes=1_000, qversion=QVersion.SARSA)
    maze = Maze(grid, model)
    maze.train()
    maze.extract_policy()
    maze.visualize_grid()
    maze.visualize_q_and_pi()
    maze.plot_cum_reward()
    
