# TF-Agents tensorflow environment boilerplate from https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
from typing import Optional, Text

import numpy as np
import scipy
import pygame
import tf_agents.typing.types as types
from pygame.color import Color
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

agent_mode = 0

class TFGridWorld(py_environment.PyEnvironment):
    def __init__(self, rows=5, cols=5, terminal_idx=24, walls=(), agent_start_idx=0, reward_value=10):
        if terminal_idx in walls:
            raise ValueError("Goal state is contained within a wall...")
        if agent_start_idx in walls:
            raise ValueError("Agent would start in a wall...")

        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(rows*cols,), dtype=np.int32, minimum=0, maximum=1, name='observation'
        )
        self._state = np.zeros(rows*cols, dtype=np.int32)
        self._agent_pos = agent_start_idx
        self._state[agent_start_idx] = 1
        self._agent_start_idx = agent_start_idx
        self._episode_ended = False
        self._terminal_idx = terminal_idx
        self._rewards = np.zeros((rows * cols), dtype=np.int32) - 1
        if terminal_idx >= 0:  # allow the user to set terminal_idx = -1 if no goal is on map.
            self._rewards[terminal_idx] = reward_value
        self._actions = np.array([rows, 1, -rows, -1])
        self._cols = np.int32(cols)
        self._rows = np.int32(rows)
        self._walls = walls
        self._window = None

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def _reset(self) -> ts.TimeStep:
        self._state[:] = 0
        self._state[self._agent_start_idx] = 1
        self._agent_pos = self._agent_start_idx
        self._episode_ended = False

        return ts.restart(observation=self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # c_pos = np.where(self._state == 1)[0][0]
        c_pos = self._agent_pos
        new_pos = self.apply_delta(c_pos, self._actions[action])
        self._episode_ended = self._terminal_idx == new_pos
        self._state[c_pos] = 0
        self._state[new_pos] = 1
        c_pos = new_pos
        self._agent_pos = c_pos

        if self._episode_ended:
            return ts.termination(observation=self._state,
                                  reward=self._rewards[c_pos])
        else:
            return ts.transition(observation=self._state,
                                 reward=self._rewards[c_pos],
                                 discount=1.0)

    def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
        global agent_mode
        if mode == 'none':
            return

        window_width = window_height = 800
        if self._window is None:
            self._window = pygame.display.set_mode((window_width, window_height))
        w = self._window

        if self._episode_ended:
            w.fill((0, 0, 0))
            pygame.display.update()
            return None

        green = Color('green')
        grey = Color('grey')
        black = Color('black')
        white = Color('white')
        red = Color('red')
        blue = Color('blue')
        brown = Color('brown')

        c_w = window_width/self._cols       # column width
        r_w = window_height/self._rows      # row width

        on_grey = True
        for i in range(self._rows):
            for j in range(self._cols):
                idx = (self._rows * self._cols) + i - ((j+1) * self._cols)
                c_box = (i * c_w, j * r_w, (i+1) * c_w, (j+1) * r_w)    # current box coords.
                if idx == self._agent_pos:
                    w.fill(red if agent_mode == 0 else blue, c_box)   # blue = in-option
                elif idx == self._terminal_idx:
                    w.fill(green, c_box)
                elif idx == self._agent_start_idx:
                    w.fill(brown, c_box)
                elif idx in self._walls:
                    w.fill(black, c_box)
                else:
                    w.fill(grey if on_grey else white, c_box)
                on_grey = not on_grey
        pygame.display.update()
        return None

    def get_shape(self):
        return self._rows, self._cols

    def game_over(self):
        return self._episode_ended

    def apply_delta(self, pos, delta) -> int:
        # if we are in the left-most column, don't try to go left
        if pos / self._cols == pos // self._cols and delta == -1:
            return pos

        # if we are in the right-most column, don't try to go right
        elif (pos % self._cols) == self._cols - 1 and delta == 1:
            return pos

        # else make sure we are not going too far down/up
        elif not 0 <= pos + delta < self._cols * self._rows:
            return pos

        # wall check
        if pos+delta in self._walls:
            return pos

        return pos + delta

    def get_adjacent_cells(self, pos):
        if pos in self._walls:
            return []
        deltas = self._actions
        adjacents = np.zeros_like(self._actions)
        for idx, delta in enumerate(deltas):
            adjacents[idx] = self.apply_delta(pos, delta)
        return adjacents

    def get_degree_matrix(self):
        state_map = np.array([x for x in range(self._cols * self._rows)], dtype=np.int32)
        degree_matrix = np.zeros_like(state_map, dtype=np.int32)
        for idx in range(state_map.shape[0]):
            degree_matrix[idx] = np.sum(self.get_adjacent_cells(idx) != idx)

        return degree_matrix

    def get_adjacency_matrix(self):
        matrix_width = self._cols * self._rows
        adjacency_matrix = np.zeros((matrix_width, matrix_width), dtype=np.int32)
        for i in range(matrix_width):
            for j in range(matrix_width):
                if i == j:
                    adjacency_matrix[i, j] = 0
                else:
                    if j in self.get_adjacent_cells(i):
                        adjacency_matrix[i, j] = 1

        return adjacency_matrix

    def get_laplacian(self):
        deg = np.diag(self.get_degree_matrix())
        adj = self.get_adjacency_matrix()
        return deg - adj

    def get_eigenvalues(self, framework='scipy'):
        lap = self.get_laplacian()
        if framework == 'numpy':
            return np.linalg.eig(lap)
        elif framework == 'scipy':
            return scipy.linalg.eig(lap)

    def test_matrix_functions(self):
        deg = self.get_degree_matrix()
        adj = self.get_adjacency_matrix()
        for i in range(self._rows*self._cols):
            assert deg[i] == np.sum(adj[i])

    def get_walls(self):
        return self._walls
