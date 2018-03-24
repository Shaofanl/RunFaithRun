from RunFaithRun.learning.DQL import DDQL, Environment
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

class GridWorld(Environment):
    def Qnetwork(self, state):
        x = state
        x = layers.flatten(x)
        x = layers.linear(x, 20)
        x = layers.linear(x, self.action_count, scope='qvs')
        return x

    USED = -1
    GROUND = 0
    TRAP = 1
    AGENT = 2
    PUNISHMENT = -1 # -100
    TRAP_PUNISHMENT = -1
    REWARD = 100

    @property
    def state(self):
        canvas = self.grid.copy()
        if not self.done():
            canvas[self.pos[0], self.pos[1]] = self.AGENT
        return canvas

    def standon(self):
        return self.grid[self.pos[0], self.pos[1]]

    def outside(self):
        if np.any(self.pos < 0):
            return True
        if np.any(self.pos >= self.grid_size):
            return True
        return False

    def failed(self):
        if self.outside():
            return True
        if not self.successful() and self.steps > self.max_steps:
            return True
        return False

    def successful(self):
        if np.all(self.pos+1 == self.grid_size):
            return True
        return False
    
    def done(self):
        if self.failed() or self.successful():
            return True
        return False
    
    def step(self, action):
        # dx=-1, dx=1, dy=-1, dy=1
        self.steps += 1

        if action <= 1:
            dx, dy = action*2-1, 0
        else:
            dx, dy = 0, (action-2)*2-1
        lastpos = self.pos.copy()
        self.pos += [dx, dy]
        if self.outside():
            self.pos = lastpos

        if self.failed():
            return self.PUNISHMENT
        elif self.successful():
            return self.REWARD

        if self.standon() in [self.TRAP, self.USED]:
            return self.TRAP_PUNISHMENT 
        elif self.standon() == self.GROUND:
            self.grid[self.pos[0], self.pos[1]] = self.USED
            return 1
        else:
            return 0

    def __init__(self, grid_size=(5, 5), trap_ratio=0.2):
        self.grid_size = grid_size
        self.trap_ratio = trap_ratio
        self.state_shape = (None,)+grid_size
        self.action_count = 4
        self.max_steps = int(np.sum(grid_size)*1.5)

    def reset(self):
        # 0: ground, 1: trap, 2: agent 
        rng = np.random.RandomState(32)

        self.grid = rng.binomial(1, self.trap_ratio, size=self.grid_size)
        self.pos = np.array([0, 0])
        self.steps = 0

    def visualize(self, state=None):
        if state is None:
            canvas = self.state
        else:
            canvas = state
        for row in canvas:
            for cell in row:
                if cell == self.GROUND:
                    print('.', end='')
                elif cell == self.USED:
                    print('-', end='')
                elif cell == self.TRAP:
                    print('X', end='')
                elif cell == self.AGENT:
                    print('+', end='')
            print("")
        
if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()

    gridworld = GridWorld((5, 5), 0.2)

    ddql = DDQL(env = gridworld)
    ddql.build(learning_rate=1e-3)  #, gamma=0.9)
    ddql.train(max_replays=10000,
            iterations=10000,
            epsilon=0.9, epsilon_decay=1-5e-4, epsilon_min=0.1)

