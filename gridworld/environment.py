import numpy as np
import cv2
from gym import spaces
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

NONSPATIAL = 0
SPATIAL = 1

HEIGHT = 9
WIDTH = 9
N_STAGES = 1  # some levels may have multiple stages - only 1 stage for now
EASY_MODE = False

LEVEL = np.array([
        [2, 0, 0, 0, 1, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0]],
        dtype=int)

class State(object):
    def __init__(self):
        self.t = None
        self.stage = 0
        self.done = None
        self.timeout = None
        self.agent_pos = [0, 0]
        self.goal_pos = [0, 0]


class Environment(object):
    def __init__(self, seed, slipping_prob):

        self.max_dim = HEIGHT - 1  # square assumption
        self.norm_factor = 2 / self.max_dim   # self.norm_factor = 2 / self.max_dim

        self.obs_form = SPATIAL
        self.easy_mode = EASY_MODE

        self.slipping_prob = slipping_prob

        self.agent_color = np.asarray([255, 255, 255])
        self.goal_color = np.asarray([153, 255, 153])
        self.ground_color = np.asarray([155, 155, 155])
        self.pit_color = np.asarray([0, 0, 0])

        self.grid = []
        self.agent_pos = []
        self.goal_pos = []
        self.pit_positions = []
        self.passage_positions = []

        for stage in range(N_STAGES):
            grid = np.copy(LEVEL)

            agent_ind = np.where(grid == 2)
            agent_pos = [agent_ind[0][0], agent_ind[1][0]]
            grid[agent_pos[0], agent_pos[1]] = 0

            goal_ind = np.where(grid == 3)
            goal_pos = [goal_ind[0][0], goal_ind[1][0]]
            grid[goal_pos[0], goal_pos[1]] = 0

            self.grid.append(grid)
            self.agent_pos.append(agent_pos)
            self.goal_pos.append(goal_pos)
            self.pit_positions.append(np.where(grid == 1))
            self.passage_positions.append(np.where(grid == 0))

        if self.obs_form == NONSPATIAL:
            self.obs_shape = (32,)
            self.obs_space = spaces.Box(low=-1, high=1, shape=self.obs_shape, dtype=np.float32)
        elif self.obs_form == SPATIAL:
            self.obs_shape = (HEIGHT, WIDTH, 3)
            self.obs_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.uint8)

        self.action_space = spaces.Discrete(4)

        self.state_id_dict = {}
        self.agent_pos_dict = {}
        self.transition_id_dict = {}

        state_id = 0
        transition_id = 0

        for s in range(N_STAGES):
            for n in range(len(self.passage_positions[s][0])):
                if not (self.passage_positions[s][0][n] == self.goal_pos[s][0] and
                        self.passage_positions[s][1][n] == self.goal_pos[s][1]):
                    self.state_id_dict[(s, self.passage_positions[s][0][n], self.passage_positions[s][1][n])] = state_id
                    self.agent_pos_dict[state_id] = (self.passage_positions[s][0][n], self.passage_positions[s][1][n])
                    for i_action in range(4):
                        self.transition_id_dict[
                            (s, self.passage_positions[s][0][n], self.passage_positions[s][1][n], i_action)] = \
                            transition_id
                        transition_id += 1
                    state_id += 1

        self.n_states = len(self.state_id_dict)
        self.n_transitions = len(self.transition_id_dict)

        self.base_obs = []
        for s in range(N_STAGES):
            base_obs = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            base_obs[self.goal_pos[s][0], self.goal_pos[s][1], 1] = 1
            for n in range(len(self.pit_positions[s][0])):
                base_obs[self.pit_positions[s][0][n], self.pit_positions[s][1][n], 2] = 1
            self.base_obs.append(base_obs)

        self.base_nonspatial_obs = []
        for s in range(N_STAGES):
            pit_positions = self.pit_positions[s]  # np.where format
            n_pits = len(pit_positions[0])
            base_obs = np.zeros((2*(2 + n_pits)), dtype=np.float32)
            for n in range(len(pit_positions[0])):
                base_obs[4 + 2*n], base_obs[4 + 2*n + 1] = pit_positions[0][n], pit_positions[1][n]
            base_obs[0], base_obs[1] = self.agent_pos[s][0], self.agent_pos[s][1]
            base_obs[2], base_obs[3] = self.goal_pos[s][0], self.goal_pos[s][1]
            base_obs *= self.norm_factor
            base_obs -= 1
            self.base_nonspatial_obs.append(base_obs)

        print(self.base_nonspatial_obs[0], len(self.base_nonspatial_obs[0]))

        self.base_obs_image = []
        for s in range(N_STAGES):
            base_obs_image = np.full((HEIGHT, WIDTH, 3), self.ground_color, dtype=np.uint8)
            base_obs_image[self.goal_pos[s][0], self.goal_pos[s][1], :] = self.goal_color
            for n in range(len(self.pit_positions[s][0])):
                base_obs_image[self.pit_positions[s][0][n], self.pit_positions[s][1][n], :] = self.pit_color
            self.base_obs_image.append(base_obs_image)

        self.optimal_actions = []
        self.dist_to_goal = []
        for s in range(N_STAGES):
            optimal_actions = np.zeros((HEIGHT, WIDTH), dtype=np.int)
            dist_to_goal = np.zeros((HEIGHT, WIDTH), dtype=np.int)

            for n in range(len(self.passage_positions[s][0])):

                y = self.passage_positions[s][0][n]
                x = self.passage_positions[s][1][n]

                if not (y == self.goal_pos[s][0] and x == self.goal_pos[s][1]):

                    grid = Grid(matrix=(1 - self.grid[s]))
                    start = grid.node(x, y)
                    end = grid.node(self.goal_pos[s][1], self.goal_pos[s][0])

                    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
                    path, runs = finder.find_path(start, end, grid)

                    dist_to_goal[y, x] = len(path) - 1

                    dif_h = path[1][1] - path[0][1]
                    dif_w = path[1][0] - path[0][0]

                    # Down, Up, Right, Left
                    if dif_w != 0:
                        if dif_w > 0:
                            optimal_actions[y, x] = 2
                        elif dif_w < 0:
                            optimal_actions[y, x] = 3
                    elif dif_h != 0:
                        if dif_h > 0:
                            optimal_actions[y, x] = 0
                        elif dif_h < 0:
                            optimal_actions[y, x] = 1

            self.optimal_actions.append(optimal_actions)
            self.dist_to_goal.append(dist_to_goal)


        self.action_quality = []

        self.action_quality_by_state_id = np.full((self.n_states, 4), 0, dtype=np.int)

        for s in range(N_STAGES):
            dist_to_goal = self.dist_to_goal[s]
            grid = self.grid[s]

            action_quality = np.full((HEIGHT, WIDTH, 4), 0, dtype=np.int)

            for n in range(len(self.passage_positions[s][0])):
                y = self.passage_positions[s][0][n]
                x = self.passage_positions[s][1][n]

                if not (y == self.goal_pos[s][0] and x == self.goal_pos[s][1]):

                    state_id = self.state_id_dict[(s, y, x)]

                    for action in range(4):
                        if action == 0:  # Down
                            y_next = y + 1
                            x_next  = x
                        elif action == 1:  # Up
                            y_next = y - 1
                            x_next = x
                        elif action == 2:  # Right
                            y_next = y
                            x_next = x + 1
                        elif action == 3:  # Left
                            y_next = y
                            x_next = x - 1

                        if y_next < 0 or y_next >= HEIGHT or x_next < 0 or x_next >= WIDTH or \
                                grid[y_next, x_next] == 1:
                            action_quality[y, x, action] = 1
                            self.action_quality_by_state_id[state_id, action] = 1
                        else:
                            diff = dist_to_goal[y_next, x_next] - dist_to_goal[y, x]
                            if diff == 1:
                                action_quality[y, x, action] = 2
                                self.action_quality_by_state_id[state_id, action] = 2
                            elif diff == -1:
                                action_quality[y, x, action] = 3
                                self.action_quality_by_state_id[state_id, action] = 3

            self.action_quality.append(action_quality)

        self.random = np.random.RandomState(seed)
        self.state = None

    # ------------------------------------------------------------------------------------------------------------------

    def reset(self):
        self.state = State()
        self.state.t = 0
        self.state.stage = 0
        self.state.done = False
        self.state.timeout = False
        self.state.agent_pos = list(self.agent_pos[0])
        self.state.goal_pos = list(self.goal_pos[0])
        return self.state_to_obs(self.state)

    # ------------------------------------------------------------------------------------------------------------------

    def transitive(self, state, action):

        slipped = self.random.random_sample() < self.slipping_prob
        agent_pos_prev = (state.agent_pos[0], state.agent_pos[1])

        if slipped:
            p = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            p[action] = 0.0
            action = self.random.choice(4, 1, p=p)[0]

        if action == 0:  # Down
            state.agent_pos[0] += 1
        elif action == 1:  # Up
            state.agent_pos[0] -= 1
        elif action == 2:  # Right
            state.agent_pos[1] += 1
        elif action == 3:  # Left
            state.agent_pos[1] -= 1

        reward = 0.0

        if state.agent_pos[0] < 0 or state.agent_pos[0] >= HEIGHT or \
                state.agent_pos[1] < 0 or state.agent_pos[1] >= WIDTH or \
                self.grid[state.stage][state.agent_pos[0], state.agent_pos[1]] == 1:
            if slipped:
                state.agent_pos[0] = agent_pos_prev[0]
                state.agent_pos[1] = agent_pos_prev[1]
            else:
                if self.easy_mode:
                    state.agent_pos[0] = agent_pos_prev[0]
                    state.agent_pos[1] = agent_pos_prev[1]
                else:
                    state.done = True

        elif state.agent_pos[0] == state.goal_pos[0] and state.agent_pos[1] == state.goal_pos[1]:
            reward = 1.0
            if state.stage == (N_STAGES - 1):
                state.done = True
            else:
                state.stage += 1
                state.goal_pos = list(self.goal_pos[state.stage])

        state.t += 1

        return reward

    # ------------------------------------------------------------------------------------------------------------------

    def state_to_obs(self, state):
        if self.obs_form == NONSPATIAL:
            if state.done:
                return np.zeros((32,), dtype=np.float32)
            obs = self.base_nonspatial_obs[state.stage].copy()
            obs[0], obs[1] = (state.agent_pos[0] * self.norm_factor) - 1, (state.agent_pos[1] * self.norm_factor) - 1
            return obs
        elif self.obs_form == SPATIAL:
            if state.done:
                return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            obs = self.base_obs[state.stage].copy()
            obs[state.agent_pos[0], state.agent_pos[1], 0] = 1
            return obs

    # ------------------------------------------------------------------------------------------------------------------

    def generate_obs(self, stage, agent_pos):
        if self.obs_form == NONSPATIAL:
            obs = self.base_nonspatial_obs[stage].copy()
            obs[0], obs[1] = (agent_pos[0] * self.norm_factor) - 1, (agent_pos[1] * self.norm_factor) - 1
            return obs
        elif self.obs_form == SPATIAL:
            obs = self.base_obs[stage].copy()
            obs[agent_pos[0], agent_pos[1], 0] = 1
            return obs

    # ------------------------------------------------------------------------------------------------------------------

    def optimal_action(self, state_in=None):
        state = self.state if state_in is None else state_in
        return self.optimal_actions[state.stage][state.agent_pos[0]][state.agent_pos[1]]

    # ------------------------------------------------------------------------------------------------------------------

    def render(self, state_in=None):
        state = self.state if state_in is None else state_in
        obs_image = self.base_obs_image[state.stage].copy()
        if 0 <= state.agent_pos[0] < HEIGHT and \
                0 <= state.agent_pos[1] < WIDTH:
            obs_image[state.agent_pos[0], state.agent_pos[1], :] = self.agent_color
        obs_image = cv2.resize(obs_image, (int(HEIGHT * 30), int(WIDTH * 30)),
                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return obs_image

    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action):
        reward = self.transitive(self.state, action)
        return self.state_to_obs(self.state), reward, self.state.done

    # ------------------------------------------------------------------------------------------------------------------

    def get_state(self):
        return self.state.stage, self.state.agent_pos

    # ------------------------------------------------------------------------------------------------------------------

    def get_state_id(self, state_in=None):
        state = self.state if state_in is None else state_in
        return self.state_id_dict[(state.stage, state.agent_pos[0], state.agent_pos[1])]

    # ------------------------------------------------------------------------------------------------------------------

    def get_transition_id(self, action, state_in=None):
        state = self.state if state_in is None else state_in
        return self.transition_id_dict[(state.stage, state.agent_pos[0], state.agent_pos[1], action)]

    # ------------------------------------------------------------------------------------------------------------------

    # (Re)sets the random state of the environment, useful to reproduce the same level sequences for evaluation purposes
    def set_random_state(self, seed):
        self.random = np.random.RandomState(seed)
