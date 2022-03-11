import numpy as np
import cv2
from gym import spaces
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# ======================================================================================================================

class Constants():
    def __init__(self):
        self.AGENT_COLOUR = np.asarray([255, 255, 255])
        self.GOAL_COLOUR = np.asarray([153, 255, 153])
        self.GROUND_COLOUR = np.asarray([155, 155, 155])
        self.PIT_COLOUR = np.asarray([0, 0, 0])
        self.RENDER_WIDTH_MULTIPLIER = 30
        self.RENDER_HEIGHT_MULTIPLIER = 30

CONSTANT = Constants()

# ======================================================================================================================

class State(object):
    def __init__(self):
        self.t = None
        self.stage = 0
        self.done = None
        self.timeout = None
        self.agent_dead = False
        self.agent_pos_prev = [0, 0]
        self.agent_pos = [0, 0]
        self.goal_pos = [0, 0]

# ======================================================================================================================

class Environment(object):
    def __init__(self,
                 seed=0,                # In-game random event(s), e.g. slipping, seed
                 slipping_prob=0.2,     # Slipping probability
                 obs_form=0,            # 0: one-hot encoding, 1: normalised coordinates, 2: binary grid stack
                 easy_mode=False,       # False: pits kill the agent, True: pits don't kill the agent
                 level_structure=None,  # level structure as numpy array, 0: ground, 1: pit, 2: agent, 3: goal
                 n_stages=1,            # number of stages in level
                 ):

        if level_structure is None:
            self.level_structure = np.array([
                [2, 0, 0, 0, 1, 0, 0, 0, 3],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0]
            ],
                dtype=int)
        else:
            self.level_structure = level_structure

        self.height = np.shape(self.level_structure)[0]
        self.width = np.shape(self.level_structure)[1]

        self.max_dim = self.height - 1  # square assumption
        self.norm_factor = 2 / self.max_dim   # self.norm_factor = 2 / self.max_dim

        self.obs_form = obs_form
        self.easy_mode = easy_mode
        self.n_stages = n_stages
        self.slipping_prob = slipping_prob

        self.n_actions = 4  # Down, Up, Right, Left

        self.grid = []
        self.agent_pos = []
        self.goal_pos = []
        self.pit_positions = []
        self.passage_positions = []

        self.n_agents = 1
        self.n_goals = 1
        self.n_pits = None
        self.n_passages = None

        # --------------------------------------------------------------------------------------------------------------
        # Construct the level

        for stage in range(self.n_stages):
            grid = np.copy(self.level_structure)

            agent_ind = np.where(grid == 2)
            agent_pos = [agent_ind[0][0], agent_ind[1][0]]
            grid[agent_pos[0], agent_pos[1]] = 0
            self.agent_pos.append(agent_pos)

            goal_ind = np.where(grid == 3)
            goal_pos = [goal_ind[0][0], goal_ind[1][0]]
            self.goal_pos.append(goal_pos)

            self.pit_positions.append(np.where(grid == 1))
            self.passage_positions.append(np.where(grid == 0))

            grid[goal_pos[0], goal_pos[1]] = 0
            self.grid.append(grid)

        print(self.passage_positions[0], len(self.passage_positions[0][0]))

        self.n_pits = len(self.pit_positions[0][0])
        self.n_passages = len(self.passage_positions[0][0])

        # --------------------------------------------------------------------------------------------------------------

        self.state_id_dict = {}
        self.agent_pos_dict = {}
        self.transition_id_dict = {}

        state_id = 0
        transition_id = 0

        for s in range(self.n_stages):
            for n in range(len(self.passage_positions[s][0])):
                if not (self.passage_positions[s][0][n] == self.goal_pos[s][0] and
                        self.passage_positions[s][1][n] == self.goal_pos[s][1]):
                    self.state_id_dict[(s, self.passage_positions[s][0][n], self.passage_positions[s][1][n])] = state_id
                    self.agent_pos_dict[state_id] = (s, self.passage_positions[s][0][n], self.passage_positions[s][1][n])
                    for i_action in range(self.n_actions):
                        self.transition_id_dict[
                            (s, self.passage_positions[s][0][n], self.passage_positions[s][1][n], i_action)] = \
                            transition_id
                        transition_id += 1
                    state_id += 1

        self.n_states = len(self.state_id_dict)  # stage, x, y
        self.n_transitions = len(self.transition_id_dict)  # stage, x, y, action

        # --------------------------------------------------------------------------------------------------------------
        # Construct observation and action spaces

        if self.obs_form == 0:  # One-hot encoding
            self.obs_shape = (self.n_states,)
            self.obs_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        elif self.obs_form == 1:  # Normalised coordinates
            self.obs_shape = (2*(self.n_agents + self.n_goals + self.n_pits),)
            self.obs_space = spaces.Box(low=-1, high=1, shape=self.obs_shape, dtype=np.float32)

        elif self.obs_form == 2:  # Binary grid stack
            self.obs_shape = (self.height, self.width, 3)
            self.obs_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.uint8)

        self.action_space = spaces.Discrete(self.n_actions)

        # --------------------------------------------------------------------------------------------------------------
        # Construct base observations

        self.base_obs = []
        self.blank_obs = None

        if self.obs_form == 0:  # One-hot encoding
            for s in range(self.n_stages):
                base_obs = np.zeros(self.obs_shape, dtype=np.float32)
                self.base_obs.append(base_obs)

            self.blank_obs = np.zeros(self.obs_shape, dtype=np.float32)

        elif self.obs_form == 1:  # Normalised coordinates
            for s in range(self.n_stages):
                pit_positions = self.pit_positions[s]
                base_obs = np.zeros(self.obs_shape, dtype=np.float32)
                for n in range(len(pit_positions[0])):
                    pit_index = 4 + (2 * n)
                    base_obs[pit_index], base_obs[pit_index + 1] = pit_positions[0][n], pit_positions[1][n]
                base_obs[0], base_obs[1] = self.agent_pos[s][0], self.agent_pos[s][1]
                base_obs[2], base_obs[3] = self.goal_pos[s][0], self.goal_pos[s][1]
                base_obs *= self.norm_factor
                base_obs -= 1
                self.base_obs.append(base_obs)

            self.blank_obs = np.zeros(self.obs_shape, dtype=np.float32)

        elif self.obs_form == 2:  # Binary grid stack
            for s in range(self.n_stages):
                base_obs = np.zeros(self.obs_shape, dtype=np.uint8)
                base_obs[self.goal_pos[s][0], self.goal_pos[s][1], 1] = 1
                for n in range(len(self.pit_positions[s][0])):
                    base_obs[self.pit_positions[s][0][n], self.pit_positions[s][1][n], 2] = 1
                self.base_obs.append(base_obs)

            self.blank_obs = np.zeros(self.obs_shape, dtype=np.uint8)

        # Base observation visual
        self.base_obs_image = []
        for s in range(self.n_stages):
            base_obs_image = np.full((self.height, self.width, 3), CONSTANT.GROUND_COLOUR, dtype=np.uint8)
            base_obs_image[self.goal_pos[s][0], self.goal_pos[s][1], :] = CONSTANT.GOAL_COLOUR
            for n in range(len(self.pit_positions[s][0])):
                base_obs_image[self.pit_positions[s][0][n], self.pit_positions[s][1][n], :] = CONSTANT.PIT_COLOUR
            self.base_obs_image.append(base_obs_image)

        # --------------------------------------------------------------------------------------------------------------
        # Pre-determine the optimal actions and distances to goal in every state

        self.optimal_actions = []
        self.dist_to_goal = []

        for s in range(self.n_stages):
            optimal_actions = np.zeros((self.height, self.width), dtype=np.int)
            dist_to_goal = np.zeros((self.height, self.width), dtype=np.int)

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
        self.action_quality_by_state_id = np.full((self.n_states, self.n_actions), 0, dtype=np.int)

        for s in range(self.n_stages):
            dist_to_goal = self.dist_to_goal[s]
            grid = self.grid[s]

            action_quality = np.full((self.height, self.width, self.n_actions), 0, dtype=np.int)

            for n in range(len(self.passage_positions[s][0])):
                y = self.passage_positions[s][0][n]
                x = self.passage_positions[s][1][n]

                if not (y == self.goal_pos[s][0] and x == self.goal_pos[s][1]):

                    state_id = self.state_id_dict[(s, y, x)]

                    for action in range(self.n_actions):
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

                        if y_next < 0 or y_next >= self.height or x_next < 0 or x_next >= self.width or \
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

        # --------------------------------------------------------------------------------------------------------------
        # Set the random state

        self.random = np.random.RandomState(seed)
        self.state = None

    # ==================================================================================================================

    def reset(self):
        self.state = State()
        self.state.t = 0
        self.state.stage = 0
        self.state.done = False
        self.state.timeout = False
        self.state.agent_dead = False
        self.state.agent_pos_prev = list(self.agent_pos[0])
        self.state.agent_pos = list(self.agent_pos[0])
        self.state.goal_pos = list(self.goal_pos[0])
        return self.state_to_obs(self.state)

    # ==================================================================================================================

    def transitive(self, state, action):

        slipped = self.random.random_sample() < self.slipping_prob
        state.agent_pos_prev[0], state.agent_pos_prev[1] = state.agent_pos[0], state.agent_pos[1]

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

        if state.agent_pos[0] < 0 or state.agent_pos[0] >= self.height or \
                state.agent_pos[1] < 0 or state.agent_pos[1] >= self.width or \
                self.grid[state.stage][state.agent_pos[0], state.agent_pos[1]] == 1:
            if slipped:
                state.agent_pos[0] = state.agent_pos_prev[0]
                state.agent_pos[1] = state.agent_pos_prev[1]
            else:
                if self.easy_mode:
                    state.agent_pos[0] = state.agent_pos_prev[0]
                    state.agent_pos[1] = state.agent_pos_prev[1]
                else:
                    state.agent_dead = True
                    state.done = True

        elif state.agent_pos[0] == state.goal_pos[0] and state.agent_pos[1] == state.goal_pos[1]:
            reward = 1.0
            if state.stage == (self.n_stages - 1):
                state.done = True
            else:
                state.stage += 1
                state.goal_pos = list(self.goal_pos[state.stage])

        state.t += 1

        return reward

    # ==================================================================================================================
    # Generate observation from the current game state

    def state_to_obs(self, state):
        if state.done:
            return self.blank_obs
        return self.generate_obs(state.stage, state.agent_pos)

    # ==================================================================================================================
    # Generate observation for a given state and an agent position

    def generate_obs(self, stage, agent_pos):
        obs = self.base_obs[stage].copy()

        if self.obs_form == 0:  # One-hot encoding
            state_id = self.get_state_id_from_position(stage, agent_pos)
            obs[state_id] = 1.0

        elif self.obs_form == 1:  # Normalised coordinates
            obs[0], obs[1] = (agent_pos[0] * self.norm_factor) - 1, (agent_pos[1] * self.norm_factor) - 1
            return obs

        elif self.obs_form == 2:  # Binary grid stack
            obs[agent_pos[0], agent_pos[1], 0] = 1

        return obs

    # ==================================================================================================================

    def generate_obs_from_state(self, state_id):
        stage, agent_pos_x, agent_pos_y = self.agent_pos_dict[state_id]
        return self.generate_obs(stage, (agent_pos_x, agent_pos_y))

    # ==================================================================================================================

    def optimal_action(self, state_in=None):
        state = self.state if state_in is None else state_in
        return self.optimal_actions[state.stage][state.agent_pos[0]][state.agent_pos[1]]

    # ==================================================================================================================

    def render(self, state_in=None):
        state = self.state if state_in is None else state_in
        obs_image = self.base_obs_image[state.stage].copy()
        if 0 <= state.agent_pos[0] < self.height and \
                0 <= state.agent_pos[1] < self.width:
            obs_image[state.agent_pos[0], state.agent_pos[1], :] = CONSTANT.AGENT_COLOUR
        obs_image = cv2.resize(obs_image, (int(self.height * CONSTANT.RENDER_HEIGHT_MULTIPLIER),
                                           int(self.width * CONSTANT.RENDER_WIDTH_MULTIPLIER)),
                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return obs_image

    # ==================================================================================================================

    def step(self, action):
        reward = self.transitive(self.state, action)
        return self.state_to_obs(self.state), reward, self.state.done

    # ==================================================================================================================

    def get_state(self):
        return self.state.stage, self.state.agent_pos

    # ==================================================================================================================

    def get_state_id(self, state_in=None):
        state = self.state if state_in is None else state_in
        if state.done:
            return None
        else:
            return self.state_id_dict[(state.stage, state.agent_pos[0], state.agent_pos[1])]

    # ==================================================================================================================

    def get_state_id_from_position(self, stage, agent_pos):
        return self.state_id_dict[(stage, agent_pos[0], agent_pos[1])]

    # ==================================================================================================================

    def get_transition_id(self, action, state_in=None):
        state = self.state if state_in is None else state_in
        return self.transition_id_dict[(state.stage, state.agent_pos[0], state.agent_pos[1], action)]

    # ==================================================================================================================

    def get_state_dist_to_goal(self, state_in=None):
        state = self.state if state_in is None else state_in
        return self.dist_to_goal[state.stage][state.agent_pos[0], state.agent_pos[1]]

    # ==================================================================================================================

    def get_pos_dist_to_goal(self, stage, pos_y, pos_x):
        return self.dist_to_goal[stage][pos_y, pos_x]

    # ==================================================================================================================

    # (Re)sets the random state of the environment, useful to reproduce the same level sequences for evaluation purposes
    def set_random_state(self, seed):
        self.random = np.random.RandomState(seed)
