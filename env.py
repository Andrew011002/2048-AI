import stable_baselines3 as sb3
import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from gym import spaces

MAPPINGS = {0: "left", 1: "right", 2: "up", 3: "down"}

# adds a random tiles to grid, or initializes grid with starting tiles
def populate(grid, n_tiles=1):
    for _ in range(n_tiles):
        row, col = np.where(grid == 0) # empty spaces
        index = np.random.randint(0, len(row)) # random spot

        # starting tiles (always 2)
        if n_tiles == 2:
            tile = 2
        # new tiles after start (2 or 4)
        else:
            tile = np.random.choice([2, 4])

        grid[row[index], col[index]] = tile # place tile

    return grid

# moves tiles to rightmost
def shift(matrix, size=4):
        new_matrix = np.zeros((size, size)) 
        # iterate rows and tiles
        for i, row in enumerate(matrix): 
            fill = size - 1
            for tile in row: 
                # make non empty tiles right most (fill)
                if tile: 
                    new_matrix[i][fill] = tile 
                    fill -= 1

        return new_matrix

# conbines like tiles and returns the new grid with the score
def combine(matrix, size=4):
    score = 0
    for i in range(size): # iterate each row
        for j in range(size - 1): # iterate each tile
            if matrix[i][j] == matrix[i][j + 1] and matrix[i][j]: # matching tiles and not empty (combine)
                matrix[i][j] *= 2
                matrix[i][j + 1] = 0
                score += matrix[i][j] # update score

    return matrix, score

# moves tiles in grid based on direction
def move(grid, direction, size=4):

    matrix = grid

    if direction == "left":
        matrix = np.flip(matrix, 1) # reverse the matrix to get movements to the right in respect to the left
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix, score = combine(matrix, size=size) # combine like tiles
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix = np.flip(matrix, 1) # reverse the matrix back to the original orientation
    
    elif direction == "right":
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix, score = combine(matrix, size=size) # combine like tiles
        matrix = shift(matrix, size=size) # push tiles to the rightmost

    elif direction == "up":
        matrix = np.rot90(matrix, 1) # rotate the matrix 90 degrees counter-clockwise to get movements to the right in respect to down
        matrix = np.flip(matrix, 1) # reverse the matrix to get movements to the right in respect to the up (left)
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix, score = combine(matrix, size=size) # combine like tiles
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix = np.flip(matrix, 1) # reverse the matrix back to the original orientation
        matrix = np.rot90(matrix, 3) # rotate the matrix 270 degrees counter-clockwise to get back to the original orientation   
    
    elif direction == "down":
        matrix = np.rot90(matrix, 1) # rotate the matrix 90 degrees counter-clockwise to get movements to the right in respect to down
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix, score = combine(matrix, size=size) # combine like tiles
        matrix = shift(matrix, size=size) # push tiles to the rightmost
        matrix = np.rot90(matrix, 3) # rotate the matrix 270 degrees counter-clockwise to get back to the original orientation
    
    # only modify the matrix if an actual move was made instead of the function being called
    if not np.array_equal(matrix, grid):
        grid = populate(matrix)
        return grid, score, 1 # moved

    return grid, score, 0 # did not move

# indicates if a horizontal move can be made
def move_horizontal(grid, size=4):
    # iterate rows in columns
    for i in range(size):
        for j in range(size - 1):
            # if adjacent tiel along the horizontal axis is the same
            if grid[i][j] == grid[i][j + 1]:
                return True
    return False

# indicates if a vertical move can be made
def move_vertical(grid, size=4):
    # iterate each rows and columns
    for i in range(size - 1):
        for j in range(size):
            # if adjacent tile along the vertical axis is the same
            if grid[i][j] == grid[i + 1][j]:
                return True
    return False

# checks if game is over based on if moves can be made or 2048 tile reached
def game_over(grid, size=4):
    if np.any(grid == 0): # if there's any open space the game is still in place
        return False

    if np.any(grid == 2048): # player won game is over
        return True

    # if any movement in all directions is the same as the original grid, the game is over
    return not move_horizontal(grid, size=size) and not move_vertical(grid, size=size)

# finds the goal space (space where the largest tile is in 1/4 corners)
def find_goal_space(grid, size=4):
    max_tile = np.max(grid) # determine max tile
    # tile in top left quadrant
    if max_tile in grid[:size // 2, :size // 2]:
        return 0, 0, "top-left"
    # tile in bottom left quadrant
    elif max_tile in grid[size // 2:, :size // 2]:
        return size - 1, 0, "bot-left" 
    # tile in top right quadrant
    elif max_tile in grid[:size // 2, size // 2:]:
        return 0, size - 1, "top-right" 
    # tile in bottom right quadrant
    elif max_tile in grid[size // 2:, size // 2:]:
        return size - 1, size - 1, "bot-right"

# finds where to slide to based on goal space
def slide_to(goal_row, goal_col, size=4):
    # slide left and up (down next best option)
    if goal_row == goal_col == 0:
        return 0, 2
    #  slide right and up (down next best option)
    if goal_row == 0 and goal_col == size - 1:
        return 1, 2
    # slide left and down (up next best option)
    if goal_row == size - 1 and goal_col == 0:
        return 0, 3
    # slide right and down (up next best option)
    if goal_row == size - 1 and goal_col == size - 1:
        return 1, 3

# indicate valid moves
def find_valid_moves(grid, size=4):
    valid_moves = [-1, -1, -1, -1] # (left, right, up, down)
    # can move left (2nd index refers to moves)
    if move(grid, "left", size=size)[2]:
        valid_moves[0] = 0
    # can move right
    if move(grid, "right", size=size)[2]:
        valid_moves[1] = 1
    # can move up
    if move(grid, "up", size=size)[2]:
        valid_moves[2] = 2
    # can move down
    if move(grid, "down", size=size)[2]:
        valid_moves[3] = 3
    return valid_moves

# determines the actions that give the maximum score
def score_maximizer(x, y, grid, size):
    # define allowable actions
    actions = [x, y, 2 if y == 3 else 3]

    # get scores for allowable actions
    moves = []
    for action in actions:
        score = move(grid, MAPPINGS[action], size=size)[1]
        # scoring more
        if score > 0:
            moves.append((score, action))
        # non-scoring move
        else:
            moves.append((score, -1))
    
    # sort actions by their scores
    moves.sort(key=lambda a: a[0], reverse=True)

    # return the actions where best move is descending
    return [action[1] for action in moves] # shape: (3,)    


class Env2048(gym.Env):

    def __init__(self, size=4):
        super(Env2048, self).__init__()
        self.size = size # for grid size
        self.action_space = spaces.Discrete(4) # amount of actions (left, right, up, & down)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32) # what we will observe

    # how the agent will modify the enviornment
    def step(self, action):

        # Agent making move will handle invalid moves by not changing the grid
        if action == 0:
            self.grid, score, moves = move(self.grid, "left", size=self.size)
        elif action == 1:
            self.grid, score, moves = move(self.grid, "right", size=self.size)
        elif action == 2:
            self.grid, score, moves = move(self.grid, "up", size=self.size)
        elif action == 3:
            self.grid, score, moves = move(self.grid, "down", size=self.size)

        # update game info
        self.states += 1
        self.score += score
        self.moves += moves
        self.grid_sum = np.sum(self.grid)
        self.discount = np.sqrt(np.log(self.states) + 1)

        # base reward for combined tiles and sum towards 2048
        self.reward = self.score - self.prevScore

        # invalid move made
        if self.moves == self.prevMoves:
            # penalty defaults to -100 but is increased for missing to combine (a) tile(s)
            self.reward = -100
        # valid move made
        else:
            # base reward for moving towards 2048 tile
            self.reward += self.grid_sum / 10 
            # get desirable moves
            best_move, med_move, least_move = self.scoring_moves
    
            # best possible move
            if action == best_move:
                self.reward += 200
            # second best possible move
            elif action == med_move:
                self.reward += 150
            # third best possible move
            elif action == least_move:
                self.reward += 100                
            # action was not a "best" scoring move
            else:
                # horizontal move towards target
                if action == self.slide_x:
                    self.reward += 75
                # vertical move towards target
                elif action == self.slide_y:
                    self.reward += 50
                # other vertical move
                elif action > 1:
                    self.reward += 25
                # worst move when could've made an alternate move
                elif np.sum(np.delete(self.valid_moves, self.worst_move)) != -3:
                    self.reward -= 100

        self.reward /= self.discount * self.scale # normalizing reward

        # checking if the game is over
        if game_over(self.grid, size=self.size):
            self.reward = 0 # don't account for points
            self.done = True

        # basic info for debugging
        info = dict(states=self.states, score=self.score, points=self.score-self.prevScore, 
                    moved="yes" if (self.moves - self.prevMoves) else "no", best_move=MAPPINGS.get(self.scoring_moves[0], "NA"), 
                    next_best=MAPPINGS.get(self.scoring_moves[1], "NA"), total=self.grid_sum, target=self.text, 
                    discount= 1 / self.discount)

        # dyanmic changes (updates only after a certain amount of states)
        if self.states % self.frequency == 0:
            self.goal_row, self.goal_col, self.text = find_goal_space(self.grid, size=self.size) # find goal row
            self.slide_x, self.slide_y = slide_to(self.goal_row, self.goal_col, size=self.size) # find best sliding moves

        # update previous info, get observation (updates for next state)
        self.prevScore = self.score
        self.prevMoves = self.moves
        self.valid_moves = find_valid_moves(self.grid, self.size) # set valid moves for next run
        
        self.scoring_moves = score_maximizer(self.slide_x, self.slide_y, self.grid, size=self.size) # find scoring moves for next run (dynamic change)
        self.observation = np.array([self.slide_x, self.slide_y] + self.scoring_moves + self.valid_moves).astype(np.float32) # get obs
        
        return self.observation, self.reward, self.done, info

    # basically inits the enviornment (creates the 2048 grid)
    def reset(self):
        # basic data
        self.done = False
        self.score = 0
        self.prevScore = 0
        self.states = 0
        self.moves = 0
        self.prevMoves = 0
        self.scale = 100

        # dynamic data
        self.frequency = 20

        # instantiate grid
        self.grid = np.zeros((self.size, self.size)).astype(int)
        self.grid = populate(self.grid, n_tiles=2)

        # grid info
        self.valid_moves = find_valid_moves(self.grid, self.size)
        self.grid_sum = np.sum(self.grid)
        self.goal_row, self.goal_col, self.text = find_goal_space(self.grid, size=self.size)
        self.slide_x, self.slide_y = slide_to(self.goal_row, self.goal_col, size=self.size)
        self.worst_move = 1 if self.slide_x == 0 else 0
        self.scoring_moves = score_maximizer(self.slide_x, self.slide_y, self.grid, size=self.size)

        # observation
        self.observation = np.array([self.slide_x, self.slide_y] + self.scoring_moves + self.valid_moves).astype(np.float32) # what the Agent learns
        return self.observation  # reward, done, info can't be included

    # basic print to see how agent does
    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError("Mode not supported")
        
        print(self.grid.astype(int))
        
if __name__ == "__main__":
    env = Env2048() # create evniornment

    check_env(env) # make sure it passes check

    # simulated our env 
    def simulate(env, episodes=100):
        for ep in range(episodes):
            accum_reward = 0
            done = False # reset done every loop
            obs = env.reset() # reset env

            # iteract with env through agent by taking actions from action space
            env.render()
            while not done:
                # sample action
                action = env.action_space.sample()
                print(f"Observation: {obs}")
                # make the action
                obs, reward, done, info = env.step(action)
                accum_reward += reward
                print(f"Action: {'left' if not action else 'right' if action == 1 else 'up' if action == 2 else 'down'}")
                print(f"Reward {reward}")
                print(f"Info: {info}")
                env.render()
            print(f"Net Reward: {accum_reward}")
            
    simulate(env, episodes=1)