import numpy as np

class Grid:

    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size)).astype(int)
        self.score = 0
        self.moves = 0
        self.populate(n_tiles=2)

    def shift(self, matrix):
        new_matrix = np.zeros((self.size, self.size)) 
        # iterate rows and tiles
        for i, row in enumerate(matrix): 
            fill = self.size - 1
            for tile in row: 
                # make non empty tiles right most (fill)
                if tile: 
                    new_matrix[i][fill] = tile 
                    fill -= 1

        return new_matrix

    def combine(self, matrix):

        for i in range(self.size): # iterate each row
            for j in range(self.size - 1): # iterate each tile
                if matrix[i][j] == matrix[i][j + 1] and matrix[i][j]: # matching tiles and not empty (combine)
                    matrix[i][j] *= 2
                    matrix[i][j + 1] = 0
                    self.score += matrix[i][j] # update score

        return matrix


    def move(self, direction):

        matrix = self.grid

        if direction == 'left':
            matrix = np.flip(matrix, 1) # reverse the matrix to get movements to the right in respect to the left
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = self.combine(matrix) # combine like tiles
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = np.flip(matrix, 1) # reverse the matrix back to the original orientation
        
        elif direction == 'right':
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = self.combine(matrix) # combine like tiles
            matrix = self.shift(matrix) # push tiles to the rightmost

        elif direction == 'up':
            matrix = np.rot90(matrix, 1) # rotate the matrix 90 degrees counter-clockwise to get movements to the right in respect to down
            matrix = np.flip(matrix, 1) # reverse the matrix to get movements to the right in respect to the up (left)
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = self.combine(matrix) # combine like tiles
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = np.flip(matrix, 1) # reverse the matrix back to the original orientation
            matrix = np.rot90(matrix, 3) # rotate the matrix 270 degrees counter-clockwise to get back to the original orientation   
        
        elif direction == 'down':
            matrix = np.rot90(matrix, 1) # rotate the matrix 90 degrees counter-clockwise to get movements to the right in respect to down
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = self.combine(matrix) # combine like tiles
            matrix = self.shift(matrix) # push tiles to the rightmost
            matrix = np.rot90(matrix, 3) # rotate the matrix 270 degrees counter-clockwise to get back to the original orientation
        
        # only modify the matrix if an actual move was made instead of the function being called
        if not np.array_equal(matrix, self.grid):
            self.grid = matrix
            self.populate() # populate if the move was made
            self.moves += 1
    
    # indicates if a horizontal move can be made
    def move_horizontal(self):
        # iterate rows in columns
        for i in range(self.size):
            for j in range(self.size - 1):
                # if adjacent tiel along the horizontal axis is the same
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True
        return False

    # indicates if a vertical move can be made
    def move_vertical(self):
        # iterate each rows and columns
        for i in range(self.size - 1):
            for j in range(self.size):
                # if adjacent tile along the vertical axis is the same
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True
        return False

    def game_over(self):
        if np.any(self.grid.astype(int) == 0): # if there's any open space the game is still in place
            return False

        # if any movement in all directions is the same as the original grid, the game is over
        return not self.move_horizontal() and not self.move_vertical()
        

    # adds a random tiles to grid, or initializes grid with starting tiles
    def populate(self, n_tiles=1):

        for _ in range(n_tiles):
            row, col = np.where(self.grid == 0) # empty spaces
            index = np.random.randint(0, len(row)) # random spot

            # starting tiles (always 2)
            if n_tiles == 2:
                tile = 2
            # new tiles after start (2 or 4)
            else:
                tile = np.random.choice([2, 4])
            self.grid[row[index], col[index]] = tile # place tile
        

    # prints the grid
    def __str__(self):
        return str(self.grid.astype(int))


if __name__ == "__main__":
    grid = Grid()
    print(grid)