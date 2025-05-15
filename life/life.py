"""The Life module."""
import numpy as np
from matplotlib import pyplot
from scipy.signal import convolve2d

glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])

blinker = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]]
)

glider_gun = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0]
])


class Game:
    """Game class."""

    def __init__(self, size):
        self.board = np.zeros((size, size))

    def play(self):
        """Start the game."""
        print("Playing life. Press ctrl + c to stop.")
        pyplot.ion()
        while True:
            self.move()
            self.show()
            pyplot.pause(0.0000005)

    def move(self):
        """Move within the game."""
        stencil = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbour_count = convolve2d(self.board, stencil, mode='same')

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                self.board[i, j] = 1 if (neighbour_count[i, j] == 3
                                         or (neighbour_count[i, j] == 2
                                         and self.board[i, j])) else 0

    def __setitem__(self, key, value):
        """Set the item value."""
        self.board[key] = value

    def show(self):
        """Show the figure plotted."""
        pyplot.clf()
        pyplot.matshow(self.board, fignum=0, cmap='binary')
        pyplot.show()

    def insert(self, pattern, coords):
        """Insert the pattern given."""
        new_grid = self.board
        a, b = 0, 0
        pattern_rows, pattern_cols = pattern.grid.shape
        centre_row, centre_col = coords
        row_start = centre_row - ((pattern_rows - 1) // 2)
        col_start = centre_col - ((pattern_cols - 1) // 2)
        for i in range(row_start, row_start + pattern_rows):
            b = 0
            for j in range(col_start, col_start + pattern_cols):
                new_grid[i, j] = pattern.grid[a, b]
                b += 1
            a += 1
        self.board = new_grid


class Pattern:
    """The pattern class."""

    def __init__(self, pattern):
        self.grid = pattern

    def flip_vertical(self):
        """Perform a vertical flip."""
        return Pattern(self.grid[::-1])

    def flip_horizontal(self):
        """Perform a horizontal flip."""
        return Pattern(self.grid[:, ::-1])

    def flip_diag(self):
        """Perform a diagonal flip (transpose)."""
        return Pattern(self.grid.T)

    def rotate(self, n):
        """Rotate the grid n times."""
        n = n % 4
        if not n:
            return Pattern(self.grid)
        if n == 1:
            first_step = self.flip_diag()
            return first_step.flip_vertical()
        if n == 2:
            first_step = self.flip_horizontal()
            return first_step.flip_vertical()
        if n == 3:
            first_step = self.flip_diag()
            return first_step.flip_horizontal()
