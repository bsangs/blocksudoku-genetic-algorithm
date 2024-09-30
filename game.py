# game.py
from blocks import Block
from score import ScoreManager
import random
import threading

class Game:
    def __init__(self, board_size=8, strategy=None):
        self.board_size = board_size
        self.board = [[None for _ in range(board_size)] for _ in range(board_size)]
        self.score_manager = ScoreManager()
        self.available_blocks = []
        self._initialize_available_blocks()
        self.current_blocks = self._draw_blocks(3)
        self.game_over = False
        self.colors = {}  # Color per block
        self.strategy = strategy  # AI Strategy (Individual)
        self.lock = threading.RLock()

    def _initialize_available_blocks(self):
        """
        Initializes and shuffles the available_blocks pool.
        """
        self.available_blocks = [block for block in Block.get_all_shapes()]
        random.shuffle(self.available_blocks)

    def _draw_blocks(self, count):
        blocks = []
        for _ in range(count):
            if not self.available_blocks:
                self._initialize_available_blocks()  # Reinitialize if empty
            if self.available_blocks:
                blocks.append(self.available_blocks.pop())
            else:
                break  # In case no blocks are available after reinitialization
        return blocks

    def replenish_blocks(self):
        """
        Adds new blocks to maintain 3 current blocks.
        """
        needed = 3 - len(self.current_blocks)
        self.current_blocks.extend(self._draw_blocks(needed))

    def select_block(self, index):
        if 0 <= index < len(self.current_blocks):
            return self.current_blocks[index]
        return None

    def place_block(self, block, top_left_x, top_left_y):
        with self.lock:
            if self.game_over:
                return False, 0  # If game over

            cells = block.get_cells(top_left_x, top_left_y)
            # Validate placement
            for x, y in cells:
                if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
                    return False, 0
                if self.board[x][y] is not None:
                    return False, 0
            # Place the block
            for x, y in cells:
                self.board[x][y] = block
            # Update score
            self.score_manager.add_placement_score(len(cells))
            # Remove used block
            self.current_blocks.remove(block)
            # Replenish blocks if empty
            if len(self.current_blocks) == 0:
                self.replenish_blocks()
            # Check and remove complete lines
            removed_cells = self.check_and_remove_complete_lines()
            if removed_cells > 0:
                self.score_manager.add_removal_score(removed_cells)
            # Change colors
            self.change_all_block_colors()
            # Check game over
            self.check_game_over()
            return True, removed_cells

    def change_all_block_colors(self):
        """
        Changes colors for all placed blocks
        """
        unique_blocks = set(block for row in self.board for block in row if block is not None)
        colors = ["red", "green", "blue", "orange", "purple", "cyan", "yellow", "magenta", "pink", "lime"]
        random.shuffle(colors)
        self.colors = {}
        for idx, block in enumerate(unique_blocks):
            self.colors[block] = colors[idx % len(colors)]

    def check_and_remove_complete_lines(self):
        """
        Checks for complete lines and removes them.
        Returns the number of cells removed.
        """
        removed_cells = 0
        for x in range(self.board_size):
            if all(self.board[x][y] is not None for y in range(self.board_size)):
                # Remove the entire line
                for y in range(self.board_size):
                    self.board[x][y] = None
                    removed_cells += 1
        return removed_cells

    def check_game_over(self):
        """
        Checks if no more moves are possible with current blocks.
        """
        for block in self.current_blocks:
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if self.can_place(block, x, y):
                        return  # Still possible moves
        self.game_over = True

    def can_place(self, block, top_left_x, top_left_y):
        """
        Checks if the block can be placed at the specified position.
        """
        cells = block.get_cells(top_left_x, top_left_y)
        for x, y in cells:
            if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
                return False
            if self.board[x][y] is not None:
                return False
        return True

    def get_valid_moves(self):
        """
        Returns a list of all possible valid moves in the current game state.
        Each move is represented as a tuple: (block_index, x, y)
        """
        valid_moves = []
        for block_index, block in enumerate(self.current_blocks):
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if self.can_place(block, x, y):
                        valid_moves.append((block_index, x, y))
        return valid_moves

    def get_score(self):
        return self.score_manager.get_score()

    def get_board(self):
        return self.board

    def get_current_blocks(self):
        return self.current_blocks

    def get_block_color(self, block):
        return self.colors.get(block, "black")

    def make_move_ai(self):
        """
        AI makes a move according to the current strategy.
        """
        with self.lock:
            if self.game_over or not self.current_blocks:
                return False, 0

            max_attempts = 10  # Limit to prevent infinite loops
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                # Strategy defines how to pick block and position
                action = self.strategy.decide_action(self)  # type: ignore
                if not action:
                    # No action possible, set game over
                    self.game_over = True
                    return False, 0
                block_index, x, y = action
                block = self.select_block(block_index)
                if not block:
                    continue  # Try again
                if not self.can_place(block, x, y):
                    continue  # Try again
                placed, removed = self.place_block(block, x, y)
                if placed:
                    return placed, removed
            # After max attempts, set game over
            self.game_over = True
            return False, 0

    def reset(self):
        """
        Resets the game to initial state.
        """
        with self.lock:
            self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
            self.score_manager = ScoreManager()
            self.available_blocks = []
            self._initialize_available_blocks()
            self.current_blocks = self._draw_blocks(3)
            self.game_over = False
            self.colors = {}
            self.change_all_block_colors()
