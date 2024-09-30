import random

class Block:
    def __init__(self, shape, name):
        """
        shape: List of tuples representing relative positions of the block cells
        name: A unique identifier for the block
        """
        self.shape = shape  # e.g., [(0,0), (1,0), (0,1)]
        self.name = name    # e.g., "Block1"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Block) and self.name == other.name

    @staticmethod
    def get_all_shapes():
        """
        Returns a list of all predefined block shapes.
        """
        predefined_shapes = [
            ([ (0,0) ], "Block1"),  # Single cell
            ([ (0,0), (1,0) ], "Block2"),  # Vertical 2
            ([ (0,0), (0,1) ], "Block3"),  # Horizontal 2
            ([ (0,0), (1,0), (0,1) ], "Block4"),  # L-shape
            ([ (0,0), (1,0), (2,0) ], "Block5"),  # Vertical 3
            ([ (0,0), (0,1), (0,2) ], "Block6"),  # Horizontal 3
            ([ (0,0), (1,0), (1,1) ], "Block7"),  # Another L-shape
            ([ (0,0), (1,0), (0,1), (1,1) ], "Block8"),  # Square
            ([ (0,0), (1,0), (2,0), (1,1) ], "Block9"),  # T-shape
            ([ (0,1), (1,0), (1,1), (1,2), (2,1) ], "Block10"),  # Plus shape
            # 3x3 Block 추가
            ([
                (0,0), (0,1), (0,2),
                (1,0), (1,1), (1,2),
                (2,0), (2,1), (2,2)
            ], "Block11")  # 3x3 Square
        ]
        return [Block(shape, name) for shape, name in predefined_shapes]

    @staticmethod
    def generate_random_block():
        """
        Randomly selects and returns a block from predefined shapes.
        """
        return random.choice(Block.get_all_shapes())

    def get_cells(self, top_left_x, top_left_y):
        """
        Returns the absolute positions of the block cells on the board
        """
        return [(top_left_x + x, top_left_y + y) for x, y in self.shape]
