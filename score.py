class ScoreManager:
    def __init__(self):
        self.score = 0

    def add_placement_score(self, num_blocks):
        self.score += num_blocks

    def add_removal_score(self, num_blocks):
        self.score += num_blocks

    def get_score(self):
        return self.score
