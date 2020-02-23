class Game:
    def __init__(self, initial_game_status, solver):
        self.solved = False
        self.tries = 0
        pass

    def generate_report(self):
        if not self.solved:
            raise ValueError("The game needs to be solved before generating a report...")
        pass

    def solve(self):
        pass