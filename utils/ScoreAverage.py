class ScoreAverage:
    def __init__(self):
        self.sum = 0
        self.history = []
        self.total = 0

    def update(self, value):
        self.sum += value
        self.history.append(value)
        self.total += len(self.history)

    @property
    def mean(self):
        return self.sum / self.total