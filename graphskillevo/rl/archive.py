class Archive:
    """Pinned initial reference plus FIFO history, never score-ranked."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def add(self, skill):
        if any(x.id == skill.id for x in self.items):
            return
        if len(self.items) >= self.capacity:
            if self.capacity == 1:
                return
            self.items.pop(1)
        self.items.append(skill)

    def slots(self):
        return self.items + [None] * (self.capacity - len(self.items))
