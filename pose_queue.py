from poseset import PoseSet

class PoseQueue:
    def __init__(self, len = 100):
        self.Size = len
        self.Queue = []



    def push(self, pose):
        self.Queue.append(PoseSet(pose))
        if len(self.Queue) > self.Size:
            self.Queue.pop(0)

    def all(self):
        return self.Queue