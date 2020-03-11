from poseset import PoseSet

class PoseQueue:
    def __init__(self, len = 100):
        self.Size = len
        self.Queue = []



    def push(self, pose):
        self.Queue.insert(0, PoseSet(pose))
        if len(self.Queue) > self.Size:
            self.Queue.pop()

    def all(self):
        return self.Queue

    def MaxY(self):
        max_shoulder = 0
        max_hip = 0
        max_knee = 0

        for pos in self.all():
            (shoulder, hip, knee) = pos.RelativeY()
            max_shoulder = max(shoulder, max_shoulder)
            max_hip = max(hip, max_hip)
            max_knee = max(knee, max_knee)
        
        return (max_shoulder, max_hip, max_knee)


    def MinY(self):
        min_shoulder = 0
        min_hip = 0
        min_knee = 0

        for pos in self.all():
            (shoulder, hip, knee) = pos.RelativeY()
            min_shoulder = min(shoulder, min_shoulder)
            min_hip = min(hip, min_hip)
            min_knee = min(knee, min_knee)
        
        return (min_shoulder, min_hip, min_knee)