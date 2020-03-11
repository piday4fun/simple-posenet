from pose_queue import PoseQueue
from enum import Enum

LONG_TIME = 32  # 一个蹲起时间
SHORT_TIME = 5

class State(Enum):
    FINISHED = 0
    DOWN     = 1
    UP       = 2


class Action(Enum):
    KEEP = 0
    DOWN = 1
    UP   = 2    

class PoseAnalyzer:
    def __init__(self):
        self.ShortQueue = PoseQueue(SHORT_TIME) # 
        self.LongQueue = PoseQueue(LONG_TIME) 

        self.state = State.FINISHED

    def do(self, pose):
        self.ShortQueue.push(pose)
        self.LongQueue.push(pose)

        (_, max_hip, _) = self.LongQueue.MaxY()        
        (_, min_hip, _) = self.LongQueue.MinY()

        action = Action.KEEP
        score = 0

        short_list = self.ShortQueue.all()
        if len(short_list) >= SHORT_TIME:
            (_, hip1, _) = short_list[0].RelativeY()
            (_, hip2, _) = short_list[SHORT_TIME - 1].RelativeY()

            if hip1 > hip2 and (hip1 - hip2) > max_hip / 5:
                action = Action.UP
            if hip1 < hip2 and (hip2 - hip1) > max_hip / 5:
                action = Action.DOWN

        done = False

        if self.state == State.FINISHED:
            if action == Action.DOWN:
                self.state = State.DOWN
        elif self.state == State.DOWN:
            if action == Action.UP:
                self.state = State.UP
        elif self.state == State.UP:
            if action == Action.KEEP or action == Action.DOWN:
                self.state = State.FINISHED
                done = True



        return (done, action, score)