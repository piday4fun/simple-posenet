from pose_draw import PartIDs


class PoseSet:
    def __init__(self, pose):
        self.Pose = pose

    def get(self, name):
        index = PartIDs[name]

        return self.Pose[index]

    def getRL_Y(self, name):
        (_, yl, _) = self.get("LEFT_" + name)
        (_, yr, _) = self.get("RIGHT_" + name)
        
        return int(round((yl + yr) / 2))

    def __getattr__(self, name):
        return self.get(name)