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

        return int((yl + yr) / 2)


    def YPos(self):
        shoulder = self.getRL_Y("SHOULDER")
        hip = self.getRL_Y("HIP")
        knee = self.getRL_Y("KNEE")
        ankle = self.getRL_Y("ANKLE")

        return (shoulder, hip, knee, ankle)

    def RelativeY(self):
        (shoulder, hip, knee, ankle) = self.YPos()  

        return (ankle - shoulder, ankle - hip, ankle - knee)

    def __getattr__(self, name):
        return self.get(name)
