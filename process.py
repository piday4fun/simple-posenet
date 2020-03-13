import time
from queue import Queue
from threading import Thread
from pose_decoder import PoseDecoder
from posenet import PoseNet
from pose_analyzer import PoseAnalyzer

class Process:
    def __init__(self, net, callback):
        self.ImageQueue = Queue()
        self.PoseQueue = Queue()
        self.OutPoseQueue = Queue()
        self.Net = net
        self.FrameCount = 0
        self.StartTime = time.time()
        self.Callback = callback

        self.Runing = True

        thread1 = Thread(target = self.Recognition)
        thread2 = Thread(target = self.Analysis)

        thread1.start()
        thread2.start()

    def setImage(self, image):
        if self.ImageQueue.empty():
            self.ImageQueue.put(image, True)

    def getPose(self):
        ret = None

        if not self.OutPoseQueue.empty():
            ret = self.OutPoseQueue.get(True)

        return ret

    def getFPS(self):
        now = time.time()

        return self.FrameCount / (now - self.StartTime)


    def Shutdwon(self):
        self.Runing = False

    def Recognition(self):
        while self.Runing:
            self.FrameCount += 1

            image = self.ImageQueue.get(True)
            output = self.Net.feed(image)
            decoder = PoseDecoder(output, 32)
            pose = decoder.decode_single()
            self.PoseQueue.put(pose, True)

            self.OutPoseQueue.put((image, pose), True)

    def Analysis(self):
        count = 0
        action_time = None
        Analyzer = PoseAnalyzer()

        while self. Runing:
            pose = self.PoseQueue.get(True)
            (done, action, _) = Analyzer.do(pose)

            if done:
                now_time = time.time()
                t = 0 if action_time == None else now_time - action_time

                action_time = now_time
                count += 1
                
                self.Callback({"count": count, "time": t})
