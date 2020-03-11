import time
import cv2
import os
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

from posenet import PoseNet
from pose_decoder import PoseDecoder
from pose_draw import PoseDrawer
from pose_queue import PoseQueue
from pose_analyzer import PoseAnalyzer

#model_path = "model/multi_person_mobilenet_v1_075_float.tflite"
model_path = "model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
rtsp_url = "rtmp://solasolo.oicp.net/live"

start_time = time.time()
read_time = 0
resize_time = 0
predict_time = 0
draw_time = 0
frames = 0

iter = None


def GIFIterator(file):
    gif = Image.open(file)
    frames_count = gif.n_frames
    frame_index = 0

    while True:
        gif.seek(frame_index)
        frame = Image.new("RGB", gif.size)
        frame.paste(gif)

        frame_index += 1
        if frame_index >= frames_count:
            frame_index = 0

        image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        yield image


def ImageIterator(path):
    images = os.listdir(path)
    image_count = len(images)
    image_index = 0

    while True:
        image_file = images[image_index]
        if image_file.endswith(".jpg"):
            image_path = path + "/" + image_file
            image = cv2.imread(image_path)

            yield image

        image_index += 1
        if image_index >= image_count:
            image_index = 0


def VedeoIterator(url):
    while True:
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        while ret:
            ret, frame = cap.read()
            if ret:
                small = cv2.resize(frame, (257, 353))

                yield small


def Timer(var):
    def decorate(func):
        def wrapper(*args):
            start_time = time.time()
            ret = func(*args)
            var += time.time() - start_time

            return ret
        return wrapper

    return decorate


def PrintTimer():
    if frames % 100 == 0:
        print("frames: %d, fps:%f, read_time:%f, resize_time:%f, predict_time:%f, draw_time:%f" % (
            frames, frames / (time.time() - start_time), read_time, resize_time, predict_time, draw_time))


def ShowFPS(image):
    text = "fps:%2.2f" % (frames / (time.time() - start_time))
    cv2.putText(image, text, (5, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)


def read(iter, skip = 1):
    for i in range(skip):
        image = iter.__next__()
    # ret_val, image = cam.read()

    return image


def SelectSource(source):
    global iter

    if source == "images":
        iter = ImageIterator("images")
    elif source == "gif":
        iter = GIFIterator("images/test.gif")
    elif source == "mp4":
        iter = VedeoIterator("images/test1.mp4")
    elif source == "rtsp":
        iter = VedeoIterator(rtsp_url)
    elif source == "camera":
        iter = VedeoIterator(0)

def ShowChart(queue):
    w = 600
    h = 400
    base1 = 100
    base2 = 300

    image = np.zeros((h, w, 3), np.uint8)
    image.fill(0)

    poses = queue.all()
    length = len(poses)

    line = lambda i, y1, y2, c: cv2.line(image, ((i - 1) * 6, y1), (i * 6, y2), c, 1)

    if length > 0:
        (shoulder_y1, hip_y1, knee_y1) = poses[0].RelativeY()

        cv2.line(image, (0, base1), (w -1, base1), (127, 127, 127), 1)        
        cv2.line(image, (0, base2), (w -1, base2), (127, 127, 127), 1)

        for index in range(1, length):
            pose = poses[index]
            (shoulder_y2, hip_y2, knee_y2) = pose.RelativeY()

            line(index, hip_y1, hip_y2, (255, 0, 0))
            line(index, shoulder_y1 - hip_y1 + base1, shoulder_y2 - hip_y2 + base1, (0, 255, 0))
            line(index, hip_y1 - knee_y1 + base2, hip_y2 - knee_y2 + base2, (0, 255, 0))
            
            hip_y1 = hip_y2
            knee_y1 = knee_y2
            shoulder_y1 =shoulder_y2

    cv2.imshow("chart", image)


def test():
    global frames

    net = PoseNet(model_path)
    drawer = PoseDrawer(net.InputSize)
    queue = PoseQueue()
    Analyzer = PoseAnalyzer()

    action_time = None

    while True:
        frames += 1

        image = read(iter, 10)
        output = net.feed(image)
        decoder = PoseDecoder(output, 32)
        pose = decoder.decode_single()
        drawer.Draw(image, pose)

        queue.push(pose)

        ShowFPS(image)
        cv2.imshow("pic", image)


        lst = queue.all()
        (shoulder, hip, knee) = lst[0].RelativeY()
        (done, action, _) = Analyzer.do(pose)
        
        if done:
            now_time = time.time()
            t = 0 if action_time == None else now_time - action_time
            
            action_time = now_time
            print("Done", t)
        # print(shoulder, hip, knee, action, done)

        ShowChart(queue)
        # PrintTimer()

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
#
# main
#


SelectSource("mp4")
test()
