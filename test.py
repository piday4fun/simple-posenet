import time
import cv2
import os
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

from posenet import PoseNet
from pose_decoder import PoseDecoder
from pose_draw import DrawPos

model_path = "model/multi_person_mobilenet_v1_075_float.tflite"
rtsp_url = ""

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
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
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


def read(iter):
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

def test():
    global frames

    net = PoseNet(model_path)

    while True:
        image = read(iter)
        output = net.feed(image)
        decoder = PoseDecoder(output)
        pose =  decoder.decode_single()
        DrawPos(image, pose)

        ShowFPS(image)
        cv2.imshow("result", image)

        frames += 1
        # PrintTimer()

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
#
# main
#

SelectSource("mp4")
test()
