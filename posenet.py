#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
import numpy as np 
import cv2

class PoseNet():
    def __init__(self, model):
        self.interpreter = tflite.Interpreter(model_path = model)
        self.interpreter.allocate_tensors()

        self.input_detail = self.interpreter.get_input_details()
        self.output_detail = self.interpreter.get_output_details()

        [input_h, input_w] = self.input_detail[0]["shape"][1:3]
        [output_h, output_w] = self.output_detail[0]["shape"][1:3]

        self.InputSize = (input_w, input_h)
        self.Stride = (round(input_w / output_w), round(input_h / output_h))

    #
    #   输入为 257*353（宽*高）的RGB图像
    #
    def input(self, image):
        self.interpreter.set_tensor(self.input_detail[0]['index'], image)


    #
    #   输出3个张量：
    #      scores：   17*23*17 (宽*高*关节) 关节热力图，每个关节 17*23个块，热力值表示可信度
    #      offsets：  17*23*34 (宽*高*(关节*2))关节块内坐标，34个层前17个是每个关节的块内Y坐标，后17个是每个关节的块内X坐标
    #      displace： 17*23*64 16个多目标偏移(相对于块)，每个目标有前向偏移X，Y，反向偏移X，Y各4层
    #
    def output(self):
        scores = self.interpreter.get_tensor(self.output_detail[0]['index'])
        offsets = self.interpreter.get_tensor(self.output_detail[1]['index'])
        displace = self.interpreter.get_tensor(self.output_detail[2]['index'])

        return (scores, offsets, displace)

    #  
    #   调整图像为网络定义的格式
    #
    def preprocess(self, image):
        # 分辨率调整为 257*353
        (w, h, d) = image.shape
        (image_w, image_h) = self.InputSize

        if not (w == image_w and h == image_h):  
            image = cv2.resize(image, (image_w, image_h))

        # 增加额外的维度
        expand_img = np.expand_dims(image, axis = 0)
        # 像素值调整为单精度浮点
        float_img = expand_img.astype('float32')
        # 像素值范围调整为 [-1, 1]
        float_img = float_img / 255 * 2.0 - 1

        return float_img

    #
    #   调用网络
    #
    def feed(self, image):
        net_image = self.preprocess(image)
        self.input(net_image)
        self.interpreter.invoke()
        
        return self.output()


   
