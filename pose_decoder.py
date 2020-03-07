class PoseDecoder:
    def __init__(self, net_result):
        self.stride = 16    # 块大小    
        self.threshold = 0.5
        self.max_detection = 5
        self.nmsr = 50
        self.lmr = 1

        (self.scores, self.offsets, self.displace) = net_result

    def decode_single(self):
        ret = {}

        # 获取关节数量
        keys = self.scores.shape[3]
        
        # 读取每一个关节的最高热点值坐标
        for key in range(keys):
            pos = self.MaxHeat(key)
            ret[key] = pos
        
        return ret

    def decode_multi(self):
        pass


    def MaxHeat(self, key):
        [count_y, count_x, keys] = self.scores.shape[1:]

        # 搜索热力值最高的块
        max_y = 0   
        max_x = 0
        max_value = self.scores[0, 0, 0, key]
        for cy in range(count_y):
            for cx in range(count_x):
                value = self.scores[0, cy, cx, key]
                if value > max_value:
                    max_y = cy
                    max_x = cx
                    max_value = value

        # 关节坐标 = 块坐标 * 块大小 + 块内坐标
        x = self.offsets[0, max_y, max_x, key + keys] + max_x * self.stride
        y = self.offsets[0, max_y, max_x, key] + max_y * self.stride

        return (x, y, max_value)
