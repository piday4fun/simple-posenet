import cv2

PartNames = [
  'NOSE',
  'LEFT_EYE',
  'RIGHT_EYE',
  'LEFT_EAR',
  'RIGHT_EAR',
  'LEFT_SHOULDER',
  'RIGHT_SHOULDER',
  'LEFT_ELBOW',
  'RIGHT_ELBOW',
  'LEFT_WRIST',
  'RIGHT_WRIST',
  'LEFT_HIP',
  'RIGHT_HIP',
  'LEFT_KNEE',
  'RIGHT_KNEE',
  'LEFT_ANKLE',
  'RIGHT_ANKLE'
]

PoseChain = [
    ['NOSE', 'LEFT_EYE'],
    ['LEFT_EYE', 'LEFT_EAR'],
    ['NOSE', 'RIGHT_EYE'],
    ['RIGHT_EYE', 'RIGHT_EAR'],
    ['NOSE', 'LEFT_SHOULDER'],
    ['LEFT_SHOULDER', 'LEFT_ELBOW'],
    ['LEFT_ELBOW', 'LEFT_WRIST'],
    ['LEFT_SHOULDER', 'LEFT_HIP'],
    ['LEFT_HIP', 'LEFT_KNEE'],
    ['LEFT_KNEE', 'LEFT_ANKLE'],
    ['NOSE', 'RIGHT_SHOULDER'],
    ['RIGHT_SHOULDER', 'RIGHT_ELBOW'],
    ['RIGHT_ELBOW', 'RIGHT_WRIST'],
    ['RIGHT_SHOULDER', 'RIGHT_HIP'],
    ['RIGHT_HIP', 'RIGHT_KNEE'],
    ['RIGHT_KNEE', 'RIGHT_ANKLE']
]

PartIDs = {}
PartLinks = {}

def build():
    for i in range(len(PartNames)):
        name = PartNames[i]
        PartIDs[name] = i

    for pos in PoseChain:
        [src, dest] = pos
        src_id = PartIDs[src]
        dest_id = PartIDs[dest]

        if src_id in PartLinks:
           PartLinks[src_id].append(dest_id) 
        else:
            PartLinks[src_id] = [dest_id]
    
class PoseDrawer:
    def __init__(self, size):
        (self.Width, self.Height) = size

    def Draw(self, image, pose):
        (height, widht, channel) = image.shape
        scale_x = widht / self.Width
        scale_y = height / self.Height

        scale = lambda x, y:  (int(x * scale_x), int(y * scale_y))

        point_color = (0,244,289)
        link_color = (0,0,255)
        hint_color = (255, 0 , 0)

        for key in pose:
            # 绘制关节
            (x, y, score) = pose[key]
            pos = scale(x, y)

            color = hint_color if key == PartIDs["LEFT_HIP"] or key == PartIDs["RIGHT_HIP"] else point_color
            cv2.circle(image, pos, 3, color, 2)


            # 绘制关节连线
            if key in PartLinks:
                link_keys = PartLinks[key]
                for lk in link_keys:
                    (link_x, link_y, link_score) = pose[lk]
                    link_pos = scale(link_x, link_y)
                    cv2.line(image, pos, link_pos, link_color, 1)

#
#
#    
build()
