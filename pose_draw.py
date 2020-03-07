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
    

def DrawPos(image, pose):
    (height, widht, channel) = image.shape
    scale_x = widht / 257.0
    scale_y = height / 353.0

    scale = lambda x, y:  (int(x * scale_x), int(y * scale_y))

    # (random.randint(50,250),random.randint(50,250),random.randint(50,250))
    point_color = (0,244,289)
    link_color = (0,0,255)

    for key in pose:
        # 绘制关节
        (x, y, score) = pose[key]
        pos = scale(x, y)
        cv2.circle(image, pos, 3, point_color, 2)


        # 绘制关节连线
        if key in PartLinks:
            link_keys = PartLinks[key]
            for lk in link_keys:
                (link_x, link_y, link_score) = pose[lk]
                link_pos = scale(link_x, link_y)
                cv2.line(image, pos, link_pos, link_color, 1)

   
build()
