def bb_intersection_over_union(coco_boxA, coco_boxB):
    # determine the (x_top_left, y_top_left, h, w)-coordinates of the intersection rectangle
    boxA = [coco_boxA[0], coco_boxA[1], coco_boxA[0]+coco_boxA[2], coco_boxA[1]+coco_boxA[3]]
    boxB = [coco_boxB[0], coco_boxB[1], coco_boxB[0]+coco_boxB[2], coco_boxB[1]+coco_boxB[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou