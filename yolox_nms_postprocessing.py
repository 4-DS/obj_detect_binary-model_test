import numpy as np


class YOLOXPostProcessing():
    r""" postprocessing for output inference of onnx model YOLOX"""
    
    def __init__(self,
                strides = [(8, 8), (16, 16), (32, 32)],             # model.bbox_head.prior_generator.strides
                offset = 0,                                         # model.bbox_head.prior_generator.offset
                num_levels = 3,                                     # model.bbox_head.prior_generator.num_levels
                cls_out_channels = 1,                               # model.bbox_head.cls_out_channels
                num_classes = 1):                                   # model.bbox_head.num_classes
        self.strides = strides
        self.offset = offset
        self.num_levels = num_levels
        self.cls_out_channels = cls_out_channels
        self.num_classes = num_classes
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _meshgrid(x, y, row_major=True):   # model.bbox_head.prior_generator._meshgrid
        xx, yy = np.meshgrid(y, x)   
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        else:
            return yy.reshape(-1), xx.reshape(-1)
    
    @staticmethod
    def bbox2result(bboxes, labels, num_classes):  
        """Convert detection results to a list of numpy arrays.
        Args:
            bboxes (np.ndarray): shape (n, 5)
            labels (np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
        else:
            return [bboxes[labels == i, :] for i in range(num_classes)]
        
    def _bbox_decode(self, priors, bbox_preds):   # model.bbox_head._bbox_decode
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = np.exp(bbox_preds[..., 2:]) * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = np.stack([tl_x, tl_y, br_x, br_y], -1)    
        return decoded_bboxes
        
    def single_level_grid_priors(self,       
                                 featmap_size,
                                 level_idx,
                                 dtype=np.float32,
                                 with_stride=False):           # model.bbox_head.prior_generator.single_level_grid_priors
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (np.arange(0, feat_w) + self.offset) * stride_w   
        shift_x = shift_x.astype(dtype)

        shift_y = (np.arange(0, feat_h) + self.offset) * stride_h  
        shift_y = shift_y.astype(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = np.stack([shift_xx, shift_yy], axis=-1)
        else:
            stride_w = np.full((shift_xx.shape[0], ), stride_w).astype(dtype)  
            stride_h = np.full((shift_yy.shape[0], ),stride_h).astype(dtype)   
            shifts = np.stack([shift_xx, shift_yy, stride_w, stride_h], axis=-1) 
        all_points = shifts
        return all_points
    
    def grid_priors(self,
                    featmap_sizes,
                    dtype=np.float32,
                    with_stride=False):  # model.bbox_head.prior_generator.grid_priors
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: numpy.float32.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors
    
    def get_bboxes(self, 
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   scale_factor = None,
                   score_thr: float = 0.01,    
                   iou_threshold: float = 0.5,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[numpy.array]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[numpy.array]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[numpy.array], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            scale_factor (numpy.array[numpy.array], Optional): Rescale coefficents for input images. Default None.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        num_imgs = cls_scores[0].shape[0]
        
        if not isinstance(scale_factor, np.ndarray):
            scale_factor = np.array([[1.0, 1.0, 1.0, 1.0] for img_id in range(num_imgs)], dtype=np.float32)
        if not scale_factor.any():
            scale_factor = np.array([[1.0, 1.0, 1.0, 1.0] for img_id in range(num_imgs)], dtype=np.float32)        
        
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.grid_priors(featmap_sizes, 
                                       dtype=cls_scores[0].dtype,
                                       with_stride=True)
        
        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.transpose(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.transpose(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.transpose(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        
        flatten_cls_scores = self._sigmoid(np.concatenate(flatten_cls_scores, axis=1))
        flatten_bbox_preds = np.concatenate(flatten_bbox_preds, axis=1)
        flatten_objectness = self._sigmoid(np.concatenate(flatten_objectness, axis=1))
        flatten_priors = np.concatenate(mlvl_priors)
        
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        
        # rescale - return boxes in original image space
        # print(scale_factor)
        flatten_bboxes = [flatten_bboxes[id_img, ..., :4] / np.expand_dims(scale_factor[id_img], axis=0) for id_img in range(num_imgs)]
        flatten_bboxes = np.stack(flatten_bboxes, axis=0)
        
        result_list = []
        
        for img_id in range(num_imgs):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]
            
            result_list.append(
                self._bboxes_nms(cls_scores, bboxes, score_factor, score_thr=score_thr, iou_thr=iou_threshold))
        
        bbox_results = [self.bbox2result(det_bboxes, det_labels, self.num_classes) for det_bboxes, det_labels in result_list]
        
        return bbox_results
    
    def _bboxes_nms(self, cls_scores, bboxes, score_factor, score_thr, iou_thr):  #model.bbox_head._bboxes_nms
        max_scores = np.max(cls_scores, 1)
        labels = np.argmax(cls_scores, 1)
        
        valid_mask = score_factor * max_scores >= score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]
        
        # return bboxes, scores, labels
        if labels.size == 0:
            return bboxes, labels
        else:
            dets, keep = NMS.batched_nms(bboxes, scores, labels, iou_threshold = iou_thr, score_threshold=score_thr)
            return dets, labels[keep]        

class NMS: 
    @staticmethod
    def nms_op(boxes,
               scores,
               iou_threshold: float = 0.5,
               score_threshold: float = 0.1,               
               offset: int = 0,
               max_num: int = -1):

        assert boxes.shape[-1] == 4
        assert boxes.shape[0] == scores.shape[0]
        assert offset in (0, 1)
               
        valid_mask = scores > score_threshold
        boxes, scores = boxes[valid_mask], scores[valid_mask]
        valid_inds = np.nonzero(valid_mask)[0]      

        inds = NMS.nms_cpu(boxes, scores, iou_threshold, offset)
        
        if max_num > 0:
            inds = inds[:max_num]
        inds = valid_inds[inds]
        
        dets = np.concatenate([boxes[inds], scores[inds].reshape(-1, 1)], axis=-1)   
        return dets, inds
    
    @staticmethod
    def nms_cpu(boxes, scores, iou_threshold, offset):
        if boxes.size == 0:
            return np.array([])
        x1_t = boxes[:, 0]
        y1_t = boxes[:, 1]
        x2_t = boxes[:, 2]
        y2_t = boxes[:, 3]

        areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset)
        order_t = np.argsort(scores)[::-1]

        nboxes = boxes.shape[0]
        select_t = np.ones(nboxes, dtype = np.bool_)

        for _i in range(nboxes):
            if not select_t[_i]:
                continue
            i = order_t[_i]
            ix1 = x1_t[i]
            iy1 = y1_t[i]
            ix2 = x2_t[i]
            iy2 = y2_t[i]
            iarea = areas_t[i]

            for _j in range(_i+1, nboxes):
                if not select_t[_j]:
                    continue
                j = order_t[_j];
                xx1 = max(ix1, x1_t[j])
                yy1 = max(iy1, y1_t[j])
                xx2 = min(ix2, x2_t[j])
                yy2 = min(iy2, y2_t[j])

                w = max(0, xx2 - xx1 + offset)
                h = max(0, yy2 - yy1 + offset)

                inter = w * h
                ovr = inter / (iarea + areas_t[j] - inter)
                if (ovr > iou_threshold):
                    select_t[_j] = False
        return order_t[select_t]
    
    @staticmethod
    def batched_nms(boxes,
                    scores,
                    idxs, 
                    iou_threshold: float = 0.5,
                    score_threshold: float = 0.1,
                    class_agnostic: bool = False,
                    offset:int = 0):
        r"""Performs non-maximum suppression in a batched fashion.

        Modified from `torchvision/ops/boxes.py#L39
        <https://github.com/pytorch/vision/blob/
        505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
        In order to perform NMS independently per class, we add an offset to all
        the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap.

        Args:
            boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
            scores (torch.Tensor): scores in shape (N, ).
            idxs (torch.Tensor): each index value correspond to a bbox cluster,
                and NMS will not be applied between elements of different idxs,
                shape (N, ).
            iou_threshold (float): IoU threshold used for NMS.
            class_agnostic (bool): if true, nms is class agnostic,
                i.e. IoU thresholding happens over all boxes,
                regardless of the predicted class. Defaults to False.

        Returns:
            tuple: kept dets and indice.

            - boxes (Tensor): Bboxes with score after nms, has shape
              (num_bboxes, 5). last dimension 5 arrange as
              (x1, y1, x2, y2, score)
            - keep (Tensor): The indices of remaining boxes in input
              boxes.
        """
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            # When using rotated boxes, only apply offsets on center.
            if boxes.shape[-1] == 5:
                # Strictly, the maximum coordinates of the rotating box
                # (x,y,w,h,a) should be calculated by polygon coordinates.
                # But the conversion from rotated box to polygon will
                # slow down the speed.
                # So we use max(x,y) + max(w,h) as max coordinate
                # which is larger than polygon max coordinate
                # max(x1, y1, x2, y2,x3, y3, x4, y4)
                max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
                offsets = idxs.astype(boxes.dtype) + (max_coordinate + np.array(1).astype(boxes.dtype))
                boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
                boxes_for_nms = np.concatenate([boxes_ctr_for_nms, bboxes[..., 2:5]], axis=-1)
            else:
                max_coordinate = boxes.max()
                offsets = idxs.astype(boxes.dtype) + (max_coordinate + np.array(1).astype(boxes.dtype))           
                boxes_for_nms = boxes + offsets[:, None]
                
        dets, keep = NMS.nms_op(boxes_for_nms, scores, iou_threshold, score_threshold, offset)
        boxes = boxes[keep]
        scores = dets[:, -1]
        boxes = np.concatenate([boxes, scores[:, None]], axis= -1)       
        return boxes, keep
    
    