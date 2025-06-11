import numpy as np


class DetectionHistory:
    def __init__(self, max_len=100, distance_threshold=2.0):
        self.max_len = max_len
        self.boxes_history = []
        self.types_history = []
        self.scores_history = []
        self.distance_threshold = distance_threshold
    
    def get_box_center(self, box):
        """获取3D框的中心点坐标"""
        return box[:3]  # 假设box的前3个值为x,y,z坐标
    
    def check_box_distance(self, box1, box2):
        """检查两个3D框的中心点距离"""
        center1 = self.get_box_center(box1)
        center2 = self.get_box_center(box2)
        distance = np.linalg.norm(center1 - center2)
        return distance
    
    def filter_with_history(self, new_boxes, new_types, new_scores):
        """
        将新检测到的框与历史记录中的框进行比较，过滤重叠的框
        Args:
            new_boxes: numpy array of shape (N, 7) for new 3D boxes
            new_types: list of new detection types
            new_scores: list of new detection scores
        Returns:
            filtered boxes, types and scores
        """
        if not self.boxes_history:  # 如果历史记录为空
            return new_boxes, new_types, new_scores
        
        # 创建一个布尔数组，标记哪些新框需要保留
        keep_mask = np.ones(len(new_boxes), dtype=bool)
        
        # 创建一个布尔数组，标记哪些历史框需要保留
        keep_history_mask = np.ones(len(self.boxes_history), dtype=bool)
        
        # 比较新框与历史框
        for i, new_box in enumerate(new_boxes):
            new_score = new_scores[i]
            
            for j, hist_box in enumerate(self.boxes_history):
                # 只有当历史框还标记为保留时才进行比较
                if keep_history_mask[j]:
                    distance = self.check_box_distance(new_box, hist_box)
                    
                    # 如果距离小于阈值，认为是重叠框
                    if distance < self.distance_threshold:
                        hist_score = self.scores_history[j]
                        
                        # 比较得分，保留得分高的框
                        if new_score > hist_score:
                            # 新框得分更高，丢弃历史框
                            keep_history_mask[j] = False
                        else:
                            # 历史框得分更高或相等，丢弃新框
                            keep_mask[i] = False
                            break
        
        # 过滤新框
        filtered_boxes = new_boxes[keep_mask]
        filtered_types = [t for i, t in enumerate(new_types) if keep_mask[i]]
        filtered_scores = [s for i, s in enumerate(new_scores) if keep_mask[i]]
        
        # 过滤历史框
        self.boxes_history = [box for i, box in enumerate(self.boxes_history) if keep_history_mask[i]]
        self.types_history = [t for i, t in enumerate(self.types_history) if keep_history_mask[i]]
        self.scores_history = [s for i, s in enumerate(self.scores_history) if keep_history_mask[i]]
        
        return filtered_boxes, filtered_types, filtered_scores
        
    def update(self, new_boxes, new_types, new_scores):
        """
        更新检测历史
        Args:
            boxes: numpy array of shape (N, 7) for 3D boxes
            types: list of detection types
            scores: list of detection scores
        """
        # 先进行重叠框过滤
        filtered_boxes, filtered_types, filtered_scores = self.filter_with_history(
            new_boxes, new_types, new_scores
        )
        
        # 将过滤后的新框添加到历史记录中
        for i in range(len(filtered_boxes)):
            self.boxes_history.append(filtered_boxes[i])
            self.types_history.append(filtered_types[i])
            self.scores_history.append(filtered_scores[i])
        
        # 保持历史记录在最大长度以内
        while len(self.boxes_history) > self.max_len:
            print(f"History length exceeded {self.max_len}, removing oldest entry.")
            self.boxes_history.pop(0)
            self.types_history.pop(0)
            self.scores_history.pop(0)
            
    def get_current(self):
        """
        获取最新的检测结果
        """
        if not self.boxes_history:
            return np.array([]), [], []
        
        # 将最新的框转换为numpy数组
        latest_boxes = np.array(self.boxes_history)
        latest_types = self.types_history
        latest_scores = self.scores_history
        
        return latest_boxes, latest_types, latest_scores
    
    def get_all_history(self):
        """
        获取所有历史检测结果
        """
        return self.boxes_history, self.types_history, self.scores_history
