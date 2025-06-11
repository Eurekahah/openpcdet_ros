import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Point
from collections import deque, defaultdict
import json
import time
import csv
import os
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from history import DetectionHistory  


class HistoryBasedEvaluator:
    def __init__(self, detection_history, csv_log_dir='/tmp/detection_logs'):
        """
        基于DetectionHistory的评估器
        Args:
            detection_history: DetectionHistory实例
        """
        self.detection_history = detection_history
        
        # 评估参数
        self.stability_window = 30  # 用于计算稳定性的历史窗口
        self.cluster_eps = 1.5      # DBSCAN聚类参数
        self.min_samples = 3        # 最小样本数
        
        # 存储评估指标历史
        self.metrics_history = deque(maxlen=100)
        self.last_evaluation_time = time.time()

        # CSV日志设置
        self.csv_log_dir = csv_log_dir
        self.setup_csv_logging()
        
        # ROS发布器
        self.metrics_pub = rospy.Publisher('/detection_metrics', Float32MultiArray, queue_size=1)
        self.report_pub = rospy.Publisher('/detection_report', String, queue_size=1)
        
        # 类别特定的合理性范围
        self.class_size_ranges = {
            'Car': {'min': [3.0, 1.5, 1.0], 'max': [6.0, 2.5, 3.0]},
            'Pedestrian': {'min': [0.5, 0.5, 1.5], 'max': [1.0, 1.0, 2.0]},
            'Cyclist': {'min': [1.0, 0.5, 1.5], 'max': [2.5, 1.0, 2.0]}
        }

        # 
        self.class_height_ranges = {
            'Car': {
                'min_z': -0.7,  # 车辆底部可能略低于道路表面
                'max_z': -0.2,   # 车辆顶部最高3.5米
                'ideal_z_range': [-0.5, -0.4]  # 车辆中心的理想z轴范围
            },
            'Pedestrian': {
                'min_z': -0.7,   # 行人中心最低0.5米（考虑身高）
                'max_z': -0.2,   # 行人中心最高2.2米
                'ideal_z_range': [-0.5, -0.4]  # 行人中心的理想z轴范围
            },
            'Cyclist': {
                'min_z': -0.7,   # 骑自行车中心最低0.0米（考虑身高）
                'max_z': -0.2,   # 骑自行车中心最高2.0米
                'ideal_z_range': [-0.5, -0.4]  # 骑自行车中心的理想z轴范围
            }
        }
    
    def setup_csv_logging(self):
        """设置CSV日志记录"""
        # 创建日志目录
        if not os.path.exists(self.csv_log_dir):
            os.makedirs(self.csv_log_dir)
        
        # 创建带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 主要指标日志文件
        self.metrics_csv_path = os.path.join(self.csv_log_dir, f"detection_metrics_{timestamp}.csv")
        
        # 详细检测数据日志文件
        self.detections_csv_path = os.path.join(self.csv_log_dir, f"detection_details_{timestamp}.csv")
        
        # 趋势分析日志文件
        self.trends_csv_path = os.path.join(self.csv_log_dir, f"detection_trends_{timestamp}.csv")
        
        # 初始化CSV文件头部
        self.init_csv_files()
        
        rospy.loginfo(f"CSV logging initialized:")
        rospy.loginfo(f"  Metrics: {self.metrics_csv_path}")
        rospy.loginfo(f"  Details: {self.detections_csv_path}")
        rospy.loginfo(f"  Trends: {self.trends_csv_path}")
    
    def init_csv_files(self):
        """初始化CSV文件头部"""
        # 主要指标CSV头部
        metrics_headers = [
            'timestamp',
            'datetime',
            'overall_quality',
            'position_reasonableness',
            'spatial_consistency',
            'score_consistency',
            'size_reasonableness',
            'detection_density',
            'class_distribution',
            'total_detections',
            'unique_classes',
            'average_score',
            'session_id'
        ]
        
        # 详细检测数据CSV头部
        details_headers = [
            'timestamp',
            'datetime',
            'detection_id',
            'class_type',
            'position_x',
            'position_y',
            'position_z',
            'size_length',
            'size_width',
            'size_height',
            'rotation',
            'confidence_score',
            'position_score',
            'size_score',
            'session_id'
        ]
        
        # 趋势分析CSV头部
        trends_headers = [
            'timestamp',
            'datetime',
            'metric_name',
            'trend_direction',
            'slope',
            'current_value',
            'moving_average',
            'volatility',
            'session_id'
        ]
        
        # 写入头部
        print("位置：  ", self.metrics_csv_path)
        self.write_csv_header(self.metrics_csv_path, metrics_headers)
        self.write_csv_header(self.detections_csv_path, details_headers)
        self.write_csv_header(self.trends_csv_path, trends_headers)
        
        # 生成会话ID
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def write_csv_header(self, filepath, headers):
        """写入CSV文件头部"""
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
        except Exception as e:
            rospy.logerr(f"Failed to write CSV header to {filepath}: {e}")
    
    def log_metrics_to_csv(self, metrics):
        """记录指标数据到CSV"""
        try:
            timestamp = time.time()
            datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            # 获取检测统计信息
            detection_summary = {
                'total_detections': len(self.detection_history.boxes_history),
                'unique_classes': len(set(self.detection_history.types_history)) if self.detection_history.types_history else 0,
                'average_score': np.mean(self.detection_history.scores_history) if self.detection_history.scores_history else 0.0
            }
            
            # 准备数据行
            row_data = [
                timestamp,
                datetime_str,
                metrics.get('overall_quality', 0.0),
                metrics.get('position_reasonableness', 0.0),
                metrics.get('spatial_consistency', 0.0),
                metrics.get('score_consistency', 0.0),
                metrics.get('size_reasonableness', 0.0),
                metrics.get('detection_density', 0.0),
                metrics.get('class_distribution', 0.0),
                detection_summary['total_detections'],
                detection_summary['unique_classes'],
                detection_summary['average_score'],
                self.session_id
            ]
            
            # 写入CSV文件
            with open(self.metrics_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            
            print("写入成功")
                
        except Exception as e:
            print(f"写入失败:{e}")
            rospy.logerr(f"Failed to log metrics to CSV: {e}")

    def log_detections_to_csv(self, boxes, types, scores):
        """记录详细检测数据到CSV"""
        try:
            timestamp = time.time()
            datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            for i, box in enumerate(boxes):
                if i >= len(types) or i >= len(scores):
                    continue
                
                # 计算单个检测的质量分数
                position_score = self.calculate_single_position_score(box, types[i])
                size_score = self.calculate_single_size_score(box, types[i])
                
                row_data = [
                    timestamp,
                    datetime_str,
                    i,  # detection_id
                    types[i],
                    float(box[0]),  # position_x
                    float(box[1]),  # position_y
                    float(box[2]),  # position_z
                    float(box[3]) if len(box) > 3 else 0.0,  # size_length
                    float(box[4]) if len(box) > 4 else 0.0,  # size_width
                    float(box[5]) if len(box) > 5 else 0.0,  # size_height
                    float(box[6]) if len(box) > 6 else 0.0,  # rotation
                    float(scores[i]),
                    position_score,
                    size_score,
                    self.session_id
                ]
                
                # 写入CSV文件
                with open(self.detections_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_data)
                    
        except Exception as e:
            rospy.logerr(f"Failed to log detections to CSV: {e}")
    
    def log_trends_to_csv(self, trends):
        """记录趋势分析到CSV"""
        try:
            timestamp = time.time()
            datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            for metric_name, trend_info in trends.items():
                if isinstance(trend_info, dict):
                    row_data = [
                        timestamp,
                        datetime_str,
                        metric_name,
                        trend_info.get('direction', 'unknown'),
                        trend_info.get('slope', 0.0),
                        trend_info.get('current_value', 0.0),
                        trend_info.get('moving_average', 0.0),
                        trend_info.get('volatility', 0.0),
                        self.session_id
                    ]
                else:
                    # 兼容旧格式
                    row_data = [
                        timestamp,
                        datetime_str,
                        metric_name,
                        str(trend_info),
                        0.0,  # slope
                        0.0,  # current_value
                        0.0,  # moving_average
                        0.0,  # volatility
                        self.session_id
                    ]
                
                # 写入CSV文件
                with open(self.trends_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_data)
                    
        except Exception as e:
            rospy.logerr(f"Failed to log trends to CSV: {e}")
    
    def calculate_single_position_score(self, box, class_type):
        """计算单个检测的位置合理性分数"""
        z_position = box[2]
        
        if class_type.lower() in ['car', 'vehicle', 'truck', 'bus']:
            height_range = self.class_height_ranges['Car']
        elif class_type.lower() in ['pedestrian', 'person', 'people']:
            height_range = self.class_height_ranges['Pedestrian']
        elif class_type.lower() in ['cyclist', 'bicycle', 'bike']:
            height_range = self.class_height_ranges['Cyclist']
        else:
            return 0.7  # 未知类别默认分数
        
        if height_range['min_z'] <= z_position <= height_range['max_z']:
            if height_range['ideal_z_range'][0] <= z_position <= height_range['ideal_z_range'][1]:
                return 1.0
            else:
                ideal_min, ideal_max = height_range['ideal_z_range']
                if z_position < ideal_min:
                    deviation = ideal_min - z_position
                else:
                    deviation = z_position - ideal_max
                return max(0.6, 1.0 - deviation * 0.3)
        else:
            if z_position < height_range['min_z']:
                return max(0.1, 0.5 + z_position * 0.5)
            else:
                excess = z_position - height_range['max_z']
                return max(0.1, 1.0 / (1.0 + excess))
    
    def calculate_single_size_score(self, box, class_type):
        """计算单个检测的尺寸合理性分数"""
        if len(box) < 6 or class_type not in self.class_size_ranges:
            return 0.7
        
        size = np.array(box[3:6])
        min_size = np.array(self.class_size_ranges[class_type]['min'])
        max_size = np.array(self.class_size_ranges[class_type]['max'])
        
        in_range = np.all((size >= min_size) & (size <= max_size))
        
        if in_range:
            return 1.0
        else:
            deviation = np.mean(np.maximum(0, min_size - size) + np.maximum(0, size - max_size))
            return 1.0 / (1.0 + deviation)

    def evaluate_all_metrics(self):
        """评估所有指标"""
        if not self.detection_history.boxes_history:
            return None
            
        boxes, types, scores = self.detection_history.get_all_history()
        
        if len(boxes) < 3:  # 需要至少3个检测结果进行评估
            return None
        
        metrics = {}
        
        # 1. 位置合理性评估
        metrics['position_reasonableness'] = self.evaluate_position_reasonableness(boxes, types, scores)
        
        # 2. 空间一致性评估
        metrics['spatial_consistency'] = self.evaluate_spatial_consistency(boxes, types)
        
        # 3. 分数一致性评估
        metrics['score_consistency'] = self.evaluate_score_consistency(scores, types)
        
        # 4. 尺寸合理性评估
        metrics['size_reasonableness'] = self.evaluate_size_reasonableness(boxes, types)
        
        # 5. 检测密度评估
        metrics['detection_density'] = self.evaluate_detection_density(boxes)
        
        # 6. 类别分布评估
        metrics['class_distribution'] = self.evaluate_class_distribution(types)
        
        # 7. 综合质量分数
        metrics['overall_quality'] = self.calculate_overall_quality(metrics)
        
        # 存储到历史记录
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        return metrics
    
    def evaluate_position_reasonableness(self, boxes, types, scores):
        """评估检测框位置的合理性（专门针对城市道路环境）"""
        if len(boxes) < 1:
            return 0.5
        
        reasonableness_scores = []
        boxes_array = np.array(boxes)
        
        # 城市道路环境的位置合理性参数
        road_surface_z = 0.0  # 假设道路表面为z=0
        
        for i, box in enumerate(boxes_array):
            if i >= len(types):
                continue
                
            class_type = types[i]
            z_position = box[2]  # z坐标
            
            # 根据类别评估z轴位置合理性
            if class_type.lower() in ['car', 'vehicle', 'truck', 'bus']:
                # 车辆位置评估
                if self.class_height_ranges['Car']['min_z'] <= z_position <= self.class_height_ranges['Car']['max_z']:
                    # 在合理范围内，进一步评估是否在理想范围
                    if self.class_height_ranges['Car']['ideal_z_range'][0] <= z_position <= self.class_height_ranges['Car']['ideal_z_range'][1]:
                        score = 1.0  # 理想位置
                    else:
                        # 计算偏离理想范围的程度
                        ideal_min, ideal_max = self.class_height_ranges['Car']['ideal_z_range']
                        if z_position < ideal_min:
                            deviation = ideal_min - z_position
                        else:
                            deviation = z_position - ideal_max
                        score = max(0.6, 1.0 - deviation * 0.3)  # 偏离惩罚
                else:
                    # 超出合理范围
                    if z_position < self.class_height_ranges['Car']['min_z']:
                        # 地下，严重不合理
                        score = max(0.1, 0.5 + z_position * 0.5)
                    else:
                        # 过高，可能是误检
                        excess = z_position - self.class_height_ranges['Car']['max_z']
                        score = max(0.1, 1.0 / (1.0 + excess))
                        
            elif class_type.lower() in ['pedestrian', 'person', 'people']:
                # 行人位置评估
                pedestrian_height_range = self.class_height_ranges['Pedestrian']
                if pedestrian_height_range['min_z'] <= z_position <= pedestrian_height_range['max_z']:
                    # 在合理范围内
                    if pedestrian_height_range['ideal_z_range'][0] <= z_position <= pedestrian_height_range['ideal_z_range'][1]:
                        score = 1.0  # 理想位置
                    else:
                        # 计算偏离理想范围的程度
                        ideal_min, ideal_max = pedestrian_height_range['ideal_z_range']
                        if z_position < ideal_min:
                            deviation = ideal_min - z_position
                        else:
                            deviation = z_position - ideal_max
                        score = max(0.7, 1.0 - deviation * 0.4)
                else:
                    # 超出合理范围
                    if z_position < pedestrian_height_range['min_z']:
                        # 过低，可能检测到腿部
                        score = max(0.3, z_position / pedestrian_height_range['min_z'])
                    else:
                        # 过高，不合理
                        excess = z_position - pedestrian_height_range['max_z']
                        score = max(0.2, 1.0 / (1.0 + excess))

            elif class_type.lower() in ['cyclist', 'bicycle', 'bike']:
                # 骑自行车位置评估
                cyclist_height_range = self.class_height_ranges['Cyclist']
                if cyclist_height_range['min_z'] <= z_position <= cyclist_height_range['max_z']:
                    # 在合理范围内
                    if cyclist_height_range['ideal_z_range'][0] <= z_position <= cyclist_height_range['ideal_z_range'][1]:
                        score = 1.0
                    else:
                        # 计算偏离理想范围的程度
                        ideal_min, ideal_max = cyclist_height_range['ideal_z_range']
                        if z_position < ideal_min:
                            deviation = ideal_min - z_position
                        else:
                            deviation = z_position - ideal_max
                        score = max(0.7, 1.0 - deviation * 0.4)
                else:
                    # 超出合理范围
                    if z_position < cyclist_height_range['min_z']:
                        # 过低，可能检测到腿部
                        score = max(0.3, z_position / cyclist_height_range['min_z'])
                    else:
                        # 过高，不合理
                        excess = z_position - cyclist_height_range['max_z']
                        score = max(0.2, 1.0 / (1.0 + excess))
            else:
                # 未知类别，给予中性分数但检查基本合理性
                if -1.0 <= z_position <= 5.0:  # 基本合理的高度范围
                    score = 0.7
                else:
                    score = 0.3
            
            reasonableness_scores.append(score)
        
        return np.mean(reasonableness_scores) if reasonableness_scores else 0.5
    
    def evaluate_spatial_consistency(self, boxes, types):
        """评估空间一致性"""
        if len(boxes) < 2:
            return 0.5
        
        consistency_scores = []
        boxes_array = np.array(boxes)
        
        # 按类别分组
        for class_type in set(types):
            class_indices = [i for i, t in enumerate(types) if t == class_type]
            if len(class_indices) < 2:
                continue
                
            class_boxes = boxes_array[class_indices]
            centers = class_boxes[:, :3]
            
            # 计算相邻检测之间的距离变化
            distances = []
            for i in range(len(centers) - 1):
                dist = np.linalg.norm(centers[i+1] - centers[i])
                distances.append(dist)
            
            if distances:
                # 距离变化的标准差，标准差越小越一致
                dist_std = np.std(distances)
                consistency = 1.0 / (1.0 + dist_std)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def evaluate_score_consistency(self, scores, types):
        """评估检测分数的一致性"""
        if len(scores) < 2:
            return 0.5
        
        consistency_scores = []
        
        # 按类别分组评估分数一致性
        for class_type in set(types):
            class_indices = [i for i, t in enumerate(types) if t == class_type]
            class_scores = [scores[i] for i in class_indices]
            
            if len(class_scores) >= 2:
                # 计算分数的变异系数
                mean_score = np.mean(class_scores)
                std_score = np.std(class_scores)
                
                if mean_score > 0:
                    cv = std_score / mean_score  # 变异系数
                    consistency = 1.0 / (1.0 + cv)
                    consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def evaluate_size_reasonableness(self, boxes, types):
        """评估检测框尺寸的合理性"""
        if not boxes:
            return 0.5
        
        reasonableness_scores = []
        boxes_array = np.array(boxes)
        
        for i, box in enumerate(boxes_array):
            class_type = types[i]
            
            if class_type in self.class_size_ranges:
                size = box[3:6]  # 假设box格式为[x,y,z,l,w,h,rotation]
                min_size = np.array(self.class_size_ranges[class_type]['min'])
                max_size = np.array(self.class_size_ranges[class_type]['max'])
                
                # 检查尺寸是否在合理范围内
                in_range = np.all((size >= min_size) & (size <= max_size))
                
                if in_range:
                    reasonableness_scores.append(1.0)
                else:
                    # 计算偏离程度
                    deviation = np.mean(np.maximum(0, min_size - size) + np.maximum(0, size - max_size))
                    score = 1.0 / (1.0 + deviation)
                    reasonableness_scores.append(score)
            else:
                # 未知类别给予中性分数
                reasonableness_scores.append(0.7)
        
        return np.mean(reasonableness_scores) if reasonableness_scores else 0.5
    
    def evaluate_detection_density(self, boxes):
        """评估检测密度的合理性"""
        if len(boxes) < 2:
            return 0.5
        
        boxes_array = np.array(boxes)
        centers = boxes_array[:, :3]
        
        # 计算最近邻距离
        distances = cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)  # 忽略自身距离
        
        min_distances = np.min(distances, axis=1)
        
        # 评估密度合理性（不应该太密集也不应该太稀疏）
        mean_min_dist = np.mean(min_distances)
        
        # 理想距离范围（根据应用场景调整）
        ideal_min_dist = 3.0  # 3米
        ideal_max_dist = 20.0  # 20米
        
        if mean_min_dist < ideal_min_dist:  # 太密集
            density_score = mean_min_dist / ideal_min_dist
        elif mean_min_dist > ideal_max_dist:  # 太稀疏
            density_score = ideal_max_dist / mean_min_dist
        else:  # 合理范围
            density_score = 1.0
        
        return min(1.0, max(0.0, density_score))
    
    def evaluate_class_distribution(self, types):
        """评估类别分布的合理性"""
        if not types:
            return 0.5
        
        # 统计各类别数量
        class_counts = defaultdict(int)
        for t in types:
            class_counts[t] += 1
        
        total_count = len(types)
        
        # 评估分布的多样性（香农熵）
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                p = count / total_count
                entropy -= p * np.log2(p)
        
        # 归一化熵值
        max_entropy = np.log2(len(class_counts)) if class_counts else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def calculate_overall_quality(self, metrics):
        """计算综合质量分数"""
        weights = {
            'position_reasonableness': 0.25,
            'spatial_consistency': 0.20,
            'score_consistency': 0.15,
            'size_reasonableness': 0.20,
            'detection_density': 0.10,
            'class_distribution': 0.10
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                overall_score += metrics[metric] * weight
                total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0
    
    def generate_evaluation_report(self, metrics):
        """生成评估报告"""
        report = {
            'timestamp': time.time(),
            'overall_quality': metrics.get('overall_quality', 0.0),
            'detailed_metrics': metrics,
            'recommendations': self.generate_recommendations(metrics),
            'detection_summary': {
                'total_detections': len(self.detection_history.boxes_history),
                'unique_classes': len(set(self.detection_history.types_history)),
                'average_score': np.mean(self.detection_history.scores_history) if self.detection_history.scores_history else 0.0
            }
        }
        
        return report
    
    def generate_recommendations(self, metrics):
        """根据评估结果生成改进建议"""
        recommendations = []
        
        if metrics.get('position_reasonableness', 0) < 0.5:
            recommendations.append("检测位置不合理，请检查z轴高度是否符合道路环境")
        
        if metrics.get('spatial_consistency', 0) < 0.5:
            recommendations.append("空间一致性较差，可能存在误检或漏检")
        
        if metrics.get('score_consistency', 0) < 0.5:
            recommendations.append("检测分数波动较大，建议检查模型置信度设置")
        
        if metrics.get('size_reasonableness', 0) < 0.5:
            recommendations.append("检测框尺寸不合理，建议检查模型参数或后处理逻辑")
        
        if metrics.get('detection_density', 0) < 0.5:
            recommendations.append("检测密度异常，可能存在重复检测或检测遗漏")
        
        return recommendations
    
    def publish_metrics(self, metrics):
        """发布评估指标到ROS话题"""
        # 发布数值指标
        metrics_msg = Float32MultiArray()
        metrics_msg.data = [
            float(metrics.get('overall_quality',         0.0)),
            float(metrics.get('position_reasonableness', 0.0)),
            float(metrics.get('spatial_consistency',     0.0)),
            float(metrics.get('score_consistency',       0.0)),
            float(metrics.get('size_reasonableness',     0.0)),
            float(metrics.get('detection_density',       0.0)),
            float(metrics.get('class_distribution',      0.0))
        ]
        self.metrics_pub.publish(metrics_msg)
        
        # 发布详细报告
        report = self.generate_evaluation_report(metrics)
        # 转换NumPy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        report = convert_to_native(report)
        report_msg = String()
        report_msg.data = json.dumps(report, indent=2)
        self.report_pub.publish(report_msg)
    
    def run_evaluation(self):
        """运行评估并发布结果"""
        current_time = time.time()
        
        # 每5秒评估一次
        if current_time - self.last_evaluation_time > 5.0:
            metrics = self.evaluate_all_metrics()
            
            if metrics:
                self.publish_metrics(metrics)

                # 记录到CSV文件
                self.log_metrics_to_csv(metrics)
                
                # 记录详细检测数据
                boxes, types, scores = self.detection_history.get_all_history()
                if boxes and types and scores:
                    self.log_detections_to_csv(boxes, types, scores)
                
                # 记录趋势分析
                trends = self.get_enhanced_trend_analysis()
                if trends:
                    self.log_trends_to_csv(trends)
                
                # 打印评估结果
                rospy.loginfo(f"Detection Quality Evaluation:")
                rospy.loginfo(f"  Overall Quality: {metrics['overall_quality']:.3f}")
                rospy.loginfo(f"  Position Reasonableness: {metrics['position_reasonableness']:.3f}")
                rospy.loginfo(f"  Spatial Consistency: {metrics['spatial_consistency']:.3f}")
                rospy.loginfo(f"  Size Reasonableness: {metrics['size_reasonableness']:.3f}")
                rospy.loginfo(f"  Score Consistency: {metrics['score_consistency']:.3f}")
                rospy.loginfo(f"  Detection Density: {metrics['detection_density']:.3f}")
                rospy.loginfo(f"  Class Distribution: {metrics['class_distribution']:.3f}")
                
                # 如果质量分数过低，发出警告
                if metrics['overall_quality'] < 0.4:
                    rospy.logwarn("Detection quality is low! Please check system parameters.")
            
            self.last_evaluation_time = current_time

    def get_enhanced_trend_analysis(self):
        """增强的趋势分析"""
        if len(self.metrics_history) < 5:
            return None
        
        recent_metrics = list(self.metrics_history)[-10:]  # 取最近10次评估
        
        trends = {}
        metric_names = ['overall_quality', 'position_reasonableness', 'spatial_consistency', 
                       'score_consistency', 'size_reasonableness', 'detection_density']
        
        for metric_name in metric_names:
            values = [m['metrics'].get(metric_name, 0) for m in recent_metrics]
            
            if len(values) > 1:
                # 计算趋势（线性回归斜率）
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                # 计算移动平均
                moving_avg = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
                
                # 计算波动性（标准差）
                volatility = np.std(values)
                
                # 确定趋势方向
                if slope > 0.01:
                    direction = 'improving'
                elif slope < -0.01:
                    direction = 'declining'
                else:
                    direction = 'stable'
                
                trends[metric_name] = {
                    'direction': direction,
                    'slope': float(slope),
                    'current_value': float(values[-1]),
                    'moving_average': float(moving_avg),
                    'volatility': float(volatility)
                }
        
        return trends

    def get_trend_analysis(self):
        """分析指标趋势"""
        if len(self.metrics_history) < 5:
            return None
        
        recent_metrics = list(self.metrics_history)[-10:]  # 取最近10次评估
        
        trends = {}
        for metric_name in ['overall_quality', 'stability_score', 'spatial_consistency']:
            values = [m['metrics'].get(metric_name, 0) for m in recent_metrics]
            
            # 计算趋势（线性回归斜率的符号）
            x = np.arange(len(values))
            if len(values) > 1:
                slope = np.polyfit(x, values, 1)[0]
                trends[metric_name] = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
            else:
                trends[metric_name] = 'unknown'
        
        return trends
    
    def export_summary_report(self, output_path=None):
        """导出汇总报告到CSV"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.csv_log_dir, f"summary_report_{timestamp}.csv")
        
        try:
            if not self.metrics_history:
                rospy.logwarn("No metrics history available for summary report")
                return
            
            # 准备汇总数据
            summary_data = []
            for entry in self.metrics_history:
                metrics = entry['metrics']
                timestamp = entry['timestamp']
                datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                summary_data.append({
                    'timestamp': timestamp,
                    'datetime': datetime_str,
                    'overall_quality': metrics.get('overall_quality', 0.0),
                    'position_reasonableness': metrics.get('position_reasonableness', 0.0),
                    'spatial_consistency': metrics.get('spatial_consistency', 0.0),
                    'score_consistency': metrics.get('score_consistency', 0.0),
                    'size_reasonableness': metrics.get('size_reasonableness', 0.0),
                    'detection_density': metrics.get('detection_density', 0.0),
                    'class_distribution': metrics.get('class_distribution', 0.0),
                    'session_id': self.session_id
                })
            
            # 写入CSV文件
            if summary_data:
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = summary_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_data)
                
                rospy.loginfo(f"Summary report exported to: {output_path}")
            
        except Exception as e:
            rospy.logerr(f"Failed to export summary report: {e}")


# 使用示例类
class DetectionEvaluationNode:
    def __init__(self):
        rospy.init_node('detection_evaluation_node')
        
        # 初始化DetectionHistory
        self.detection_history = DetectionHistory(max_len=100, distance_threshold=2.0)
        
        # 初始化评估器
        self.evaluator = HistoryBasedEvaluator(self.detection_history)
        
        # 模拟检测结果订阅（替换为你的实际话题）
        # self.detection_sub = rospy.Subscriber('/detection_results', YourMsgType, self.detection_callback)
        
        # 评估定时器
        self.eval_timer = rospy.Timer(rospy.Duration(1.0), self.evaluation_callback)
        
        rospy.loginfo("Detection Evaluation Node initialized")
    
    def detection_callback(self, msg):
        """处理检测结果回调（需要根据你的消息类型调整）"""
        # 从消息中提取boxes, types, scores
        # boxes = extract_boxes_from_msg(msg)
        # types = extract_types_from_msg(msg)
        # scores = extract_scores_from_msg(msg)
        
        # 更新历史记录
        # self.detection_history.update(boxes, types, scores)
        pass
    
    def evaluation_callback(self, event):
        """定时评估回调"""
        self.evaluator.run_evaluation()


if __name__ == '__main__':
    try:
        node = DetectionEvaluationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass