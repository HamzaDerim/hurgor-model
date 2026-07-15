import numpy as np
import pandas as pd
import math
import cv2
from ultralytics import YOLO
from src.hurgor.matcher import ReferenceMatcher

class HurgorVision:
    def __init__(self, model_path="models/hurgor_final.onnx"):
        # 1. YOLO MODELİ ENTEGRASYONU (Yeni Eklendi)
        self.model = YOLO(model_path, task="detect")
        # Referans Nesne Eşleştirici
        self.ref_matcher = ReferenceMatcher("assets/references/Referans_Nesne_01.JPG")
        # 2. MEVCUT KAMERA VE MATEMATİK PARAMETRELERİN (Korundu)
        self.fx = 960.0
        self.fy = 960.0
        self.cx = 960.0
        self.cy = 540.0
        
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ], dtype=np.float32)
        
        self.K_inv = np.linalg.inv(self.K)
        self.dist_coeffs = np.array([-0.25, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        self.telemetry_data = {}

    def load_telemetry(self, csv_path):
        # (MEVCUT KOD - KORUNDU)
        try:
            df = pd.read_csv(csv_path)
            for index, row in df.iterrows():
                frame_id = int(row['Frame_ID'])
                self.telemetry_data[frame_id] = {
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude']),
                    'alt': float(row['Altitude']),
                    'roll': math.radians(float(row['Roll'])),
                    'pitch': math.radians(float(row['Pitch'])),
                    'yaw': math.radians(float(row['Yaw']))
                }
        except Exception as e:
            pass

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        # (MEVCUT KOD - KORUNDU)
        R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        return np.dot(R_z, np.dot(R_y, R_x))

    def get_3d_coordinates(self, u, v, frame_id, z_depth_estimate=25.0):
        # (MEVCUT KOD - KORUNDU)
        pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist_coeffs)
        x_norm, y_norm = undistorted[0][0]
        
        p_cam = np.array([x_norm, y_norm, 1.0])
        
        if frame_id not in self.telemetry_data:
            rel_x = p_cam[0] * z_depth_estimate
            rel_y = p_cam[1] * z_depth_estimate
            return float(rel_x), float(rel_y), 0.0

        telemetry = self.telemetry_data[frame_id]
        Z_iha = telemetry['alt']
        R = self.euler_to_rotation_matrix(telemetry['roll'], telemetry['pitch'], telemetry['yaw'])
        X_iha = telemetry['lon']
        Y_iha = telemetry['lat']

        d_world = np.dot(R, p_cam)
        dx, dy, dz = d_world[0], d_world[1], d_world[2]
        
        if dz >= 0:
            return 0.0, 0.0, 0.0 
            
        s = -Z_iha / dz
        X_target = X_iha + s * dx
        Y_target = Y_iha + s * dy
        
        return float(X_target), float(Y_target), 0.0

    def add_meters_to_gps(self, lat, lon, dx, dy):
        # (MEVCUT KOD - KORUNDU)
        R_earth = 6378137.0 
        delta_lat = (dy / R_earth) * (180.0 / math.pi)
        delta_lon = (dx / (R_earth * math.cos(math.pi * lat / 180.0))) * (180.0 / math.pi)
        return lat + delta_lat, lon + delta_lon

    # ==========================================
    # YENİ EKLENEN KISIM: YOLO INFERENCE DÖNGÜSÜ
    # ==========================================
    def run_inference(self, frame, frame_id=None):
        """
        Görüntüyü YOLO'ya sokar, nesneleri bulur ve 3D koordinatlarını hesaplar.
        threaded_pipeline.py veya inference.py buradan beslenecek.
        """
        results = self.model.predict(source=frame, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 2D Bounding Box merkezini (x,y) al
                x, y, w, h = box.xywh[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = self.model.names[cls_id]
                
                # 3D Uzaydaki konumunu hesapla (telemetri varsa tam, yoksa tahmini)
                rel_x, rel_y, _ = self.get_3d_coordinates(x, y, frame_id) if frame_id else (0.0, 0.0, 0.0)
                
                detections.append({
                    'class': cls_name,
                    'bbox': [x, y, w, h],
                    'rel_x': rel_x,
                    'rel_y': rel_y,
                    'conf': conf
                })
        # ==========================================
        # GÖREV 3: REFERANS NESNE (Şablon Eşleme) ARAMASI
        # ==========================================
        ref_detection = self.ref_matcher.find_reference(frame)
        if ref_detection:
            # Bulduğu hedefin de 3D metrik koordinatını hesapla!
            ref_x, ref_y, _ = self.get_3d_coordinates(
                ref_detection['bbox'][0], 
                ref_detection['bbox'][1], 
                frame_id
            ) if frame_id else (0.0, 0.0, 0.0)
            
            ref_detection['rel_x'] = ref_x
            ref_detection['rel_y'] = ref_y
            
            detections.append(ref_detection)
                
        return detections


# MEVCUT SINIFLAR - TAMAMEN KORUNDU
class VisualOdometry:
    def __init__(self):
        self.uav_x = 0.0
        self.uav_y = 0.0
        self.prev_anchors = {}
        self.valid_anchors = ['goal', 'car'] 

    def update_position(self, current_detections):
        movement_x = 0.0
        movement_y = 0.0
        anchor_found = False

        for det in current_detections:
            cls_name = det['class']
            if cls_name in self.valid_anchors:
                curr_rel_x = det['rel_x']
                curr_rel_y = det['rel_y']

                if cls_name in self.prev_anchors:
                    prev_rel_x, prev_rel_y = self.prev_anchors[cls_name]
                    movement_x = prev_rel_x - curr_rel_x
                    movement_y = prev_rel_y - curr_rel_y
                    anchor_found = True
                    break 

        if anchor_found:
            self.uav_x += movement_x
            self.uav_y += movement_y

        self.prev_anchors.clear()
        for det in current_detections:
            if det['class'] in self.valid_anchors:
                self.prev_anchors[det['class']] = (det['rel_x'], det['rel_y'])

        return self.uav_x, self.uav_y

class TargetTracker:
    def __init__(self, fps=30.0):
        self.fps = fps
        self.tracks = {}  
        self.next_id = 1
        self.distance_threshold = 10.0 

    def update(self, detections, frame_id, uav_x, uav_y):
        tracked_detections = []
        for det in detections:
            target_global_x = uav_x + det['rel_x']
            target_global_y = uav_y + det['rel_y']
            
            best_match_id = None
            min_dist = self.distance_threshold
            
            for track_id, track_data in self.tracks.items():
                dist = math.hypot(target_global_x - track_data['x'], target_global_y - track_data['y'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = track_id
            
            if best_match_id is None:
                best_match_id = self.next_id
                self.next_id += 1
                speed_kmh = 0.0
            else:
                prev_track = self.tracks[best_match_id]
                dx = target_global_x - prev_track['x']
                dy = target_global_y - prev_track['y']
                dframes = frame_id - prev_track['frame_id']
                
                if dframes > 0:
                    dt = dframes / self.fps 
                    vx = dx / dt 
                    vy = dy / dt 
                    speed_ms = math.hypot(vx, vy)
                    speed_kmh = speed_ms * 3.6 
                else:
                    speed_kmh = prev_track.get('speed', 0.0)
            
            self.tracks[best_match_id] = {
                'x': target_global_x,
                'y': target_global_y,
                'frame_id': frame_id,
                'speed': speed_kmh
            }
            
            det['track_id'] = best_match_id
            det['speed_kmh'] = speed_kmh
            tracked_detections.append(det)
            
        return tracked_detections