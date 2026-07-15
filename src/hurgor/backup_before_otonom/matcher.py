import cv2
import numpy as np

class ReferenceMatcher:
    def __init__(self, reference_image_path):
        # Türkçe karakter sorununa karşı numpy ile bayt okuması yapıyoruz
        img_array = np.fromfile(reference_image_path, np.uint8)
        self.ref_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if self.ref_img is None:
            print(f"[WARN] Referans resim yuklenemedi: {reference_image_path}")
        
        # ORB dedektörü (Saniyede 30 FPS çalışacak kadar hafiftir, Edge cihaz dostudur)
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Referans resmin özelliklerini (keypoints) sadece 1 KERE hesapla ve hafızada tut
        if self.ref_img is not None:
            self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_img, None)
        else:
            self.kp_ref, self.des_ref = None, None
            
        # Özellik Eşleştirici (Brute-Force Hamming Distance)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.MIN_MATCH_COUNT = 12 # Minimum kaç iyi eşleşme olursa "hedefi buldum" desin?

    def find_reference(self, frame):
        if self.des_ref is None:
            return None

        # Gelen canlı kareyi griye çevir ve özelliklerini çıkar
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)
        
        if des_frame is None:
            return None
            
        # Şablon ile canlı kareyi eşleştir
        matches = self.bf.match(self.des_ref, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # En iyi eşleşmeleri al
        good_matches = matches[:50] 
        
        if len(good_matches) > self.MIN_MATCH_COUNT:
            # Eşleşen noktaların piksellerini al
            src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # RANSAC ile Homografi matrisini hesapla (Hatalı eşleşmeleri eler, 3D perspektifi hesaplar)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                h, w = self.ref_img.shape
                # Referans resmin 4 köşesini canlı görüntüye iz düşür
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Sınır kutusunu (Bounding Box) hesapla [x, y, w, h]
                x_min, y_min = np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1])
                x_max, y_max = np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1])
                
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                w_box = x_max - x_min
                h_box = y_max - y_min
                
                # Güven skorunu eşleşme başarısına göre oranla
                confidence = min(len(good_matches) / 50.0, 1.0)
                
                return {
                    "class": "goal", # Şartnamede referans hedef genellikle "goal" olarak adlandırılır
                    "bbox": [float(x_center), float(y_center), float(w_box), float(h_box)],
                    "conf": float(confidence)
                }
        return None