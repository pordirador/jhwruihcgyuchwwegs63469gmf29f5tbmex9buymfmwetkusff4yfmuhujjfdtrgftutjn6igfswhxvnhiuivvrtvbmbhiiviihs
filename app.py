# app.py (бэкенд)
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import os

app = Flask(__name__)

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class FaceAnalyzer:
    def _landmark_to_pixel(self, lm, image_shape):
        h, w = image_shape[:2]
        return int(lm.x * w), int(lm.y * h)

    def find_lowest_mandible_points(self, landmarks):
        upper_left_idx = 172
        upper_right_idx = 397
        lower_left_candidates = [148, 176, 149, 150]
        lower_right_candidates = [382, 400, 381, 380]

        upper_left_y = landmarks[upper_left_idx].y
        upper_right_y = landmarks[upper_right_idx].y

        lower_left_idx = max(
            (i for i in lower_left_candidates if landmarks[i].y > upper_left_y),
            key=lambda i: landmarks[i].y,
            default=upper_left_idx,
        )
        lower_right_idx = max(
            (i for i in lower_right_candidates if landmarks[i].y > upper_right_y),
            key=lambda i: landmarks[i].y,
            default=upper_right_idx,
        )

        return upper_left_idx, lower_left_idx, upper_right_idx, lower_right_idx

    def analyze_landmarks(self, landmarks):
        results = []
        for key, metric in self.metrics.items():
            try:
                if key == "jaw_angle":
                    a1_idx, a2_idx, b1_idx, b2_idx = self.find_lowest_mandible_points(landmarks)
                    points = [landmarks[i] for i in (a1_idx, a2_idx, b1_idx, b2_idx)]
                    value = metric["calculate"](*points)
                elif key == "eye_angle":
                    points = [landmarks[i] for i in metric["points"]]
                    value = metric["calculate"](*points)
                else:
                    points = [landmarks[i] for i in metric["points"]]
                    value = metric["calculate"](*points)

                status = "ok" if self.is_in_ideal_range(value, metric.get("ideal_range")) else "bad"
                value_str = ", ".join(f"{v:.3f}" for v in value) if isinstance(value, tuple) else f"{value:.3f}"
                
                results.append({
                    "name": key,
                    "value": value,
                    "value_str": value_str,
                    "status": status,
                    "description": metric['description'],
                    "ideal_range": metric.get('ideal_range'),
                    "points": metric["points"]
                })

            except Exception as e:
                results.append({
                    "name": key,
                    "value": None,
                    "status": "error",
                    "message": str(e),
                    "description": metric['description'],
                    "ideal_range": metric.get('ideal_range'),
                    "points": metric["points"]
                })
        return results

    def is_in_ideal_range(self, value, ideal_range):
        if ideal_range is None:
            return True
        if isinstance(value, tuple):
            return all(ideal_range[i][0] <= v <= ideal_range[i][1] for i, v in enumerate(value))
        return ideal_range[0] <= value <= ideal_range[1]

    def __init__(self):
        self.current_image_height = 0
        self.current_image_width = 0
        self.metrics = self.init_metrics()

    def init_metrics(self):
        return {
            "jaw_to_cheek_ratio": {
                "points": (58, 288, 34, 264),
                "calculate": self.calc_length_ratio,
                "ideal_range": (0.85, 0.92),
                "description": "Соотношение челюсти к скулам (0.85-0.92)",
            },
            "jaw_angle": {
                "points": (),
                "calculate": self.calc_mandible_angle,
                "ideal_range": (70, 90),
                "description": "Угол мандибулы (70°–90°)",
            },
            "eye_angle": {
                "points": (33, 2, 263),
                "calculate": self.calc_eye_angle,
                "ideal_range": (85, 95),
                "description": "Угол между глазами через нос (85°-95°)",
            },
            "pupil_to_cheek_ratio": {
                "points": (468, 473, 34, 264),
                "calculate": self.calc_length_ratio,
                "ideal_range": (0.445, 0.477),
                "description": "Отношение зрачков к скулам (0.445-0.477)",
            },
            "eye_shape_ratio": {
                "points": (133, 33, 159, 145),
                "calculate": lambda p1, p2, p3, p4: self.calc_length(p1, p2) / self.calc_length(p3, p4) + 1,
                "ideal_range": (2.8, 3.8),
                "description": "Разрез глаз (длина/высота, 2.8-3.8)",
            },
            "philtrum_chin_ratio": {
                "points": (2, 13, 17, 152),
                "calculate": self.calc_philtrum_chin_ratio_corrected,
                "ideal_range": (2.0, 2.5),
                "description": "Фильтрум-чин (2.0-2.5)",
            },
            "face_horizontal_ratio": {
    "points": (10, 9, 2, 152, 151),  # лоб(10), переносица(9), нос(2), подбородок(152), глабелла(151)
    "calculate": self.calc_face_horizontal_ratio,
    "ideal_range": ((0.30, 0.36), (0.30, 0.36), (0.30, 0.36)),
    "description": "Face Horizontal Ratio (3 части: верх-лоб, лоб-нос, нос-подбородок)",
},
            "face_vertical_ratio": {
                "points": (127, 33, 133, 362, 263, 356),
                "calculate": self.calc_face_vertical_ratio,
                "ideal_range": ((0.18, 0.22),) * 5,
                "description": "Face Vertical Ratio (5 частей, по горизонтали)",
            },
            "face_aspect_ratio": {
    "points": (34, 263, 10, 152, 58, 288, 151),  # точки: скулы, челюсть, лоб, подбородок, переносица
    "calculate": self.calc_face_aspect_ratio,
    "ideal_range": (1.3, 1.5),
    "description": "Face Aspect Ratio (height/max_width)",
},
        }
        

    def calc_length(self, p1, p2):
        return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

    def calc_length_ratio(self, p1, p2, p3, p4):
        return self.calc_length(p1, p2) / self.calc_length(p3, p4)

    def calc_philtrum_chin_ratio_corrected(self, p1, p2, p3, p4):
        philtrum_length = self.calc_length(p1, p2) * 0.75
        chin_length = self.calc_length(p3, p4)
        return chin_length / philtrum_length if philtrum_length != 0 else 0.0

    def calc_mandible_angle(self, a1, a2, b1, b2):
        vec1 = np.array([a2.x - a1.x, a2.y - a1.y])
        vec2 = np.array([b2.x - b1.x, b2.y - b1.y])

        cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))
        return 180 - angle

    def calc_eye_angle(self, left_eye, nose, right_eye):
        vec_left = np.array([left_eye.x - nose.x, left_eye.y - nose.y])
        vec_right = np.array([right_eye.x - nose.x, right_eye.y - nose.y])
    
        norm_left = np.linalg.norm(vec_left)
        norm_right = np.linalg.norm(vec_right)
        if norm_left == 0 or norm_right == 0:
            return 0.0
    
        cosine = np.dot(vec_left, vec_right) / (norm_left * norm_right)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def calc_eye_angle_pixel(self, left_eye, nose, right_eye):
        vec_left = np.array([left_eye[0]-nose[0], left_eye[1]-nose[1]])
        vec_right = np.array([right_eye[0]-nose[0], right_eye[1]-nose[1]])

        norm_left = np.linalg.norm(vec_left)
        norm_right = np.linalg.norm(vec_right)

        if norm_left == 0 or norm_right == 0:
            return 0.0

        cosine = np.dot(vec_left, vec_right) / (norm_left * norm_right)
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))

    def calc_face_horizontal_ratio(self, lm_forehead, lm_nose_bridge, lm_nose, lm_chin, _):
        # Убрали последний параметр, так как он больше не нужен
        h, w = self.current_image_height, self.current_image_width
        
        if h == 0 or w == 0:
            print("Ошибка: нулевые размеры изображения")
            return (0.0, 0.0, 0.0)
        
        try:
            forehead = self._landmark_to_pixel(lm_forehead, (h, w))
            nose_bridge = self._landmark_to_pixel(lm_nose_bridge, (h, w))
            nose = self._landmark_to_pixel(lm_nose, (h, w))
            chin = self._landmark_to_pixel(lm_chin, (h, w))
        except Exception as e:
            print(f"Ошибка конвертации landmarks: {str(e)}")
            return (0.0, 0.0, 0.0)

        # Вектор от лба к переносице (теперь используем nose_bridge вместо glabella)
        vec_fb = np.array([nose_bridge[0] - forehead[0], nose_bridge[1] - forehead[1]])
        norm = np.linalg.norm(vec_fb)
        
        if norm < 1e-6:
            print("Ошибка: forehead и nose_bridge совпадают!")
            return (0.0, 0.0, 0.0)
        
        vec_norm = vec_fb / norm
        dist_nose_bridge_to_forehead = np.sqrt((forehead[0] - nose_bridge[0])**2 +
                                        (forehead[1] - nose_bridge[1])**2)
        
        # Виртуальная точка линии роста волос
        virtual_hairline = (
            int(forehead[0] - vec_norm[0] * dist_nose_bridge_to_forehead),
            int(forehead[1] - vec_norm[1] * dist_nose_bridge_to_forehead)
        )
        
        # Расстояния между ключевыми точками
        d1 = np.sqrt((nose_bridge[0] - virtual_hairline[0])**2 + 
                    (nose_bridge[1] - virtual_hairline[1])**2)  # hairline to brow
        d2 = np.sqrt((nose[0] - nose_bridge[0])**2 + 
                    (nose[1] - nose_bridge[1])**2)  # brow to nose
        d3 = np.sqrt((chin[0] - nose[0])**2 + 
                    (chin[1] - nose[1])**2)  # nose to chin
        
        total = d1 + d2 + d3
        if total == 0:
            return (0.0, 0.0, 0.0)
        
        return (d1/total, d2/total, d3/total)

    def calc_face_vertical_ratio(self, lm_left_face, lm_outer_left_eye, lm_inner_left_eye,
                              lm_inner_right_eye, lm_outer_right_eye, lm_right_face):
        d1 = abs(lm_outer_left_eye.x - lm_inner_left_eye.x)
        d2 = abs(lm_inner_left_eye.x - lm_inner_right_eye.x)
        d3 = abs(lm_inner_right_eye.x - lm_outer_right_eye.x)
        d4 = abs(lm_outer_left_eye.x - lm_left_face.x)
        d5 = abs(lm_right_face.x - lm_outer_right_eye.x)
        total = d1 + d2 + d3 + d4 + d5
        return (d1/total, d2/total, d3/total, d4/total, d5/total) if total != 0 else (0,0,0,0,0)

    def calc_face_aspect_ratio(self, p_left_cheek, p_right_cheek, p_forehead, p_chin, p_left_jaw, p_right_jaw, p_nose_bridge):
    # Используем сохраненные размеры изображения
        h, w = self.current_image_height, self.current_image_width
    
        if h == 0 or w == 0:
            print("Ошибка: нулевые размеры изображения")
            return 0.0
        
        try:
            # Конвертируем landmarks в пиксели
            forehead = self._landmark_to_pixel(p_forehead, (h, w))
            chin = self._landmark_to_pixel(p_chin, (h, w))
            left_cheek = self._landmark_to_pixel(p_left_cheek, (h, w))
            right_cheek = self._landmark_to_pixel(p_right_cheek, (h, w))
            left_jaw = self._landmark_to_pixel(p_left_jaw, (h, w))
            right_jaw = self._landmark_to_pixel(p_right_jaw, (h, w))
            nose_bridge = self._landmark_to_pixel(p_nose_bridge, (h, w))
        except Exception as e:
            print(f"Ошибка конвертации landmarks: {str(e)}")
            return 0.0
        
        # Высота лица (от подбородка до виртуальной точки над лбом)
        chin_to_forehead = np.sqrt((forehead[0] - chin[0])**2 + (forehead[1] - chin[1])**2)
        vertical_dist = abs(nose_bridge[1] - forehead[1])
        face_height = chin_to_forehead + vertical_dist
        
        # Ширина лица (максимальная между скулами и челюстью)
        cheek_width = abs(right_cheek[0] - left_cheek[0])
        jaw_width = abs(right_jaw[0] - left_jaw[0])
        face_width = max(cheek_width, jaw_width)
        
        if face_width == 0:
            print("Ошибка: нулевая ширина лица")
            return 0.0
        
        ratio = face_height / face_width
        return ratio

    def analyze_face(self, image):
        try:
            # Конвертируем изображение в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Получаем landmarks
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            landmarks = results.multi_face_landmarks[0].landmark
            self.current_image_height, self.current_image_width = image.shape[:2]
            
            # Анализируем лицо
            return self.analyze_landmarks(landmarks)
            
        except Exception as e:
            print(f"Error analyzing face: {str(e)}")
            return None

    def analyze_landmarks(self, landmarks):
        results = []
        for key, metric in self.metrics.items():
            try:
                # ... логика расчета каждой метрики ...
                pass
            except Exception as e:
                results.append({
                    "name": key,
                    "value": None,
                    "status": "error",
                    "message": str(e),
                    "description": metric['description'],
                    "ideal_range": metric.get('ideal_range'),
                    "points": metric["points"]
                })
        return results

analyzer = FaceAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Получаем изображение от фронтенда
        data = request.json
        image_data = data['image'].split(',')[1]  # Удаляем префикс data:image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Анализируем лицо
        results = analyzer.analyze_face(img)
        
        if results is None:
            return jsonify({"error": "No face detected"}), 400
            
        return jsonify({"results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
