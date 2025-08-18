from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения с настройками CORS
app = Flask(__name__)
CORS(app, resources={
    r"/analyze": {
        "origins": ["https://pordirador.github.io"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

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
    def __init__(self):
        self.current_image_height = 0
        self.current_image_width = 0
        self.metrics = self._init_metrics()

    def _landmark_to_pixel(self, lm, image_shape):
        """Конвертирует нормализованные координаты landmark в пиксельные координаты"""
        h, w = image_shape[:2]
        return int(lm.x * w), int(lm.y * h)

    def _init_metrics(self):
        """Инициализация всех метрик для анализа лица"""
        return {
            "jaw_to_cheek_ratio": {
                "points": (58, 288, 34, 264),
                "calculate": self._calc_length_ratio,
                "ideal_range": (0.85, 0.92),
                "description": "Соотношение челюсти к скулам (0.85-0.92)",
            },
            "jaw_angle": {
                "points": (),
                "calculate": self._calc_mandible_angle,
                "ideal_range": (70, 90),
                "description": "Угол мандибулы (70°–90°)",
            },
            "eye_angle": {
                "points": (33, 2, 263),
                "calculate": self._calc_eye_angle,
                "ideal_range": (85, 95),
                "description": "Угол между глазами через нос (85°-95°)",
            },
            "pupil_to_cheek_ratio": {
                "points": (468, 473, 34, 264),
                "calculate": self._calc_length_ratio,
                "ideal_range": (0.445, 0.477),
                "description": "Отношение зрачков к скулам (0.445-0.477)",
            },
            "eye_shape_ratio": {
                "points": (133, 33, 159, 145),
                "calculate": lambda p1, p2, p3, p4: self._calc_length(p1, p2) / self._calc_length(p3, p4) + 1,
                "ideal_range": (2.8, 3.8),
                "description": "Разрез глаз (длина/высота, 2.8-3.8)",
            }
        }

    def _calc_length(self, p1, p2):
        """Вычисляет расстояние между двумя точками"""
        return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

    def _calc_length_ratio(self, p1, p2, p3, p4):
        """Вычисляет соотношение длин между двумя парами точек"""
        return self._calc_length(p1, p2) / self._calc_length(p3, p4)

    def _calc_mandible_angle(self, a1, a2, b1, b2):
        """Вычисляет угол мандибулы"""
        vec1 = np.array([a2.x - a1.x, a2.y - a1.y])
        vec2 = np.array([b2.x - b1.x, b2.y - b1.y])

        cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))
        return 180 - angle

    def _calc_eye_angle(self, left_eye, nose, right_eye):
        """Вычисляет угол между глазами"""
        vec_left = np.array([left_eye.x - nose.x, left_eye.y - nose.y])
        vec_right = np.array([right_eye.x - nose.x, right_eye.y - nose.y])

        norm_left = np.linalg.norm(vec_left)
        norm_right = np.linalg.norm(vec_right)
        if norm_left == 0 or norm_right == 0:
            return 0.0

        cosine = np.dot(vec_left, vec_right) / (norm_left * norm_right)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def _is_in_ideal_range(self, value, ideal_range):
        """Проверяет, находится ли значение в идеальном диапазоне"""
        if ideal_range is None:
            return True
        if isinstance(value, tuple):
            return all(ideal_range[i][0] <= v <= ideal_range[i][1] for i, v in enumerate(value))
        return ideal_range[0] <= value <= ideal_range[1]

    def analyze_face(self, image):
        """Анализирует лицо на изображении"""
        try:
            # Конвертируем изображение в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Получаем landmarks
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                logger.warning("No faces detected in the image")
                return None
                
            landmarks = results.multi_face_landmarks[0].landmark
            self.current_image_height, self.current_image_width = image.shape[:2]
            
            # Анализируем landmarks
            analysis_results = []
            for key, metric in self.metrics.items():
                try:
                    if key == "jaw_angle":
                        a1_idx, a2_idx, b1_idx, b2_idx = self._find_lowest_mandible_points(landmarks)
                        points = [landmarks[i] for i in (a1_idx, a2_idx, b1_idx, b2_idx)]
                        value = metric["calculate"](*points)
                    else:
                        points = [landmarks[i] for i in metric["points"]]
                        value = metric["calculate"](*points)

                    status = "ok" if self._is_in_ideal_range(value, metric.get("ideal_range")) else "bad"
                    value_str = ", ".join(f"{v:.3f}" for v in value) if isinstance(value, tuple) else f"{value:.3f}"
                    
                    analysis_results.append({
                        "name": key,
                        "value": value,
                        "value_str": value_str,
                        "status": status,
                        "description": metric['description'],
                        "ideal_range": metric.get('ideal_range'),
                        "points": metric["points"]
                    })

                except Exception as e:
                    logger.error(f"Error calculating metric {key}: {str(e)}")
                    analysis_results.append({
                        "name": key,
                        "value": None,
                        "status": "error",
                        "message": str(e),
                        "description": metric['description'],
                        "ideal_range": metric.get('ideal_range'),
                        "points": metric["points"]
                    })
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            return None

    def _find_lowest_mandible_points(self, landmarks):
        """Находит ключевые точки для анализа мандибулы"""
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

# Создаем экземпляр анализатора
analyzer = FaceAnalyzer()

@app.route('/')
def serve_frontend():
    """Отдает фронтенд приложение"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Отдает статические файлы фронтенда"""
    return send_from_directory('../frontend', path)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    """API endpoint для анализа изображения"""
    if request.method == 'OPTIONS':
        # Предварительный CORS запрос
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "https://pordirador.github.io")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    try:
        # Проверяем наличие изображения
        if not request.json or 'image' not in request.json:
            logger.warning("No image data received")
            return jsonify({"error": "No image provided"}), 400
            
        # Декодируем изображение
        image_data = request.json['image'].split(',')[1]  # Удаляем префикс data:image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Failed to decode image")
            return jsonify({"error": "Invalid image data"}), 400
        
        # Анализируем лицо
        results = analyzer.analyze_face(img)
        
        if results is None:
            logger.warning("No face detected in the image")
            return jsonify({"error": "No face detected"}), 400
            
        response = jsonify({"results": results})
        response.headers.add("Access-Control-Allow-Origin", "https://pordirador.github.io")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
