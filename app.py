from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Разрешаем запросы с вашего GitHub Pages и других доменов
CORS(app, origins=[
    "https://pordirador.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Добавьте другие домены при необходимости
])

class FaceAnalyzer:
    def __init__(self):
        self.metrics = self._init_metrics()

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
                "description": "Угол мандибулы (85°–95°)",
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
                "calculate": lambda p1, p2, p3, p4: self._calc_length(p1, p2) / self._calc_length(p3, p4),
                "ideal_range": (2.8, 3.8),
                "description": "Разрез глаз (длина/высота, 2.8-3.8)",
            },
            "media_canthal_angle": {
                "points": (362, 386, 374, 133, 153, 145),  # Левый + правый глаз
                "calculate": self._calc_media_canthal_angle,
                "ideal_range": (20, 42),
                "description": "Медиальный кантальный угол (35°-70°)",
            }
        }

    def _calc_length(self, p1, p2):
        """Вычисляет расстояние между двумя точками"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _calc_length_ratio(self, p1, p2, p3, p4):
        """Вычисляет соотношение длин между двумя парами точек"""
        return self._calc_length(p1, p2) / self._calc_length(p3, p4)

    def _calc_mandible_angle(self, a1, a2, b1, b2):
        """Вычисляет угол мандибулы"""
        vec1 = np.array([a2[0] - a1[0], a2[1] - a1[1]])
        vec2 = np.array([b2[0] - b1[0], b2[1] - b1[1]])

        cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))
        return 180 - angle

    def _calc_eye_angle(self, left_eye, nose, right_eye):
        """Вычисляет угол между глазами"""
        vec_left = np.array([left_eye[0] - nose[0], left_eye[1] - nose[1]])
        vec_right = np.array([right_eye[0] - nose[0], right_eye[1] - nose[1]])

        norm_left = np.linalg.norm(vec_left)
        norm_right = np.linalg.norm(vec_right)
        if norm_left == 0 or norm_right == 0:
            return 0.0

        cosine = np.dot(vec_left, vec_right) / (norm_left * norm_right)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def _calc_canthus_angle(self, canthus_point, upper_point, lower_point):
        """Вычисляет угол кантуса для одного глаза"""
        vec_upper = np.array([upper_point[0] - canthus_point[0], upper_point[1] - canthus_point[1]])
        vec_lower = np.array([lower_point[0] - canthus_point[0], lower_point[1] - canthus_point[1]])

        norm_upper = np.linalg.norm(vec_upper)
        norm_lower = np.linalg.norm(vec_lower)
        if norm_upper == 0 or norm_lower == 0:
            return 0.0

        cosine = np.dot(vec_upper, vec_lower) / (norm_upper * norm_lower)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def _calc_media_canthal_angle(self, left_canthus, left_upper, left_lower, right_canthus, right_upper, right_lower):
        """Вычисляет медиальный кантальный угол для обоих глаз"""
        left_angle = self._calc_canthus_angle(left_canthus, left_upper, left_lower)
        right_angle = self._calc_canthus_angle(right_canthus, right_upper, right_lower)
        
        # Возвращаем среднее значение углов обоих глаз
        return (left_angle + right_angle) / 2

    def _is_in_ideal_range(self, value, ideal_range):
        """Проверяет, находится ли значение в идеальном диапазоне"""
        if ideal_range is None:
            return True
        return ideal_range[0] <= value <= ideal_range[1]

    def analyze_landmarks(self, landmarks, image_width, image_height):
        """Анализирует landmarks лица"""
        try:
            # Нормализуем координаты landmarks
            normalized_landmarks = []
            for lm in landmarks:
                x = lm['x'] * image_width
                y = lm['y'] * image_height
                normalized_landmarks.append((x, y))

            analysis_results = []
            for key, metric in self.metrics.items():
                try:
                    if key == "jaw_angle":
                        # Находим точки для анализа мандибулы
                        upper_left_idx, lower_left_idx, upper_right_idx, lower_right_idx = self._find_lowest_mandible_points(normalized_landmarks)
                        points = [
                            normalized_landmarks[upper_left_idx],
                            normalized_landmarks[lower_left_idx],
                            normalized_landmarks[upper_right_idx],
                            normalized_landmarks[lower_right_idx]
                        ]
                        value = metric["calculate"](*points)
                    else:
                        points = [normalized_landmarks[i] for i in metric["points"]]
                        value = metric["calculate"](*points)

                    status = "ok" if self._is_in_ideal_range(value, metric.get("ideal_range")) else "bad"
                    value_str = f"{value:.3f}"

                    analysis_results.append({
                        "name": key,
                        "value": value,
                        "value_str": value_str,
                        "status": status,
                        "description": metric['description'],
                        "ideal_range": metric.get('ideal_range')
                    })

                except Exception as e:
                    logger.error(f"Error calculating metric {key}: {str(e)}")
                    analysis_results.append({
                        "name": key,
                        "value": None,
                        "status": "error",
                        "message": str(e),
                        "description": metric['description'],
                        "ideal_range": metric.get('ideal_range')
                    })

            return analysis_results

        except Exception as e:
            logger.error(f"Error analyzing landmarks: {str(e)}")
            return None

    def _find_lowest_mandible_points(self, landmarks):
        """Находит ключевые точки для анализа мандибулы"""
        # Индексы точек MediaPipe Face Mesh
        upper_left_idx = 172
        upper_right_idx = 397
        lower_left_candidates = [148, 176, 149, 150]
        lower_right_candidates = [382, 400, 381, 380]

        upper_left_y = landmarks[upper_left_idx][1]
        upper_right_y = landmarks[upper_right_idx][1]

        lower_left_idx = max(
            (i for i in lower_left_candidates if landmarks[i][1] > upper_left_y),
            key=lambda i: landmarks[i][1],
            default=upper_left_idx,
        )
        lower_right_idx = max(
            (i for i in lower_right_candidates if landmarks[i][1] > upper_right_y),
            key=lambda i: landmarks[i][1],
            default=upper_right_idx,
        )

        return upper_left_idx, lower_left_idx, upper_right_idx, lower_right_idx

# Создаем экземпляр анализатора
analyzer = FaceAnalyzer()

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_landmarks():
    """API endpoint для анализа landmarks"""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"})

    try:
        data = request.json
        if not data or 'landmarks' not in data:
            return jsonify({"error": "No landmarks provided"}), 400

        landmarks = data['landmarks']
        image_width = data.get('image_width', 1)
        image_height = data.get('image_height', 1)

        if len(landmarks) < 468:  # MediaPipe Face Mesh имеет 468 landmarks
            return jsonify({"error": "Invalid landmarks data"}), 400

        # Анализируем landmarks
        results = analyzer.analyze_landmarks(landmarks, image_width, image_height)

        if results is None:
            return jsonify({"error": "Failed to analyze landmarks"}), 400

        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
