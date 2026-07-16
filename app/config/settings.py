from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "rex-vision-ai"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "rex_vision"
    MYSQL_USER: str = "rex_user"
    MYSQL_PASSWORD: str = "change-me"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/6"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_FACE_COLLECTION: str = "rex_face_embeddings"
    QDRANT_GESTURE_COLLECTION: str = "rex_gesture_embeddings"
    QDRANT_SCENE_COLLECTION: str = "rex_scene_embeddings"
    QDRANT_MEMORY_COLLECTION: str = "rex_visual_memory"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "rex-vision"
    MINIO_SECURE: bool = False

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CLIENT_ID: str = "rex-vision-ai"
    KAFKA_CONSUMER_GROUP: str = "rex-vision-ai-v1"

    # Auth
    USER_JWT_SECRET_KEY: str = "change-me"
    USER_JWT_ALGORITHM: str = "HS256"
    USER_JWT_ISSUER: str = "rex-auth-service"
    USER_JWT_AUDIENCE: str = "rex-platform"

    # Internal
    INTERNAL_SERVICE_TOKEN: str = "change-me"
    ROBOT_SERVICE_URL: str = "http://rex-robot-service:8000"
    OWNERSHIP_CACHE_TTL_SECONDS: int = 60

    # Device
    DEVICE: str = "cuda"
    CUDA_VISIBLE_DEVICES: str = "0"

    # Model paths
    OBJECT_MODEL_PATH: str = "models/object-detector.pt"
    FACE_MODEL_NAME: str = "buffalo_l"
    EXPRESSION_MODEL_PATH: str = "models/expression.onnx"
    GESTURE_MODEL_PATH: str = "models/gesture.pt"
    SCENE_CAPTION_MODEL_PATH: str = "models/scene-captioner"

    # Thresholds
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.50
    OBJECT_IOU_THRESHOLD: float = 0.45
    FACE_KNOWN_THRESHOLD: float = 0.75
    FACE_UNCERTAIN_THRESHOLD: float = 0.60
    EXPRESSION_CONFIDENCE_THRESHOLD: float = 0.65
    GESTURE_CONFIDENCE_THRESHOLD: float = 0.75
    GESTURE_STABILITY_FRAMES: int = 5

    # Vision pipeline
    VISION_INPUT_WIDTH: int = 640
    VISION_INPUT_HEIGHT: int = 640
    VISION_MAX_FPS: int = 20
    SCENE_DESCRIPTION_INTERVAL_SECONDS: float = 1.0
    EVENT_COOLDOWN_SECONDS: int = 10

    # Feature flags (defaults)
    OBJECT_DETECTION_ENABLED: bool = True
    FACE_DETECTION_ENABLED: bool = True
    FACE_RECOGNITION_ENABLED: bool = True
    EXPRESSION_ESTIMATION_ENABLED: bool = False
    PERSON_TRACKING_ENABLED: bool = True
    GESTURE_DETECTION_ENABLED: bool = True
    SCENE_UNDERSTANDING_ENABLED: bool = True
    SCENE_DESCRIPTION_ENABLED: bool = True
    VISUAL_OBSTACLE_ENABLED: bool = False
    LOW_LIGHT_DETECTION_ENABLED: bool = True
    INCIDENT_CAPTURE_ENABLED: bool = True

    # Training
    TRAINING_ENABLED: bool = True
    MAX_CONCURRENT_TRAINING_JOBS: int = 1
    MODEL_ARTIFACT_PREFIX: str = "models/"
    DATASET_PREFIX: str = "datasets/"

    # GHCR
    GHCR_IMAGE: str = "ghcr.io/OWNER/11-rex-vision-ai"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
