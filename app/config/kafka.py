from app.config.settings import settings

# Kafka topic constants
TOPIC_STATE_UPDATED = "rex.vision.state.updated.v1"
TOPIC_CONTEXT_UPDATED = "rex.vision.context.updated.v1"
TOPIC_OBJECT_DETECTED = "rex.vision.object.detected.v1"
TOPIC_PERSON_DETECTED = "rex.vision.person.detected.v1"
TOPIC_FACE_RECOGNIZED = "rex.vision.face.recognized.v1"
TOPIC_UNKNOWN_PERSON = "rex.vision.unknown-person.detected.v1"
TOPIC_EXPRESSION_ESTIMATED = "rex.vision.expression.estimated.v1"
TOPIC_GESTURE_DETECTED = "rex.vision.gesture.detected.v1"
TOPIC_PERSON_MOVEMENT = "rex.vision.person-movement.detected.v1"
TOPIC_FALL_CANDIDATE = "rex.vision.fall-candidate.detected.v1"
TOPIC_TARGET_UPDATED = "rex.vision.target.updated.v1"
TOPIC_TARGET_LOST = "rex.vision.target.lost.v1"
TOPIC_SCENE_UPDATED = "rex.vision.scene.updated.v1"
TOPIC_LOW_LIGHT = "rex.vision.low-light.detected.v1"
TOPIC_VISUAL_OBSTACLE = "rex.vision.visual-obstacle.detected.v1"
TOPIC_MODEL_STATUS_CHANGED = "rex.vision.model.status-changed.v1"
TOPIC_TRAINING_COMPLETED = "rex.vision.training.completed.v1"
TOPIC_TRAINING_FAILED = "rex.vision.training.failed.v1"

# Topics consumed
TOPIC_ROBOT_MODE_CHANGED = "rex.robot.mode-changed.v1"
TOPIC_VISION_CONFIG_UPDATED = "rex.robot.vision-config.updated.v1"
TOPIC_AGENT_VISION_QUERY = "rex.agent.vision-query.requested.v1"
TOPIC_VOICE_VISION_CONTEXT = "rex.voice.vision-context.requested.v1"

KAFKA_BOOTSTRAP_SERVERS = settings.KAFKA_BOOTSTRAP_SERVERS
KAFKA_CLIENT_ID = settings.KAFKA_CLIENT_ID
KAFKA_CONSUMER_GROUP = settings.KAFKA_CONSUMER_GROUP
