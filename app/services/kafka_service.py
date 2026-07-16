import json
import logging
from aiokafka import AIOKafkaProducer
from app.config.settings import settings

logger = logging.getLogger(__name__)


class KafkaService:
    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                client_id=settings.KAFKA_CLIENT_ID,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retry_backoff_ms=500,
                request_timeout_ms=10000,
            )
            await self._producer.start()
            logger.info("Kafka producer started")
        except Exception as exc:
            logger.error(f"Kafka producer failed to start: {exc}")
            self._producer = None

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def publish(self, topic: str, payload: dict) -> None:
        if not self._producer:
            logger.debug(f"Kafka not available — skipping publish to {topic}")
            return
        try:
            await self._producer.send_and_wait(topic, payload)
        except Exception as exc:
            logger.error(f"Kafka publish failed for {topic}: {exc}")


kafka_service = KafkaService()
