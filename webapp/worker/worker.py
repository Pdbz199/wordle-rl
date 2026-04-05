from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

import pika
from pika.exceptions import AMQPConnectionError

from shared.config import RuntimeSettings
from shared.inference import InferenceEngine


LOGGER = logging.getLogger("wordle-worker")


def _connect_with_retry(rabbitmq_url: str) -> pika.BlockingConnection:
    while True:
        try:
            return pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        except AMQPConnectionError:
            LOGGER.warning("RabbitMQ not ready yet, retrying in 3 seconds.")
            time.sleep(3)


def _parse_payload(raw_body: bytes) -> dict:
    payload = json.loads(raw_body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    return payload


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    settings = RuntimeSettings.from_env()
    engine = InferenceEngine(
        model_path=settings.model_path,
        data_dir=settings.data_dir,
        max_turns=settings.max_turns,
        top_k_default=settings.top_k_default,
        top_k_max=settings.top_k_max,
    )

    LOGGER.info("Loaded model '%s'.", settings.model_path)
    connection = _connect_with_retry(settings.rabbitmq_url)
    channel = connection.channel()
    channel.queue_declare(queue=settings.rpc_queue, durable=True)
    channel.basic_qos(prefetch_count=1)

    def on_request(ch, method, properties, body: bytes) -> None:
        request_id = str(uuid4())
        response: dict
        try:
            payload = _parse_payload(body)
            request_id = str(payload.get("request_id") or request_id)
            turns = payload.get("turns", [])
            top_k = payload.get("top_k")

            response = engine.suggest(
                turns=turns,
                top_k=top_k,
                request_id=request_id,
            )
        except Exception as exc:
            response = {"request_id": request_id, "error": str(exc)}

        if properties.reply_to:
            ch.basic_publish(
                exchange="",
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(
                    correlation_id=properties.correlation_id,
                    content_type="application/json",
                ),
                body=json.dumps(response).encode("utf-8"),
            )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=settings.rpc_queue, on_message_callback=on_request, auto_ack=False)
    LOGGER.info("Worker is listening on queue '%s'.", settings.rpc_queue)
    channel.start_consuming()


if __name__ == "__main__":
    main()
