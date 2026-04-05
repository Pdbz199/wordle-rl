from __future__ import annotations

import json
import time
import uuid

import pika
from pika.exceptions import AMQPError


class RpcClientError(RuntimeError):
    """Base class for RPC communication errors."""


class RpcTimeoutError(RpcClientError):
    """Raised when a worker does not respond before timeout."""


class RabbitRpcClient:
    def __init__(self, rabbitmq_url: str, request_queue: str, timeout_seconds: float = 8.0) -> None:
        self.rabbitmq_url = rabbitmq_url
        self.request_queue = request_queue
        self.timeout_seconds = max(float(timeout_seconds), 0.5)

    def call(self, payload: dict) -> dict:
        correlation_id = str(uuid.uuid4())
        request_body = json.dumps(payload).encode("utf-8")

        response_bytes = self._call_raw(correlation_id=correlation_id, request_body=request_body)
        try:
            return json.loads(response_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RpcClientError("Worker returned invalid JSON.") from exc

    def _call_raw(self, correlation_id: str, request_body: bytes) -> bytes:
        response: bytes | None = None
        params = pika.URLParameters(self.rabbitmq_url)
        deadline = time.monotonic() + self.timeout_seconds

        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=self.request_queue, durable=True)
            callback_queue = channel.queue_declare(queue="", exclusive=True, auto_delete=True).method.queue

            def on_response(_ch, _method, properties, body: bytes) -> None:
                nonlocal response
                if properties.correlation_id == correlation_id:
                    response = body

            channel.basic_consume(queue=callback_queue, on_message_callback=on_response, auto_ack=True)
            channel.basic_publish(
                exchange="",
                routing_key=self.request_queue,
                body=request_body,
                properties=pika.BasicProperties(
                    reply_to=callback_queue,
                    correlation_id=correlation_id,
                    delivery_mode=2,
                    content_type="application/json",
                ),
            )

            while response is None and time.monotonic() < deadline:
                connection.process_data_events(time_limit=0.2)
        except AMQPError as exc:
            raise RpcClientError(f"RabbitMQ connection error: {exc}") from exc
        finally:
            try:
                if "connection" in locals() and connection.is_open:
                    connection.close()
            except Exception:
                pass

        if response is None:
            raise RpcTimeoutError("Timed out waiting for worker response.")
        return response
