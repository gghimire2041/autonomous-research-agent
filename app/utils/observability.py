"""Observability setup: logging, metrics, and tracing."""

import logging
import sys
from typing import Optional

import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, Gauge

from app.utils.config import get_settings

settings = get_settings()

# Prometheus Metrics
task_counter = Counter('agent_tasks_total', 'Total number of tasks', ['status'])
task_duration = Histogram('agent_task_duration_seconds', 'Task execution time')
tool_calls = Counter('agent_tool_calls_total', 'Total tool calls', ['tool_name', 'status'])
active_tasks = Gauge('agent_active_tasks', 'Number of active tasks')


def setup_logging():
    """Setup structured logging with structlog."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )


def setup_metrics():
    """Setup Prometheus metrics collection."""
    if not settings.ENABLE_METRICS:
        return
    
    # Metrics are automatically collected via prometheus_client
    # Additional custom metrics can be defined here
    pass


def setup_tracing():
    """Setup OpenTelemetry distributed tracing."""
    if not settings.ENABLE_TRACING:
        return
    
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    
    # Set up Jaeger exporter if endpoint provided
    if settings.JAEGER_ENDPOINT:
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
            collector_endpoint=settings.JAEGER_ENDPOINT,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
    
    # Instrument libraries
    RequestsInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()


def get_tracer(name: str):
    """Get tracer instance."""
    return trace.get_tracer(name)
