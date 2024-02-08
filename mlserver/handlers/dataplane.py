from prometheus_client import Counter, Summary, Gauge
from typing import Optional

from ..errors import ModelNotReady
from ..context import model_context
from ..settings import Settings
from ..registry import MultiModelRegistry
from ..types import (
    MetadataModelResponse,
    MetadataServerResponse,
    InferenceRequest,
    InferenceResponse,
)
from ..middleware import InferenceMiddlewares
from ..cloudevents import CloudEventsMiddleware
from ..utils import generate_uuid
from ..logging import logger


class DataPlane:
    """
    Internal implementation of handlers, used by both the gRPC and REST
    servers.
    """

    def __init__(self, settings: Settings, model_registry: MultiModelRegistry):
        self._settings = settings
        self._model_registry = model_registry

        self._inference_middleware = InferenceMiddlewares(
            CloudEventsMiddleware(settings)
        )

        # TODO: Update to standardised set of labels
        self._ModelInferRequestTotal = Counter(
            "model_infer_request_total",
            "Model infer request total count",
            ["model", "version"],
        )
        self._ModelInferRequestSLA = Gauge(
            "model_infer_request_sla",
            "Model request Service Level Agreement (SLA)",
            ["model", "version"],
        )
        self._ModelInferRequestSuccess = Counter(
            "model_infer_request_success",
            "Model infer request success count",
            ["model", "version"],
        )
        self._ModelInferRequestFailure = Counter(
            "model_infer_request_failure",
            "Model infer request failure count",
            ["model", "version"],
        )
        self._ModelInferRequestDuration = Summary(
            "model_infer_request_duration",
            "Model infer request duration",
            ["model", "version"],
        )

    async def live(self) -> bool:
        return True

    async def ready(self) -> bool:
        models = await self._model_registry.get_models()
        return all([model.ready for model in models])

    async def model_ready(self, name: str, version: Optional[str] = None) -> bool:
        model = await self._model_registry.get_model(name, version)
        return model.ready

    async def metadata(self) -> MetadataServerResponse:
        return MetadataServerResponse(
            name=self._settings.server_name,
            version=self._settings.server_version,
            extensions=self._settings.extensions,
        )

    async def model_metadata(
        self, name: str, version: Optional[str] = None
    ) -> MetadataModelResponse:
        model = await self._model_registry.get_model(name, version)
        # TODO: Make await optional for sync methods
        with model_context(model.settings):
            return await model.metadata()

    async def infer(
        self,
        payload: InferenceRequest,
        name: str,
        version: Optional[str] = None,
    ) -> InferenceResponse:
        infer_duration = self._ModelInferRequestDuration.labels(
            model=name, version=version
        ).time()
        infer_errors = self._ModelInferRequestFailure.labels(
            model=name, version=version
        ).count_exceptions()

        self._ModelInferRequestTotal.labels(model=name, version=version).inc()
        try:
            sla = payload.inputs[0].parameters.extended_parameters['sla']
        except (AttributeError, TypeError):
            sla = 0
        self._ModelInferRequestSLA.labels(model=name, version=version).set(sla)

        with infer_duration, infer_errors:
            if payload.id is None:
                payload.id = generate_uuid()

            model = await self._model_registry.get_model(name, version)
            if not model.ready:
                raise ModelNotReady(name, version)

            self._inference_middleware.request_middleware(payload, model.settings)

            # TODO: Make await optional for sync methods
            with model_context(model.settings):
                prediction = await model.predict(payload)

            # Ensure ID matches
            prediction.id = payload.id

            self._inference_middleware.response_middleware(prediction, model.settings)

            self._ModelInferRequestSuccess.labels(model=name, version=version).inc()

            return prediction
