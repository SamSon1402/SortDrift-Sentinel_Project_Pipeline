from __future__ import annotations

import os


class PostHogOps:
    """Optional YC W20 analytics sink for incident/release lifecycle events."""

    def __init__(self) -> None:
        api_key = os.getenv("POSTHOG_API_KEY")
        if not api_key:
            self.client = None
            return
        try:
            from posthog import Posthog
        except ImportError as exc:
            raise RuntimeError("install optional dependency: pip install -e '.[yc]'") from exc
        self.client = Posthog(api_key, host=os.getenv("POSTHOG_HOST", "https://us.i.posthog.com"))

    def capture(self, deployment_id: str, event: str, properties: dict) -> None:
        if self.client is None:
            return
        self.client.capture(
            distinct_id=deployment_id,
            event=event,
            properties=properties,
        )
