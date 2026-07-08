from functools import lru_cache
from app.providers.base import BaseProvider
from app.providers.fireworks import FireworksProvider
from app.providers.local import LocalProvider


@lru_cache
def _local() -> LocalProvider:
    return LocalProvider()


@lru_cache
def _cloud() -> FireworksProvider:
    return FireworksProvider()


def get_provider(route: str) -> BaseProvider:
    """route: 'local' or 'cloud'. Cached so HTTP clients are reused."""
    if route == "cloud":
        return _cloud()
    return _local()