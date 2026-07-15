from typing import Protocol


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, **cfg) -> str: ...

    async def acomplete(self, system: str, user: str, **cfg) -> str: ...
