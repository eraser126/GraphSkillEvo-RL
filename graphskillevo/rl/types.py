from dataclasses import dataclass
import hashlib


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Skill:
    content: str
    origin: str = "initial"
    age: int = 0

    @property
    def id(self):
        return digest(self.content)


@dataclass(frozen=True)
class Score:
    hard: float
    soft: float
    reflection: str = ""


@dataclass(frozen=True)
class Edit:
    content: str | None
    operator: str
    donor_id: str | None = None
    attempts: int = 0
    errors: tuple = ()
