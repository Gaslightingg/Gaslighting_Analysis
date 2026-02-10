from src.storage import models  # noqa: F401
from src.storage.db import Base, engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
