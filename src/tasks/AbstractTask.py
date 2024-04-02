from abc import ABC, abstractmethod


class AbstractTask(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        pass
