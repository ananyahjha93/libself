from abc import ABC, abstractmethod


class AbstractSSLTask(ABC):
    def __init__(self):
        super(AbstractSSLTask, self).__init__()
        self.transform = self.generate_transform()

    @abstractmethod
    def generate_transform(self):
        pass

    @abstractmethod
    def ssl_task(self, img):
        pass
