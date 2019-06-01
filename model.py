from pathlib import Path

from text_generator import TextGenerator


class Model:
    def __init__(self, author, path='', sequence_length=50):
        path = Path(path)
        with open(path / author, encoding='utf-8') as f:
            text = f.read()
        self.generator = TextGenerator(text)

        self.generator.load_model((path / author).name)
        self.author = author


models = {
    "Толстой": Model("tolstoy"),
}

