import sys
from threading import Thread

from PyQt5 import QtCore, QtWidgets, QtGui

import design
from model import models


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.predictions = set()
        self.prediction_threads = []
        self.predictions_number = 10
        self.temperature = 0.0
        self.predictor = None
        self.sequence_length = 0

        self.setupUi(self)
        self.setup_models()
        self.text_edit.cursorPositionChanged.connect(self.predict)
        self.words_list_widget.itemClicked.connect(self.complete)
        self.temperature_slider.valueChanged.connect(self.update_temperature)
        self.author_cbox.currentTextChanged.connect(self.update_model)

    def setup_models(self):
        self.author_cbox.clear()
        for name in models:
            self.author_cbox.addItem(name)
        self.update_model()

    def update_model(self):
        if self.predictor is not None:
            self.predictor.stop_predicting()
        name = self.author_cbox.currentText()
        self.predictor = models[name].generator
        self.predictor.start_predicting()
        self.sequence_length = self.predictor.sequence_length

    def update_temperature(self):
        self.temperature = (self.temperature_slider.value() /
                            self.temperature_slider.maximum())
        self.temperature_label.setText("Температура: %.2f" % (self.temperature))
        self.predict()

    def predict(self):
        self.stop_predictions()
        self.predictions.clear()
        self.words_list_widget.clear()
        prefix = self.get_prefix()
        for i in range(self.predictions_number):
            t = Thread(target=self.predict_one, args=(prefix,))
            t.start()
            self.prediction_threads.append(t)

    def stop_predictions(self):
        self.predictor.predictions_cancelled = True
        for t in self.prediction_threads:
            t.join()
        self.predictor.predictions_cancelled = False
        self.prediction_threads.clear()

    def predict_one(self, prefix):
        item = self.predictor.generate_word(prefix, self.temperature)[0]
        if item and item not in self.predictions:
            self.predictions.add(item)
            self.words_list_widget.addItem(item)

    def get_prefix(self):
        cursor = self.text_edit.textCursor()
        pos = cursor.position()
        text = self.text_edit.toPlainText()
        prefix = text[max(0, pos - self.sequence_length):pos]
        return prefix

    def complete(self):
        word = self.words_list_widget.selectedItems()[0].text()
        cursor = self.text_edit.textCursor()
        cursor.insertText(word)
        self.text_edit.setFocus()

    def closeEvent(self, *args, **kwargs):
        self.predictor.stop_predicting()
        super().closeEvent(*args, **kwargs)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
