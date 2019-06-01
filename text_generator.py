import numpy as np
import string, os
from random import shuffle, randint
import re
import time
from queue import Queue
from threading import Thread
from multiprocessing.dummy import Pool

from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Lambda
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.backend import epsilon
from keras import backend as K


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                         num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


def make_alphabet_from_text(text):
    return ''.join(list(sorted(set(text))))


def make_mappings_from_alphabet(alphabet):
    char_to_index_mapping = {c: alphabet.find(c) for c in alphabet}
    index_to_char_mapping = {i: alphabet[i] for i in range(len(alphabet))}
    return char_to_index_mapping, index_to_char_mapping


def cleanup_text(text):
    punctuations = '"' + "(),.''!:;*—-"
    text = re.sub("[^0-9а-яА-Я {}]+".format(punctuations), ' ', text)
    text = re.sub(" +", ' ', text)
    return text


def apply_temperature(preds, temperature=0.0):
    if temperature == 0.0:
        return preds
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + epsilon()) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return probas


def sample(probas):
    index = np.argmax(probas)
    return index


class TextGenerator:
    def __init__(self, text, sequence_length=50, batch_size=128):
        text = cleanup_text(text)
        self.text = text
        self.sequence_length = sequence_length
        self.alphabet = make_alphabet_from_text(text)
        self.alphabet = "зуНео—н:фИр75*эЩхАЯТъбО2( 'ивыМсд;ЕжУГ9ДЖШЙ)йщгч,!Рь-ю0ЮЦцЗК86ЧПС1клтпшФ4ХЛ.маяВЭ3Б"
        self.char_to_index, self.index_to_char = make_mappings_from_alphabet(self.alphabet)
        self.indices = self.encode_text(text)
        self.batch_size = batch_size
        self.predicted_sequences = {}
        self.inputs_queue = Queue()
        self.prediction_thread = None
        self.is_predicting = False
        self.max_waiting_time = 0.01
        self.predictions_cancelled = False

    def encode_text(self, text):
        return np.fromiter(map(self.char_to_index.__getitem__, text), dtype=np.int8)

    def sequence_generator(self, begin, end):
        total_words = len(self.alphabet)
        indices = self.indices
        sequence_length = self.sequence_length
        batch_size = self.batch_size
        return ((np.array([indices[i:i + sequence_length]
                           for i in range(j, j + batch_size)]),
                 indices[j + sequence_length:j + sequence_length + batch_size])
                for j in range(begin, end - sequence_length - batch_size))

    def seed_from_origin(self):
        index = randint(0, len(self.text) - self.sequence_length)
        return self.text[index:index + self.sequence_length]

    def create_model(self, layers_num=3, size=128, dropout=0.0, recurrent_dropout=0.0):
        total_words = len(self.alphabet)
        model = Sequential()

        model.add(OneHot(input_dim=total_words,
                         input_length=self.sequence_length))

        for i in range(1, layers_num + 1):
            model.add(
                GRU(
                    size,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=i != layers_num,
                )
            )

        model.add(Dense(total_words, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=["accuracy"])

        self.model = model

    def train(self, train_size, epochs=20):
        test_gen = self.sequence_generator(0, train_size)
        val_gen = self.sequence_generator(train_size + 1, train_size + 10 ** 5)
        history = self.model.fit_generator(
            test_gen,
            steps_per_epoch=train_size // self.batch_size,
            epochs=epochs,
        )
        return history

    def save_model(self, filename):
        self.model.save(filename + '.h5')

    def load_model(self, filename):
        self.model = load_model(filename + '.h5')

    def generate_word(self, prefix, temperature=0.5):
        seqlen = self.sequence_length
        prefix = cleanup_text(prefix)
        prefix = ' ' * max(0, seqlen - len(prefix)) + prefix
        result = bytearray()
        prefix = prefix[-seqlen:]
        is_word_re = re.compile(r'\w')
        char_scores = []
        max_size = 100
        encoded_prefix = self.encode_text(prefix)
        encoded_prefix = np.resize(encoded_prefix, max_size + seqlen)
        for i in range(max_size):
            if self.predictions_cancelled:
                break
            d = self.predict(encoded_prefix[i:i + seqlen])
            index = apply_temperature(d, temperature).argmax()
            c = self.alphabet[index]
            if not is_word_re.match(c):
                break
            encoded_prefix[i + seqlen] = index
            char_scores.append(d[index])
            result.extend(c.encode())
        return result.decode(), np.array(char_scores)

    def score_word(self, prefix, word, temperature=0.5):
        seqlen = self.sequence_length
        if prefix[-1] != ' ': prefix += ' '
        prefix = prefix[-seqlen:]
        char_scores = []
        max_size = 100
        encoded_prefix = self.encode_text(prefix)
        encoded_prefix = np.resize(encoded_prefix, max_size + seqlen)
        for i, c in enumerate(word):
            d = self.predict(encoded_prefix[i:i + seqlen])
            index = self.char_to_index[c]
            encoded_prefix[i + seqlen] = index
            char_scores.append(d[index])
        return np.array(char_scores)

    def predict(self, x):
        xt = tuple(x)
        if self.predicted_sequences.get(xt) is None:
            self.inputs_queue.put(x)
            while self.predicted_sequences.get(xt) is None:
                time.sleep(0.02)
        return self.predicted_sequences[xt]

    def predicting(self):
        while self.is_predicting:
            begin_time = time.perf_counter()
            cur_time = begin_time
            while (self.inputs_queue.qsize() < self.batch_size and
                   cur_time - begin_time < self.max_waiting_time):
                time.sleep(.02)
                cur_time = time.perf_counter()
            batch = []
            for i in range(min(self.batch_size, self.inputs_queue.qsize())):
                x = self.inputs_queue.get()
                batch.append(x)
                self.inputs_queue.task_done()
            if batch:
                print(len(batch))
                batch_output = self.model.predict([batch], batch_size=self.batch_size)
                for x, y in zip(batch, batch_output):
                    self.predicted_sequences[tuple(x)] = y

    def start_predicting(self):
        self.model.predict([[[0] * self.sequence_length]])
        self.prediction_thread = Thread(target=self.predicting)
        self.is_predicting = True
        self.prediction_thread.start()

    def stop_predicting(self):
        self.is_predicting = False
        self.prediction_thread.join()