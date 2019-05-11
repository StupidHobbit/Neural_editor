class TextGenerator:
    def __init__(self, text, sequence_length=50, drop_rate=0.80, batch_size=4096):
        text = cleanup_text(text)
        print(len(text))
        self.text = text
        self.sequence_length = sequence_length
        self.alphabet = make_alphabet_from_text(text)
        self.char_to_index, self.index_to_char = make_mappings_from_alphabet(self.alphabet)
        self.indices = self.encode_text(text)
        self.batch_size = batch_size

    def encode_text(self, text):
        return np.fromiter(map(self.char_to_index.__getitem__, text), dtype=np.int8)

    def sequence_generator(self, begin, end):
        total_words = len(self.alphabet)
        indices = self.indices
        sequence_length = self.sequence_length
        batch_size = self.batch_size
        return ((np.array([to_categorical(indices[i:i + sequence_length], total_words)
                           for i in range(j, j + batch_size)]),
                 indices[j + sequence_length:j + sequence_length + batch_size])
                for j in range(begin, end - sequence_length - batch_size))

    def seed_from_origin(self):
        index = randint(0, len(text) - self.sequence_length)
        return self.text[index:index + self.sequence_length]

    def create_model(self):
        total_words = len(self.alphabet)
        model = Sequential()
        size = 128
        model.add(GRU(size,
                      dropout=0.1,
                      recurrent_dropout=0.4,
                      return_sequences=True,
                      input_shape=(self.sequence_length, total_words)
                      )
                  )
        model.add(GRU(size,
                      dropout=0.1,
                      recurrent_dropout=0.4,
                      return_sequences=True)
                  )
        model.add(GRU(size,
                      dropout=0.1,
                      recurrent_dropout=0.4))
        # model.add(Dropout(0.5))

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

    def generate(self, n, temperature=0.0, prefix=""):
        total_words = len(self.alphabet)
        result = bytearray(prefix.encode())
        prefix = '{:>{}}'.format(prefix, self.sequence_length)
        for i in range(n):
            d = self.model.predict([[to_categorical(self.encode_text(prefix), total_words)]])[0]
            c = self.index_to_char[textgenrnn_sample(d, temperature)]
            prefix = prefix[1:] + c
            result.extend(c.encode())
        return result.decode()

    def generate_word(self, prefix, temperature=0.5):
        return 'hello', []

    def score_word(self, prefix, word, temperature=0.5):
        total_words = len(self.alphabet)
        if prefix[-1] != ' ': prefix += ' '
        result = bytearray()
        prefix = '{:>{}}'.format(prefix, self.sequence_length)
        char_scores = []
        for c in word:
            d = self.model.predict([[to_categorical(self.encode_text(prefix), total_words)]])[0]
            print(d)
            d = apply_temperature(d, temperature)
            print(d)
            prefix = prefix[1:] + c
            index = self.char_to_index[c]
            char_scores.append(d[0][index])
        return np.array(char_scores)    