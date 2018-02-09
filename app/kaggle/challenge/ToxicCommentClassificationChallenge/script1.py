from data.kaggle.challenge.ToxicCommentClassification import train_filepath, test_filepath, filepath


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 100


train = pd.read_csv(train_filepath)
test = pd.read_csv(test_filepath)
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

from model.k.layers.dense.Dense import MultiDense


class EmbeddingBiLSTM(object):

    def __init__(self,
                 name: str = 'EmbeddingBiLSTM',

                 embedding_word_number: int = None,
                 embedding_vector_length: int = None,
                 embedding_dropout: float = None,
                 embedding_kwargs: dict = None,

                 lstm_units: int = None,
                 lstm_dropout: float = None,
                 lstm_kwargs: dict = None,

                 dense: list = None

                 ):
        """
        构造函数, 设置模型参数
        :param
        name: 模型各层的命名规则为[模型名称].[层名称]
        :param
        embedding_word_number: Embedding层的词典大小
        :param
        embedding_vector_length: Embedding层的输出序列长度
        :param
        embedding_dropout: Embedding层的dropout.若为None或1, 则无该Dropout层.
        :param
        embedding_kwargs: Embedding层的其他参数, 参考
        keras.layers.embeddings.Embedding
        :param
        lstm_units: LSTM层的神经元数
        :param
        lstm_dropout: LSTM层的dropout.若为None或1, 则无该Dropout层.
        :param
        lstm_kwargs: LSTM层的其他参数, 参考
        keras.layers.LSTM
        :param
        dense: 定义每个全连接层的名称, 神经元数, 参数初始化函数, 激活函数, dropout等.若为None或[], 则无全连接层
        """

        self.name = name

        if embedding_kwargs is not None:
            assert 'input_dim' not in embedding_kwargs
            assert 'output_dim' not in embedding_kwargs
            assert 'input_length' not in embedding_kwargs
            self.embeddings_kwargs = embedding_kwargs
        else:
            self.embeddings_kwargs = {}

        self.embedding_name = 'embedding'
        if 'name' in self.embeddings_kwargs:
            self.embedding_name = self.embeddings_kwargs.pop('name')

        self.embedding_word_number = embedding_word_number
        self.embedding_vector_length = embedding_vector_length
        assert embedding_dropout is None or (0 < embedding_dropout < 1)
        self.embedding_dropout = embedding_dropout

        if lstm_kwargs is not None:
            assert 'units' not in lstm_kwargs
            self.lstm_kwargs = lstm_kwargs
        else:
            self.lstm_kwargs = {}

        self.lstm_name = 'lstm'
        if 'name' in self.lstm_kwargs:
            self.lstm_name = self.lstm_kwargs.pop('name')
        self.lstm_units = lstm_units
        assert lstm_dropout is None or (0 < lstm_dropout < 1)
        self.lstm_dropout = lstm_dropout

        if dense is None:
            dense = [
                {'units': self.lstm_units, 'activation': 'relu', 'dropout': 0.1, 'name': 'd'},
                {'units': 1, 'activation': 'sigmoid', 'dropout': 0.1, 'name': 'd'}
            ]

        self._layer_dense = MultiDense(name=self.name, dense_kwargs_list=dense)

    def __call__(self, inputs, **kwargs):

        embedding_layer_name = self.name + '.' + self.embedding_name
        x_embedded = Embedding(name=embedding_layer_name,
                               input_dim=self.embedding_word_number,
                               output_dim=self.embedding_vector_length,
                               **self.embeddings_kwargs
                               )(inputs)

        x_lstm = Bidirectional(LSTM(units=self.lstm_units, return_sequences=True))(x_embedded)
        pool = GlobalMaxPool1D()(x_lstm)
        pool = Dropout(0.1)(pool)

        dense = self._layer_dense(pool)

        return dense


class RunnerBiLSTM(object):

    def __init__(self, model, save_path):
        self._model = model
        self._save_path = save_path

    @classmethod
    def build(cls, save_path):
        """
        :param self:
        :return:
        """

        inp = Input(shape=(maxlen, ))

        op = EmbeddingBiLSTM(name='BiLSTM',
                             embedding_word_number=max_features,
                             embedding_vector_length=128,
                             embedding_dropout=None,
                             embedding_kwargs=None,

                             lstm_units=50,
                             lstm_dropout=None,
                             lstm_kwargs=None,

                             dense=[
                                 {'units': 50, 'activation': 'relu', 'dropout': 0.1, 'name': 'd'},
                                 {'units': 6, 'activation': 'sigmoid', 'dropout': 0.1, 'name': 'd'}
                             ]
                             )(inp)

        model = Model(inputs=inp, outputs=op)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return cls(model, save_path)

    @classmethod
    def load(cls):
        """"""

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):

        file_path = filepath("weights_base.best.hdf5")

        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

        callbacks_list = [checkpoint, early] #early

        model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

    def predict(self,):
        """

        :return:
        """
        y_test = model.predict(X_te)

        sample_submission = pd.read_csv(filepath("sample_submission.csv"))

        sample_submission[list_classes] = y_test

        sample_submission.to_csv(filepath("baseline.csv"), index=False)


def get_model():

    inp = Input(shape=(maxlen, ))

    op = EmbeddingBiLSTM(name='BiLSTM',
                         embedding_word_number=max_features,
                         embedding_vector_length=128,
                         embedding_dropout=None,
                         embedding_kwargs=None,

                         lstm_units=50,
                         lstm_dropout=None,
                         lstm_kwargs=None,

                         dense=[
                             {'units': 50, 'activation': 'relu', 'dropout': 0.1, 'name': 'd'},
                             {'units': 6, 'activation': 'sigmoid', 'dropout': 0.1, 'name': 'd'}
                         ]
                         )(inp)

    model = Model(inputs=inp, outputs=op)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = get_model()
batch_size = 32
epochs = 2


file_path = filepath("weights_base.best.hdf5")

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict(X_te)


sample_submission = pd.read_csv(filepath("sample_submission.csv"))

sample_submission[list_classes] = y_test


sample_submission.to_csv(filepath("baseline.csv"), index=False)


if __name__ == '__main__':
    """
    """
