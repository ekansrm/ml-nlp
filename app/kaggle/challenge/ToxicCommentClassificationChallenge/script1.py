
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import json
from keras.models import Model, load_model
from keras.layers import Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data.kaggle.challenge.ToxicCommentClassification import train_filepath, test_filepath, filepath

max_features = 20000
maxlen = 100

train = pd.read_csv(train_filepath)
test = pd.read_csv(test_filepath)
train = train.sample(frac=0.001)
test = test.sample(frac=0.001)

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


class Runner(object):
    """
    Runner的职责是负责整个环境的管理和持久化

    """

    model_filename = 'model'
    params_filename = 'params'

    def __init__(self, model, save_dir, inputs_len, inputs_vocab, outputs_class):
        self._model = model
        self._save_dir = save_dir
        self._inputs_len = inputs_len
        self._inputs_vocab = inputs_vocab
        self._outputs_class = outputs_class
        self._idx = None
        self._x = None
        self._y = None

    @classmethod
    def init(cls, save_dir, inputs_len, inputs_vocab, outputs_class):
        """
        :param params_filename:
        :param model_filename:
        :param save_dir:
        :param outputs_class:
        :param inputs_vocab:
        :param inputs_len:
        :param cls:
        :return:
        """

        inp = Input(shape=(inputs_len,))

        op = EmbeddingBiLSTM(name='BiLSTM',
                             embedding_word_number=inputs_vocab,
                             embedding_vector_length=128,
                             embedding_dropout=None,
                             embedding_kwargs=None,

                             lstm_units=50,
                             lstm_dropout=None,
                             lstm_kwargs=None,

                             dense=[
                                 {'units': 50, 'activation': 'relu', 'dropout': 0.1, 'name': 'd'},
                                 {'units': len(outputs_class), 'activation': 'sigmoid', 'dropout': 0.1, 'name': 'd'}
                             ]
                             )(inp)

        model = Model(inputs=inp, outputs=op)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # 保存参数
        params_dict = {
            'save_dir': save_dir,
            'inputs_len': inputs_len,
            'outputs_class': outputs_class,
        }

        try:
            os.makedirs(save_dir)
        except OSError:
            pass

        with open(file=os.path.join(save_dir, cls.params_filename), mode="w+") as fp:
            fp.write(json.dumps(params_dict))

        return cls(model, save_dir, inputs_len, inputs_vocab, outputs_class)

    @classmethod
    def load(cls, save_dir):
        """
        :return:
        """
        model_filepath = os.path.join(save_dir, cls.model_filename)
        if not os.path.isfile(model_filepath):
            model_filepath = model_filepath + '.best'
        kwargs = json.load(open(file=model_filepath, mode="r"))
        model = load_model(os.path.join(save_dir, cls.params_filename))
        kwargs['model'] = model
        return cls(**kwargs)

    def fit(self, batch_size=32, epochs=10, verbose=1, callbacks=None,
            validation_split=0.1, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):

        if callbacks is None:
            suffix = '.%d.checkpoint' '.epoch-{epoch:02d}' '.val_loss-{val_loss:.6f}' '.val_acc-{val_acc:.6f}'

            model_checkpoint_best_path = os.path.join(self._save_dir, self.model_filename + '.best')

            model_checkpoint_better_path = os.path.join(self._save_dir, self.model_filename + suffix)

            checkpoint = ModelCheckpoint(
                model_checkpoint_better_path, save_best_only=False, verbose=1)

            checkpoint_best = ModelCheckpoint(
                model_checkpoint_best_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

            early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

            callbacks = [checkpoint, checkpoint_best, early]

        self._model.fit(self._x, self._y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                        validation_split=validation_split, validation_data=validation_data, shuffle=shuffle,
                        class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, **kwargs)

        return self

    def predict(self,
                batch_size=None,
                verbose=0,
                steps=None):
        """

        :return:
        """
        self._y = self._model.predict(self._x, batch_size=batch_size, verbose=verbose, steps=steps)
        return self

    def submit(self, save_path):

        sample_submission = pd.DataFrame(columns=['id'] + self._outputs_class)
        sample_submission['id'] = self._idx
        sample_submission[self._outputs_class] = self._y
        sample_submission.to_csv(save_path, index=False)

    def set_idx(self, idx):
        self._idx = idx
        return self

    def set_x(self, x):
        self._x = x
        return self

    def set_y(self, y):
        self._y = y
        return self

    @property
    def idx(self):
        return self._idx

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


def get_model():
    inp = Input(shape=(maxlen,))

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

# file_path = filepath("weights_base.best.hdf5")
#
# checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#
# early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
#
#
# callbacks_list = [checkpoint, early] #early
# model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
#
# model.load_weights(file_path)
#
# y_test = model.predict(X_te)
#
#
# sample_submission = pd.read_csv(filepath("sample_submission.csv"))
#
# sample_submission[list_classes] = y_test
#
#
# sample_submission.to_csv(filepath("baseline.csv"), idx=False)


if __name__ == '__main__':
    """
    """
    Runner\
        .init(save_dir=filepath("BiLSTM"),
              inputs_len=maxlen, inputs_vocab=max_features,
              outputs_class=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])\
        .set_x(X_t).set_y(y).fit(batch_size=batch_size, epochs=epochs, validation_split=0.1)\
        .set_x(X_te).set_idx(test['id']).predict(verbose=True).submit("123.csv")
