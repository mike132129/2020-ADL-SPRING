from tensorflow import keras
from sklearn.utils import class_weight
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, SpatialDropout1D 
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import Constant

def biLSTM_baseline(embedding_matrix, MAX_LEN, num_words, 
					EMBEDDING_DIM, LSTM_units, LSTM_dropout):
	input_dimen = Input(shape=(MAX_LEN,))
	model = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, input_length=MAX_LEN,
						embeddings_initializer=Constant(embedding_matrix), trainable=False)(input_dimen)

	model = Bidirectional(LSTM(units=LSTM_units, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout))(model)
	model = Bidirectional(LSTM(units=LSTM_units, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout))(model)
	out = TimeDistributed(Dense(1, activation='sigmoid'))(model)
	model = Model(input_dimen, out)
	

	return model





