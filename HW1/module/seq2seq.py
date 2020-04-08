from tensorflow import keras
from sklearn.utils import class_weight
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, SpatialDropout1D 
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import Constant

def seq_2_seq(X_embedding, MAX_LEN, num_words,
				EMBEDDING_DIM, LSTM_units, LSTM_dropout):

	# Encoder
	encoder_inputs = Input(shape=(MAX_LEN,))
	encoder_embedding = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, 
						input_length = MAX_LEN, embeddings_initializer=Constant(X_embedding), 
						trainable=False)(encoder_inputs)

	# LSTM
	encoder_lstm = LSTM(units=LSTM_units, return_state=True, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout)
	encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)

	# Decoder
	decoder_inputs = Input(shape=(None,))
	decoder_embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, trainable=True)
	decoder_embedding = decoder_embedding_layer(decoder_inputs)
	decoder_lstm = LSTM(units=LSTM_units, return_state=True, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout)
	decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
	
	output = TimeDistributed(Dense(num_words, activation='softmax'))(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], output)

	return model



