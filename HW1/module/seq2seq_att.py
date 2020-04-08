import tensorflow
from sklearn.utils import class_weight
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Dropout, Flatten, RepeatVector, merge
from tensorflow.python.keras.layers import Bidirectional, SpatialDropout1D, dot, Activation, concatenate, Add, Multiply, Permute, Concatenate
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers.core import Permute
from tensorflow.python.keras.regularizers import l2

def seq_2_seq_att_LSTM(X_embedding, MAX_LEN, num_words,
                EMBEDDING_DIM, LSTM_units, LSTM_dropout):

    # Encoder
    # Encoder input shape is (batch size, max length)
    encoder_inputs = Input(shape=(MAX_LEN,))
    encoder_embedding = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, 
                        input_length = MAX_LEN, embeddings_initializer=Constant(X_embedding), 
                        trainable=False)(encoder_inputs)

    # LSTM
    encoder_lstm = LSTM(units=LSTM_units, return_state=True, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, trainable=True)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = LSTM(units=LSTM_units, return_state=True, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention
    attention_weight = dot([decoder_outputs, encoder_outputs], axes=[2, 2], normalize=True) # cosine similarity
    attention = Activation('softmax')(attention_weight)

    context = dot([attention, encoder_outputs], axes=[2,1])
    decoder_combined_context = concatenate([context, decoder_outputs])

    att_output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context) 
    output = TimeDistributed(Dense(num_words, activation="softmax"))(att_output)
    
    model = Model(inputs=[encoder_inputs,decoder_inputs], outputs=output)

    return model

def seq_2_seq_biLSTM_att(X_embedding, MAX_LEN, num_words,
                EMBEDDING_DIM, LSTM_units, LSTM_dropout):
    
    # Encoder
    # [?, 100]
    encoder_inputs = Input(shape=(MAX_LEN,))

    # [?, 100, 300]
    encoder_embedding = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, 
                        input_length = MAX_LEN, embeddings_initializer=Constant(X_embedding), 
                        trainable=False)(encoder_inputs)

    # LSTM
    
    encoder_lstm = Bidirectional(LSTM(units=LSTM_units, return_state=True, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout))
    # [?, 100, 300]
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    # [?, 300]
    state_h = concatenate([forward_h, backward_h])
    state_c = concatenate([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    # [?, 30]
    decoder_inputs = Input(shape=(None,))
    decoder_embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, trainable=True)
    # [?, 30, 300]
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = LSTM(units=2*LSTM_units, return_state=True, return_sequences=True, recurrent_dropout=LSTM_dropout, dropout=LSTM_dropout)
    # [?, 30, 300]
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    # [?, 30, 100]
    attention_weight = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention_weight)

    # [?, 30, 300]
    context = dot([attention, encoder_outputs], axes=[2,1]) #[?, 100, 300] = dot([?,?,100] , [?, 100, 300])
    
    # [?, 30, 600]
    decoder_combined_context = concatenate([context, decoder_outputs])

    # [?, 30, 64]
    att_output = TimeDistributed(Dense(128, activation="tanh"))(decoder_combined_context) 
    # [?, 30, 39093]
    output = TimeDistributed(Dense(num_words, activation="softmax"))(att_output)
    
    model = Model(inputs=[encoder_inputs,decoder_inputs], outputs=output)

    return model
