import tensorflow as tf
from tensorflow.keras import layers, models

def opt2_cnn(input_width, num_classes, num_hidden_layers, dropout_rate=0.5):
	inputs = tf.keras.Input(shape=(input_width, 36, 3))
	
	x = inputs
	for _ in range(num_hidden_layers):
		x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(dropout_rate)(x)
	
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation='relu')(x)
	x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def opt2_lstm(num_classes, num_hidden_layers, dropout_rate=0.5, max_len=10):
	inputs = tf.keras.Input(shape=(max_len, 36, 3)) 
	
	x = layers.Reshape(target_shape=(-1, 36*3))(inputs)
	for _ in range(num_hidden_layers):
		x = layers.LSTM(64, return_sequences=True)(x)
		x = layers.Dropout(dropout_rate)(x)
	
	x = layers.LSTM(64)(x)
	x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def audio_cnn(input_width, num_classes, num_hidden_layers, dropout_rate=0.5):
	inputs = tf.keras.Input(shape=(input_width, 140))
	
	x = inputs
	for _ in range(num_hidden_layers):
		x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
		x = layers.MaxPooling1D((2))(x)
		x = layers.Dropout(dropout_rate)(x)
	
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation='relu')(x)
	x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def audio_lstm(num_classes, num_hidden_layers, dropout_rate=0.5, max_len=10):
	inputs = tf.keras.Input(shape=(max_len, 140)) 
	
	x = inputs
	for _ in range(num_hidden_layers):
		x = layers.LSTM(64, return_sequences=True)(x)
		x = layers.Dropout(dropout_rate)(x)
	
	x = layers.LSTM(64)(x)
	x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model
