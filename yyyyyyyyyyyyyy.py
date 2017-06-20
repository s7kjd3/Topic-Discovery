import keras
from keras.layers import Dense

# input shape: (nb_samples, 32)
encoder = keras.layers.containers.Sequential([Dense(16, input_dim=32), Dense(8)])
decoder = keras.layers.containers.Sequential([Dense(16, input_dim=8), Dense(32)])

autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
model = Sequential()
model.add(autoencoder)

# training the autoencoder:
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, X_train, nb_epoch=10)

# predicting compressed representations of inputs:
autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
model.compile(optimizer='sgd', loss='mse')
representations = model.predict(X_test)

# the model is still trainable, although it now expects compressed representations as targets:
model.fit(X_test, representations, nb_epoch=1)  # in this case the loss will be 0, so it's useless

# to keep training against the original inputs, just switch back output_reconstruction to True:
autoencoder.output_reconstruction = True
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, X_train, nb_epoch=10)