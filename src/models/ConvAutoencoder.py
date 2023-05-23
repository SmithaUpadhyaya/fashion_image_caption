from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import numpy as np

class ConvAutoencoder:

	@staticmethod
	def build(width, height, depth, filters = (32, 64), latentDim = 32):
	
    	# width, height, depth: initialize the input shape of the input. Width x Height x channel/depth
        # filters: tuple that contains the set of filters for convolution operations
        # latentDim: number of neurons in our fully-connected (Dense) latent vector. It's output the feature extracted, that will be passed to the decoder layer.

		inputShape = (height, width, depth)
		chanDim = -1

        # define the input to the encoder
		inputs = Input(shape = inputShape)
		x = inputs

        # loop over the number of filters
		for f in filters:

			# apply a CONV => RELU => BN operation
			x = Conv2D(f, (3, 3), strides = 2, padding = "same")(x)
			x = LeakyReLU(alpha = 0.2)(x)
			x = BatchNormalization(axis = chanDim)(x)

        # flatten the network and then construct our latent vector
		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x) #Extracted feature


		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")


        # start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))

		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv2DTranspose(f, (3, 3), strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha = 0.2)(x)
			x = BatchNormalization(axis = chanDim)(x)

        
        # apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid")(x)



		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")


        # our autoencoder is the encoder + decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")
		

        # return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)

        #https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/

