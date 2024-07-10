import os
import numpy as np
import config as cfg
from gui.gui import update_train_plot, update_val_plot, init_gui
import gc
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, 
    Concatenate, 
    Conv2D, 
    Conv2DTranspose, 
    LeakyReLU, 
    Activation, 
    Dropout, 
    BatchNormalization
)
from matplotlib import pyplot as plt
from tensorflow.keras.losses import mean_squared_error
from .stn import Localization, BilinearInterpolation

class colors:
	BLUE = '\033[34m'
	GREEN = '\033[32m'
	RED = '\033[31m'
	YELLOW = '\033[33m'
	CYAN = '\033[36m'
	ENDC = '\033[0m'

######################################################################################
# Data Loading / Preprocess
######################################################################################
def load_imgs(path, size=(256,256)):
	src_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in os.listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size, color_mode='rgba')
		# convert to numpy array
		pixels = img_to_array(pixels)
		src_list.append(pixels)
	return np.asarray(src_list)

def preprocess_data(data):
    # Ensure data has at least 3 elements
    if len(data) < 3:
        raise ValueError("Expected at least 3 arrays in data.")
    
    # Unpack arrays based on the length of data
    X1 = data[0]
    X2 = data[1]
    X3 = data[2]
    
    # If there's a fourth element, unpack it; otherwise, set it to None
    X4 = data[3] if len(data) > 3 else None
    
    # Scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5
    
    # Check if X4 is not None, then scale it as well
    if X4 is not None:
        X4 = (X4 - 127.5) / 127.5
    
    # Return the scaled arrays in a list
    if X4 is not None:
        return [X1, X2, X3, X4]
    else:
        return [X1, X2, X3]

######################################################################################
# Define the discriminator function - Based on PatchGAN Classifier
######################################################################################
def define_discriminator(image_shape, lr):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)  # Generated
	# target image input
	in_target_image = Input(shape=image_shape)  # GT 
	# concatenate images, channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	"""
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	"""
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(learning_rate=lr, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

######################################################################################
# Defining generator - Based on UNet 
######################################################################################
# Setup Encoder block (Encoder being a block in the FIRST half of the model)
def encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# Setup Decoder block (Decoder being a block in the SECOND half of the model)
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)

	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate(axis=-1)([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

def depth_aware_loss(y_true, y_pred):
    # Assuming y_true is the ground truth and y_pred is the generated output
    input_images = tf.concat([y_true, y_pred], axis=-1)
    
    in_depth = input_images[..., -1]  # Extract the depth map from the input
    threshold_depth = 100 / 255.0  # Adjust this threshold value based on dataset

    # Create a binary mask based on the threshold
    depth_mask = tf.cast(tf.math.greater(in_depth, threshold_depth), tf.float32)
    depth_mask = tf.expand_dims(depth_mask, axis=-1)  # Expand the mask dimension to match y_true and y_pred

    # Apply the depth mask to the predicted and target images
    masked_pred = y_pred * depth_mask
    masked_true = y_true * depth_mask

    # Image-based loss (e.g., mean squared error)
    image_loss = mean_squared_error(masked_true, masked_pred)

    # Depth-aware loss: penalize the model for rendering the foreground in lighter depth regions
    depth_penalty = tf.math.abs(masked_pred - masked_true) * depth_mask

    # Combine image loss and depth-aware loss
    total_loss = image_loss + 0.5 * tf.reduce_mean(depth_penalty)  # Adjust weights as needed

    return total_loss

# Generator UNet Architecture
def define_generator(image_shape, use_stn):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape, name='background_input')
	in_obj = Input(shape=image_shape, name='object_input')
	in_depth = Input(shape=image_shape, name='depth_input')

    # Combine features & Run STN on foreground
	if use_stn:
		theta = Localization()(in_obj)
		transformed_obj = BilinearInterpolation(height=256, width=256)([in_obj, theta])
		combined_features = Concatenate()([in_image, transformed_obj])
	else:
		combined_features = Concatenate()([in_image, in_obj, in_depth])
	
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = encoder_block(combined_features, 64, batchnorm=False)
	e2 = encoder_block(e1, 128)
	e3 = encoder_block(e2, 256)
	e4 = encoder_block(e3, 512)
	e5 = encoder_block(e4, 512)
	e6 = encoder_block(e5, 512)
	e7 = encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(1024, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD512-CD512-C512-C256-C128-C64
	d1 = decoder_block(b, e7, 512, dropout=False)
	d2 = decoder_block(d1, e6, 512, dropout=False)
	d3 = decoder_block(d2, e5, 512, dropout=False)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
		
    # Output
	g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model([in_image, in_obj, in_depth], out_image)

	return model

######################################################################################
# Define the combined GAN Model
######################################################################################

def define_gan(g_model, d_model, image_shape, lr):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
            
	# define the source image
	in_src = Input(shape=image_shape)
	in_obj = Input(shape=image_shape)
	in_depth = Input(shape=image_shape)
	# suppy the image as input to the generator 
	gen_out = g_model([in_src, in_obj, in_depth])
	# supply the input image and generated image as inputs to the discriminator
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and disc. output as outputs
	model = Model([in_src, in_obj, in_depth], [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=lr, beta_1=0.5)
    #Total loss is the weighted sum of adversarial loss (BCE),  L1 loss (MAE) and Depth Aware Loss (DAL)
	model.compile(loss=['binary_crossentropy', 'mae', depth_aware_loss], optimizer=opt, loss_weights=[1,100, 100])
	return model

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	if len(dataset) == 4:
		backgrounds, paired, objects, depth = dataset
		ix = np.random.randint(0, backgrounds.shape[0], n_samples)
		X1, X2, X3, X4 = backgrounds[ix], paired[ix], objects[ix], depth[ix]
	elif len(dataset) == 3:
		backgrounds, objects, depth = dataset
		paired = np.zeros_like(objects)  # Dummy paired data if not used
		ix = np.random.randint(0, backgrounds.shape[0], n_samples)
		X1, X2, X3, X4 = backgrounds[ix], paired[ix], objects[ix], depth[ix]

	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2, X3, X4], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, backgrounds, objects, depth, patch_shape):
	# generate fake instance
	X = g_model([backgrounds, objects, depth])
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# GANS do not converge, so we save some samples as a plot and the model periodically
def summarize_performance(g_model, dataset, n_samples, curr_epoch, curr_step):
	# select a sample of input images
	[backgrounds, paired, objects, depth], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	generated, _ = generate_fake_samples(g_model, backgrounds, objects, depth, 1)
	# scale all pixels from [-1,1] to [0,1]
	backgrounds = (backgrounds + 1) / 2.0
	paired = (paired + 1) / 2.0
	generated = (generated + 1) / 2.0
	objects = (objects + 1) / 2.0
	
	# plot real source images
	for i in range(n_samples):
		plt.subplot(4, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(backgrounds[i])
		#plt.title('Background')
    # plot generated target image
	for i in range(n_samples):
		plt.subplot(4, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(generated[i])
		#plt.title('Generated')
    # plot real target image
	for i in range(n_samples):
		plt.subplot(4, n_samples, 1 + n_samples * 2 + i)
		plt.axis('off')
		plt.imshow(paired[i])
		#plt.title('Paired')
	for i in range(n_samples):
		plt.subplot(4, n_samples, 1 + n_samples * 3 + i)
		plt.axis('off')
		plt.imshow(objects[i])
		#plt.title('Paired')
	if not os.path.exists(cfg.PLOTS_DIR):
		os.makedirs(cfg.PLOTS_DIR)
	# save plot to file
	filename1 = f'{cfg.PLOTS_DIR}/plot_{int(curr_epoch)}.png'
	plt.savefig(filename1)
	plt.close()

	# Define the directory path
	# Check if the directory exists, if not, create it
	if not os.path.exists(cfg.MODEL_DIR):
		os.makedirs(cfg.MODEL_DIR)
	# save the generator model
	g_model.save(f'{cfg.MODEL_DIR}/g_model_epoch{int(curr_epoch)}.h5')
	#g_model.save(f'logs/cgan/models/g_model/g_model.h5')
	# save the discriminator model
	# save the GAN model
	print(f'{colors.YELLOW}Saved generator at epoch: {int(curr_epoch)}{colors.ENDC}')

######################################################################################
# Train function
######################################################################################

def train(d_model, g_model, gan_model, dataset, val_dataset=None):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	backgrounds, paired, objects, depth = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(backgrounds) / cfg.BATCH_SIZE)
	# calculate the number of training iterations
	n_steps = bat_per_epo * cfg.EPOCHS
	# start GUI 
	init_gui()
	# manually enumerate epochs
	for i in range(n_steps):
			# select a batch of real samples
			[xback, xpaired, xobj, xdep], y_real = generate_real_samples(dataset, cfg.BATCH_SIZE, n_patch)
			# generate a batch of fake samples
			xgen, y_fake = generate_fake_samples(g_model, xback, xobj, xdep, n_patch)
			# update discriminator for real samples
			d_loss1 = d_model.train_on_batch([xback, xpaired], y_real)
			# update discriminator for generated samples
			d_loss2 = d_model.train_on_batch([xback, xgen], y_fake)
			d_loss = d_loss1 + d_loss2
			# update the generator
			g_loss, _, _ = gan_model.train_on_batch([xback, xobj, xdep], [y_real, xpaired])
			# calc current steps/epochs
			curr_step = i+1
			curr_epoch = (curr_step)/(n_steps/cfg.EPOCHS)
			# update GUI
			update_train_plot(curr_epoch, curr_step, d_loss, g_loss)
			# Validation step
			if cfg.USE_VAL:
				if (i+10) % cfg.VAL_FREQUENCY == 0:
            		# Select a batch of validation samples
					[val_backgrounds, val_paired, val_objects, val_maps], _ = generate_real_samples(val_dataset, cfg.VAL_SAMPLES, n_patch)
					val_loss = gan_model.evaluate([val_backgrounds, val_objects, val_maps], [np.ones((cfg.VAL_SAMPLES, n_patch, n_patch, 1)), val_paired], verbose=0)
					# Predict an image for the validation plot
					generated = g_model.predict([np.expand_dims(val_backgrounds[np.random.randint(0, len(val_backgrounds))], axis=0), 
										 		np.expand_dims(val_objects[np.random.randint(0, len(val_objects))], axis=0), 
												np.expand_dims(val_objects[np.random.randint(0, len(val_maps))], axis=0)])
					
					generated = (generated + 1) / 2.0
					update_val_plot(curr_epoch, curr_step, val_loss, generated)

			# summarize model performance
			if (i+1) % (bat_per_epo * 1) == 0:
				summarize_performance(g_model, dataset, 3, curr_epoch, curr_step)