# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:45:54 2019

@author: uqmbran
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
# from tensorflow.losses import absolute_difference
# tf.compat.v1.disable_eager_execution()
#Set up the parameters

K.clear_session()
#The number of data feed into system when training. (The bigger batchsize, the more memory consumed)
batch_size = 20 # Can do 32 on older version of ft, otherwwise 25
validation_split = 0.2
N = 256
iterations = 100


# loss_function = 'mean_squared_error'
metric_function = 'mean_absolute_error'
loss_function = 'mean_squared_error'

early_stop_min_delta = 0.001
early_stop_patience = 15

plotting_threshold = 0.7

@tf.function(autograph=False)
def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in tf.range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

#Unet network structure
def get_can_keras(img_rows, img_cols, num_channels, learning_rate):
    #Take the dimension of the image and feed into the unet
    #The input:
    #img_rows -> image height
    #img_cols -> image width
    #number_channels      -> Image channels
    inputs = Input((img_rows, img_cols, num_channels))
    
    # Get initializer
    #initializer = tf.function(tf.keras.initializers.Orthogonal(), autograph=False) # <== This is what prevents the OOM (comment it out to test)
    
    DEPTH = 32
    
    padding_size = 'same'
    
    
    # First convolution
    conv1 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=1, kernel_initializer=identity_initializer())(inputs)

    # Second convolution
    conv2 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=2, kernel_initializer=identity_initializer())(conv1)
    
    # Third convolution76
    conv3 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=4, kernel_initializer=identity_initializer())(conv2)
    
    # Fourth convolution
    conv4 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=8, kernel_initializer=identity_initializer())(conv3)
    
    # Fourth convolution
    conv5 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=16, kernel_initializer=identity_initializer())(conv4)
    
    # Sixth convolution
    conv6 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=32, kernel_initializer=identity_initializer())(conv5)
    
    # Seventh convolution
    conv7 = Conv2D(DEPTH, (3,3), activation=lrelu, padding=padding_size, dilation_rate=64, kernel_initializer=identity_initializer())(conv6)
    
    # Eighth convolution
    outputs = Conv2D(num_channels, (1,1), activation=None, padding=padding_size)(conv7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr = learning_rate), loss=loss_function, metrics=[metric_function])
    model.summary()
    tf.keras.utils.plot_model(model, 'my_first_model.png')
    return model

def normalizer(image):
    return (image - image.mean()) / image.std()

if __name__ == '__main__':
    
    import filenames

    import nibabel as nib #loading Nifti images
    from tqdm import tqdm
    import time
    import matplotlib.pyplot as plt
    
    # Basic model info
    num_channels = 1
    learning_rate = 0.001
    
    # Load some data to get the shape of inputs
    # Create file for result
    result_file = open("output_can/result.txt", "w+")
    
    # Load target (label) data
    labelPath = "slices_pad/"
    # MR Data
    artefactPath = "slices_artefact/"
    
    # Get list of filenames and case IDs from path where 3D volumes are
    groundList, caseList = filenames.getSortedFileListAndCases(labelPath, 0, "*.nii.gz", True)
    # Get image size
    image_rows, image_cols = nib.load(groundList[0]).get_data().astype(np.float32).shape
    print("Images shape:", image_rows, ", ", image_cols)
    
    
    
    
    # Generate undersampled images and place into required directory
    # undersampler(2.5, image_rows, labelPath, artefactPath)
    
    # Load MR Data and its labels
    
    # Get list of filenames and case IDs from path where 3D volumes are
    artefactList, caseList = filenames.getSortedFileListAndCases(artefactPath, 0, "*.nii.gz", True)
    
    # Check the artefact images match ground images
    if len(groundList) != len(artefactList):
        print("Warning:Images and labels don't match!")
        
        
        
        
        
    # Create 3D arrays to store all images and labels
    artefacts = np.ndarray((len(artefactList), image_rows, image_cols), dtype=np.float32)
    grounds = np.ndarray((len(groundList), image_rows, image_cols), dtype=np.float32)
    
    # Set train and test count
    # trainCount = int(np.ceil(len(groundList) * 0.95))
    trainCount = 1200
    testCount = int(len(groundList) - trainCount)
    
    # Create 3D arrays to store training images and labels
    trainArtefacts = artefacts[0:trainCount, :, :]
    trainGround = grounds[0:trainCount, :, :]
    
    # Create 3D arrays to store test images and labels
    testArtefacts = artefacts[trainCount:-1, :, :]
    testGround = grounds[trainCount:-1, :, :]
    
    i = 0
    count = 0
    outCount  = 0
    
    # Load each image with its original image and case number together
    for artefact, ground, case in tqdm(zip(artefactList, groundList, caseList)):
        
        # Load MR nifti file
        img = normalizer(nib.load(artefact).get_data().astype(np.float32))
        
        # Load ground nifti file
        label = normalizer(nib.load(ground).get_data().astype(np.float32))
        
        # If within the training portion
        if count < trainCount:
            trainArtefacts[count] = img
            trainGround[count] = label
        else:
            testArtefacts[count - trainCount] = img
            testGround[count - trainCount] = label
        
        # Fill full arrays
        artefacts[count] = img
        grounds[count] = label
        
        count += 1
        
        if count == len(groundList) - 1:
            break
        
    # Add the extra axis for "channel" data
    artefacts = artefacts[..., np.newaxis]
    grounds = grounds[..., np.newaxis]
    trainArtefacts = trainArtefacts[..., np.newaxis]
    trainGround = trainGround[..., np.newaxis]
    testArtefacts = testArtefacts[..., np.newaxis]
    testGround = testGround[..., np.newaxis]
    print("Used", trainCount, "images for training")
    
    # Training label image
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(trainArtefacts[0,:,:,0])
    ax[1].imshow(trainGround[0,:,:,0])
    ax[0].set_title("Training Input")
    ax[1].set_title("Training Groundtruth")
    fig.tight_layout()
    output_path = "output_can/sample_image_tissue_"+'.png'
    fig.savefig(output_path)
    
    ## Training stage
    
    # Start training
    print("Training ...")
    result_file.write("Training with " + str(iterations) + " iterations\n")
    start = time.time()
    model = get_can_keras(image_rows, image_cols, num_channels, learning_rate)
    
    # Set up callback function, used to save best params
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss',
                                        save_best_only=True)
    # Callback for early stop
    earlystopper = EarlyStopping(monitor='loss',
                                  min_delta=early_stop_min_delta,
                                  patience=early_stop_patience, verbose=1)
    # Callback to tensorboard
    tenor_board = TensorBoard(log_dir=("logs/" +str(time.time())))
    # keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
    model.fit(trainArtefacts, trainGround, batch_size=batch_size, epochs=iterations,
              verbose=1, shuffle=True, validation_split=validation_split,
              callbacks=[earlystopper,model_checkpoint])
    
    end = time.time()
    elapsed = end - start
    elapsed = round(elapsed, 2)
    print("Training took " +str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    result_file.write("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total\n")
    
    # Restore the weights
    model.load_weights('./weights.h5')
    
    
    
    # Testing Stage
    print("Testing ...")
    start = time.time()
    reconstruction = model.predict(testArtefacts, verbose=1)
    end = time.time()
    elapsed = end - start
    elapsed = round(elapsed, 2)
    print("Prediction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    result_file.write("Prediction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total\n")
    
    # Plot prediction results
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    for i in range(0, testCount-1):
        index = int(i)
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
        ax[0].imshow(testArtefacts[index,:,:,0], aspect="auto")
        ax[1].imshow(testGround[index,:,:,0], aspect="auto")
        mask = reconstruction[index,:,:,0]
        ax[2].imshow(mask, aspect="auto")
        ax[0].set_title("Input")
        ax[1].set_title("Ground truth")
        ax[2].set_title("Prediction")
        fig.tight_layout()
        output_path = "output/test_number_"+str(index)+'.png'
        fig.savefig(output_path)
        plt.close(fig)

    #Close the result file 
    result_file.close()
    