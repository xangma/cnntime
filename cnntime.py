'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from numpy import random
from keras.models import model_from_json
from tqdm import tqdm
import os
import numpy as np
from glob import glob
import sklearn.utils
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import copy

#SETTINGS
remake_data = True # Saving doesn't work at the moment, so setting this to False is bad.
data_augmentation = True # This must be kept True for the moment, also there's not much reason to turn it off.
MULTIGPU = False # This is in development ...
datadir = '/mnt/lustre/moricex/MGPICOLAruns' # Where the runs are stored
simstouse = '/voxelised_oldnorm_500/*' # Voxelised sims. voxelised_oldnorm/ is freq densities, voxelised/ is normed freqs. Sum to 1.
traintestsplit = [0.9,0.1] # Train test split for binary classification
numsamples = 700 # Num sims to use, bear in mind these will be augmented (so each one is 24 'unique' sims)

# MODEL SETTINGS
load_model = False # Load a model? 
modelname = 'test_model'
num_classes = 2 # This must be 2 for the moment
batch_size = 10
epochs = 10

print('--------')
print('Program start')
print('--------')
print('SETTINGS')
print('--------')
print('Remake data? %s' %remake_data)
print('Data augmentation? %s' %data_augmentation)
print('MultiGPU? %s' %MULTIGPU)
print('Directory runs are in: %s' %datadir)
print('Which sims to use: %s' %simstouse)
print('Train test split: %s' %traintestsplit)
print('Number of samples to use %s' %numsamples)
print('--------')
print('MODEL SETTINGS')
print('--------')
print('Load model? %s' %load_model)
print('Model name: %s' %modelname)
print('Number of classes: %s' %num_classes)
print('Batch size: %s' %batch_size)
print('Epochs: %s' %epochs)
print('--------')

def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no. [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=2):
    """Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor, 
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []
    for g in range(n_gpus):
        with tf.device('/device:SYCL:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])

def rotations24(polycube,y):
    # imagine shape is pointing in axis 0 (up)
    # 4 rotations about axis 0
    data=[]
    datay=[]
    rot1,rot1y=rotations4(polycube, 0,y)
    data.append(rot1)
    datay.append(rot1y)
    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    rot2,rot2y=rotations4(rot90(polycube, 2, axis=1), 0,y)
    data.append(rot2)
    datay.append(rot2y)
    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    rot3,rot3y=rotations4(rot90(polycube, axis=1), 2,y)
    data.append(rot3)
    datay.append(rot3y)
    rot4,rot4y=rotations4(rot90(polycube, -1, axis=1), 2,y)
    data.append(rot4)
    datay.append(rot4y)
    
    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    rot5,rot5y=rotations4(rot90(polycube, axis=2), 1,y)
    data.append(rot5)
    datay.append(rot5y)
    rot6,rot6y=rotations4(rot90(polycube, -1, axis=2), 1,y)
    data.append(rot6)
    datay.append(rot6y)
    return data, datay

def rotations4(polycube, axis,y):
    """List the four rotations of the given cube about the given axis."""
    data=[]
    datay=[]
    for i in range(4):
        data.append(rot90(polycube, i, axis))
        datay.append(y)
    return data,datay

def rot90(m_, k=1, axis=2):
    m = copy.copy(m_)
    """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m

def data_generator(x,y):
#     batch_features = np.zeros((len(x), 64, 64, 64, 1))
#     batch_labels = np.zeros((len(y),2))
    dataarr=[]
    dataarry=[]
    for i in tqdm(range(len(x))):
#        print('Augmenting cube %s' %i)
        data,datay = rotations24(x[i],y[i])
        dataarr.append(data)
        dataarry.append(datay)
    return dataarr,dataarry

def save_model(model, modelname):
    model_json = model.to_json()
    with open(modelname, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelname+'_weights.h5')
    print("Saved model to disk")

def load_model(modelname):
    json_file = open(modelname, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelname+'_weights.h5')
    print("Loaded model from disk")


# READ IN DATA
if 'x_train' in locals(): # Check if data is loaded, else load it (Might remove this)
    print('Data already loaded, skipping')
    print('--------')
#else:
##os.chdir('MGPICOLAruns')
#    if remake_data == False:
#        x_train,x_test = np.load('cats_x.npy')
#        y_train,y_test = np.load('cats_y.npy')
if remake_data==True:
    print('Remaking data')
    print('--------')
    os.chdir(datadir)
    cwd = os.getcwd()
    dirs = os.listdir(cwd)

    print(dirs)
    print('--------')
    print('Which runs would you like to include?')
    print('--------')
#    run_nums = np.array(input())
    run_names=[]
#    for i in run_nums:
#        run_names.append(dirs[i])
    run_names.append(dirs[0])
    run_names.append(dirs[1])
    XX=[]
    print(len(run_names))
    for i in range(len(run_names)):
        fnames = glob(cwd+'/'+run_names[i]+simstouse)
        arrays = [np.load(f) for f in fnames]
        XX.extend(arrays)
    
    XX = [item[0] for item in XX]
    if num_classes == 2:
    	yy = np.hstack((np.zeros(np.int(len(XX)/2)),np.ones(np.int(len(XX)/2))))
    
    XX,yy = sklearn.utils.shuffle(XX,yy,random_state=2000)
    
    x_train=XX[0:np.int(numsamples*traintestsplit[0])]
    x_test=XX[np.int(numsamples*traintestsplit[0]):np.int((numsamples*traintestsplit[0])+(numsamples*traintestsplit[1]))]
    y_train=yy[0:np.int(numsamples*traintestsplit[0])]
    y_test=yy[np.int(numsamples*traintestsplit[0]):np.int((numsamples*traintestsplit[0])+(numsamples*traintestsplit[1]))]
    del XX,yy
    x_train = np.float32(x_train)
    x_test = np.float32(x_test)
    # Convert class vectors to binary class matrices.
#    y_train = keras.utils.to_categorical(y_train, num_classes)
#    y_test = keras.utils.to_categorical(y_test, num_classes)
    #x_train=list(x_train)
    y_train=np.float32(y_train)
    #x_test=list(x_test)
    y_test=np.float32(y_test)
    if data_augmentation == True:    
        with tf.device('/device:SYCL:0'):
            print('data_augmentation == True. Rotating sims ...')
#            for i in range(180):
#                print('Rotating x_train %s' %i)
#                for k in [1,2]:
#                    for ax in [0,1,2]:
#                        in_ =  rot90(x_train[i], k=k, axis=ax)
#        #                y_ = rot90(y_train[i], k=k, axis=ax)
#                        x_train=np.concatenate((x_train,[in_]))
#                        y_train=np.vstack((y_train,y_train[i]))
#            for i in range(20):
#                print('Rotating x_test %s' %i)
#                for k in [1,2]:
#                    for ax in [0,1,2]:
#                        in_ =  rot90(x_test[i], k=k, axis=ax)
#        #                y_ = rot90(y_train[i], k=k, axis=ax)
#                        x_test=np.concatenate((x_test,[in_]))
#                        y_test=np.vstack((y_test,y_test[i]))
            trainaug,trainaugy=data_generator(x_train,y_train)
            trainaug=[item for sublist in trainaug for item in sublist]
            trainaug=[item for sublist in trainaug for item in sublist]
            trainaugy=[item for sublist in trainaugy for item in sublist]
            trainaugy=[item for sublist in trainaugy for item in sublist]
            testaug,testaugy=data_generator(x_test,y_test)
            testaug=[item for sublist in testaug for item in sublist]
            testaug=[item for sublist in testaug for item in sublist]
            testaugy=[item for sublist in testaugy for item in sublist]
            testaugy=[item for sublist in testaugy for item in sublist]
    x_train,y_train=np.array(trainaug),np.array(trainaugy)
    x_test,y_test=np.array(testaug),np.array(testaugy)
    del trainaug, trainaugy, testaug, testaugy
    newxtrlen=len(x_train)
    newytrlen=len(y_train)
    newxtelen=len(x_test)
    newytelen=len(y_test)
    x_train,y_train = sklearn.utils.shuffle(x_train,y_train,random_state=2001)
    x_test,y_test = sklearn.utils.shuffle(x_test,y_test,random_state=2002)
    x_train=x_train.reshape(newxtrlen,64,64,64,1)
    x_test=x_test.reshape(newxtelen,64,64,64,1)
    print('--------')
    print('Saving remade data ...')
#    np.save('cats_x',[x_train,x_test])
#    np.save('cats_y',[y_train,y_test])
    print('Saving complete')
    print('--------')
# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train=np.array(x_train)
#y_train=np.array(y_train)
#x_test=np.array(x_test)
#y_test=np.array(y_test)


newxtrlen=len(x_train)
newytrlen=len(y_train)
newxtelen=len(x_test)
newytelen=len(y_test)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('--------')

#with tf.device('/device:SYCL:0'):
with tf.device('/cpu:0'):
    model = Sequential()
    
    act = keras.layers.advanced_activations.LeakyReLU(alpha=0.01)
#    act = Activation('relu')
    model.add(Conv3D(2, ([3,3,3]), input_shape=(64,64,64,1)))
    model.add(act)
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(6, ([4,4,4])))
    model.add(act)
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(6, ([9,9,9])))
    model.add(act)
    model.add(BatchNormalization())
    model.add(Conv3D(1, ([3,3,3])))
    model.add(act)
    model.add(BatchNormalization())
#    model.add(Conv3D(2, ([2,2,2])))
#    model.add(act)
#    model.add(BatchNormalization())
    model.add(Conv3D(1, ([2,2,2])))
    model.add(act)
    model.add(BatchNormalization())
    model.add(Flatten())
#    model.add(Dense(1024))
#    model.add(act)
#    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(act)
#    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    opt = keras.optimizers.SGD(lr=0.001)
    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#mygenerator=generator(x_train.reshape(newxtrlen,64,64,64),y_train)
#x_train = x_train.astype('float32')
#x_test = np.array(x_test).astype('float32')

if MULTIGPU == True:
    model = to_multi_gpu(model)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])    

#with tf.device('/device:SYCL:0'):
with tf.device('/cpu:0'):
    if data_augmentation:
        print('Using data augmentation.')
        model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
        save_model(model, modelname)
#save model

#    else:
#        model.fit_generator(mygenerator, steps_per_epoch = 2000, epochs = 50, verbose=2, callbacks=[], validation_data=(x_test, y_test), class_weight=None, workers=1)
#    else:
#        print('Using real-time data augmentation.')
#        # This will do preprocessing and realtime data augmentation:
#        datagen = ImageDataGenerator(
#            featurewise_center=False,  # set input mean to 0 over the dataset
#            samplewise_center=False,  # set each sample mean to 0
#            featurewise_std_normalization=False,  # divide inputs by std of the dataset
#            samplewise_std_normalization=False,  # divide each input by its std
#            zca_whitening=False,  # apply ZCA whitening
#            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#            horizontal_flip=True,  # randomly flip images
#            vertical_flip=False)  # randomly flip images
#    
#        # Compute quantities required for feature-wise normalization
#        # (std, mean, and principal components if ZCA whitening is applied).
#        datagen.fit(x_train)
#    
#        # Fit the model on the batches generated by datagen.flow().
#        model.fit_generator(datagen.flow(x_train, y_train,
#                                         batch_size=batch_size),
#                            steps_per_epoch=x_train.shape[0] // batch_size,
#                            epochs=epochs,
#                            validation_data=(x_test, y_test))
