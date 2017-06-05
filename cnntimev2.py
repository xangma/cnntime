'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

#SETTINGS
remake_data = False # Saving doesn't work at the moment, so setting this to False is bad.
data_augmentation = True # This must be kept True for the moment, also there's not much reason to turn it off.
MULTIGPU = False # This is in development ...
datadir = '/mnt/lustre/moricex/MGPICOLAruns' # Where the runs are stored
#90datadir = '/users/moricex/PAPER2' # Where the runs are stored
simstouse = '/voxelised_500/*' # Voxelised sims. voxelised_oldnorm/ is freq densities, voxelised/ is normed freqs. Sum to 1.
traintestsplit = [0.8,0.2] # Train test split for binary classification
numsamples = 1000 # Num sims to use when remaking data, bear in mind these will be augmented (so each one is 24 'unique' sims)
ncats=1 # Cats to split when remaking + saving data

# MODEL SETTINGS
load_old_model = True # Load a model? 
modelname = 'TEST_RUN_voxelised_softmax'
num_classes = 2 # This must be 2 for the moment
batch_size = 1 # If you change this you have to remake_data, because the chunk size of the hdf5 file will need to change
steps_per_epoch = 15000
epochs = 10
validation_steps=5000

import tables
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
import h5py
import threading

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import copy

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

f_storage,l_storage = [],[]

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
    rot1,rot1y=rotations4(polycube, 0,y).__next__()
    data.extend(rot1)
    datay.extend(rot1y)
    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    rot2,rot2y=rotations4(rot90(polycube, 2, axis=1).__next__(), 0,y).__next__()
    data.extend(rot2)
    datay.extend(rot2y)
    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    rot3,rot3y=rotations4(rot90(polycube, axis=1).__next__(), 2,y).__next__()
    data.extend(rot3)
    datay.extend(rot3y)
    rot4,rot4y=rotations4(rot90(polycube, -1, axis=1).__next__(), 2,y).__next__()
    data.extend(rot4)
    datay.extend(rot4y)
    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    rot5,rot5y=rotations4(rot90(polycube, axis=2).__next__(), 1,y).__next__()
    data.extend(rot5)
    datay.extend(rot5y)
    rot6,rot6y=rotations4(rot90(polycube, -1, axis=2).__next__(), 1,y).__next__()
    data.extend(rot6)
    datay.extend(rot6y)
    yield data, datay

def rotations4(polycube, axis,y):
    """List the four rotations of the given cube about the given axis."""
    data=[]
    datay=[]
    for i in range(4):
        data.append(rot90(polycube, i, axis).__next__())
        datay.append(y)
    yield data,datay

def rot90(m_, k=1, axis=2):
    m = copy.copy(m_)
    """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    yield m

def save_model(model, modelname):
    model_json = model.to_json()
    with open(modelname, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelname+'_weights.h5', overwrite=True)
    print("Saved model to disk")

def load_model(modelname):
    json_file = open(modelname, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelname+'_weights.h5')
    print("Loaded model from disk")
    return loaded_model

def save_data(x,y,hdf5_path,i,f_storage,l_storage):
    hdf5_file = h5py.File(hdf5_path, "a")
    if i == 0:
        f_storage = hdf5_file.create_dataset('features', data=x, maxshape=(None, 64, 64, 64),chunks=(batch_size, 64, 64, 64))
        l_storage = hdf5_file.create_dataset('labels', data=y, maxshape=(None,2),chunks=(batch_size,2))
    print(f_storage.shape[0],l_storage.shape[0])
    if i > 0:
        f_storage.resize(f_storage.shape[0]+len(x),axis=0)
        l_storage.resize(l_storage.shape[0]+len(y),axis=0)
        f_storage[np.int(f_storage.shape[0]-len(x)):] = x
        l_storage[np.int(l_storage.shape[0]-len(y)):] = y
    return f_storage,l_storage

class printbatch(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        if batch%10 == 0:
            print("Batch " + str(batch) + " ends")
    def on_epoch_begin(self, epoch, logs={}):
        print(logs)
    def on_epoch_end(self, epoch, logs={}):
        print(logs)

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def simpleGeneratortest(batch_size,steps_per_epoch,traintestsplit,trainortest):
    x_train = f.get('features')
    y_train = f.get('labels')
    total_examples = len(x_train)
    examples_at_a_time = batch_size
    range_examples = int(total_examples/examples_at_a_time)
    if data_augmentation == True:
        steps_per_epoch = 15000/24
    if trainortest == "train":
        while 1:
            used_sims=[]
            # To get batch_size working again, make the rand_sim vars a list the range of batch_size, check and replace elements that appear in used_sims, then for every element in that list, append the rotated cubes to an array, note the numbers down in used_sims, and yield all the sims in the batch. Should be scalable from batch_size = 1 upwards, but it might be nice to warn the user about chunk sizes when dealing with a large number of sims. You're probably going to want to remake your data with chunks=(batch_size,64,64,64) or whatever when you change batch_size.
            for i in range(range_examples): # samples
                rand_sim = np.random.randint(0,steps_per_epoch)
                rand_sim_rot = np.random.randint(0,24)
                while [rand_sim, rand_sim_rot] in used_sims:
                    rand_sim = np.random.randint(0,steps_per_epoch)
                    rand_sim_rot = np.random.randint(0,24)
                else:
                    results = rotations24(x_train[rand_sim*examples_at_a_time:(rand_sim+1)*examples_at_a_time], y_train[rand_sim*examples_at_a_time:(rand_sim+1)*examples_at_a_time]).__next__()
                    used_sims.append([rand_sim,rand_sim_rot])
                    yield results[0][rand_sim_rot].reshape(1,64,64,64,1),results[1][rand_sim_rot] 
    else:
        while 1:
            used_sims_test=[]
            for i in range(range_examples): # samples
                rand_sim = np.random.randint(0,(range_examples-steps_per_epoch))
                rand_sim_rot = np.random.randint(0,24)
                while [rand_sim, rand_sim_rot] in used_sims_test:
                    rand_sim = np.random.randint(0,(range_examples-steps_per_epoch))
                    rand_sim_rot = np.random.randint(0,24)
                else:
                    results = rotations24(x_train[np.int((rand_sim*examples_at_a_time)+(batch_size*steps_per_epoch)):np.int(((rand_sim+1)*examples_at_a_time)+(batch_size*steps_per_epoch))], y_train[np.int((rand_sim*examples_at_a_time)+(batch_size*steps_per_epoch)):np.int(((rand_sim+1)*examples_at_a_time)+(batch_size*steps_per_epoch))]).__next__()
                    used_sims_test.append([rand_sim,rand_sim_rot])
                    yield results[0][rand_sim_rot].reshape(1,64,64,64,1),results[1][rand_sim_rot] 
#sg = simpleGenerator(batch_size)
sg = simpleGeneratortest(batch_size,steps_per_epoch,traintestsplit,"train")
sgt = simpleGeneratortest(batch_size,steps_per_epoch,traintestsplit,"test")
pb = printbatch()

if (remake_data==False & load_old_model==False):
    hdf5_path = "/mnt/lustre/moricex/MGPICOLAruns/cat2.hdf5"
    f = h5py.File(hdf5_path, 'r', driver='stdio')
    dset = f["features"]
    totsamples=dset.shape[0]
    desiredsamples=((steps_per_epoch/24)*batch_size)+((validation_steps/24)*batch_size)
    if totsamples < desiredsamples:
        exit()
    print('Data file contains %s samples' %totsamples)
    print('You''ve asked for %s samples (train+test)' %desiredsamples)
else:
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
    run_names.append(dirs[1])
    run_names.append(dirs[2])
    XX,x_train=[],[]
    print(len(run_names))
    for i in range(len(run_names)):
        fnames = glob(cwd+'/'+run_names[i]+simstouse)
        arrays = [np.load(f) for f in fnames]
        XX.extend(arrays)
    XX = [item[0] for item in XX]
    if num_classes == 2:
    	yy = np.hstack((np.zeros(np.int(len(XX)/2)),np.ones(np.int(len(XX)/2))))
    XX,yy = sklearn.utils.shuffle(XX,yy,random_state=2000)
    XX = np.float32(XX)
    yy = np.float32(yy)
    # Convert class vectors to binary class matrices.
    yy = keras.utils.to_categorical(yy, num_classes)
    if data_augmentation == True:
        with tf.device('/device:SYCL:0'):
            print('data_augmentation == True. Rotating sims ...')
            XXaug,yyaug=[],[]
            hdf5_path = "cat2.hdf5"
            hdf5_file = h5py.File(hdf5_path, "w")
            for i in range(ncats):
                XXaugtemp, yyaugtemp = XX[np.int(i*(len(XX)/ncats)):np.int((i+1)*(len(XX)/ncats))],yy[np.int(i*(len(yy)/ncats)):np.int((i+1)*(len(yy)/ncats))]
                XXaugtemp, yyaugtemp = sklearn.utils.shuffle(XXaugtemp,yyaugtemp,random_state=np.random.random_integers(0,9999999))
                print('Saving cat %s' %i)
                f_storage,l_storage = save_data(XXaugtemp,yyaugtemp,hdf5_path,i,f_storage,l_storage)
            print('--------')
            print('Saving complete')
            hdf5_file.close()
            print('--------')
#            del XXaugtemp,yyaugtemp

#with tf.device('/device:SYCL:0'):
with tf.device('/cpu:0'):
    model = Sequential()
    
    act = keras.layers.advanced_activations.LeakyReLU(alpha=0.01)
#    act = Activation('relu')
    model.add(Conv3D(2, ([3,3,3]), input_shape=(64,64,64,1)))
    model.add(BatchNormalization())
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(AveragePooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(6, ([4,4,4])))
    model.add(BatchNormalization())
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(AveragePooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(6, ([9,9,9])))
    model.add(BatchNormalization())
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(Conv3D(1, ([3,3,3])))
    model.add(BatchNormalization())
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(Conv3D(2, ([2,2,2])))
    model.add(BatchNormalization())
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(Conv3D(1, ([2,2,2])))
    model.add(BatchNormalization())
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    opt = keras.optimizers.SGD(lr=0.001)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

if MULTIGPU == True:
    model = to_multi_gpu(model)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])    

if load_old_model == True:
    model=load_model(modelname)
    opt = keras.optimizers.Adam(lr=.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    opt = keras.optimizers.SGD(lr=0.001)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#with tf.device('/device:SYCL:0'):
with tf.device('/cpu:0'):
    if data_augmentation:
        print('Using data augmentation.')
#        model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
        model.fit_generator(sg, steps_per_epoch=steps_per_epoch, nb_epoch=epochs, verbose=1, validation_data=sgt, validation_steps=validation_steps,max_q_size=4,pickle_safe=False, workers=4)
        save_model(model, modelname)

