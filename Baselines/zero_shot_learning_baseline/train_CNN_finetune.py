#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Baseline codes for zero-shot learning task.
This python script is train a deep feature extractor (CNN).
The command is:     python train_CNN.py Animals True 0.05
The first parameter is the super-class.
The second parameter is the flag whether data preparation is to be implemented. You should choose True at the first running.
The third parameter is the learning rate of the deep network (MobileNet).
The trained model will be saved at 'model/mobile_Animals_wgt.h5'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

from keras import Model, optimizers
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


def main():
    # Parameters
    if len(sys.argv) == 4:
        superclass = sys.argv[1]
        imgmove = sys.argv[2]
        if imgmove == 'False':
            imgmove = False
        else:
            imgmove = True
        lr = float(sys.argv[3])
    else:
        print('Parameters error')
        exit()

    # The constants
    classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    date = '20180321'

    trainpath = 'trainval_'+superclass+'/train'
    valpath = 'trainval_'+superclass+'/val'

    if not os.path.exists('model'):
        os.mkdir('model')

    # Train/validation data preparation
    if imgmove:
        if not os.path.exists('trainval_'+superclass):
            os.mkdir('trainval_'+superclass)
            os.mkdir(trainpath)
            os.mkdir(valpath)
            sourcepath = '../zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_train_'+date\
                         +'/zsl_'+testName[superclass[0]]+'_'+str(superclass).lower()+'_train_images_'+date
            categories = os.listdir(sourcepath)
            for eachclass in categories:
                if eachclass[0] == superclass[0]:
                    print(eachclass)
                    os.mkdir(trainpath+'/'+eachclass)
                    os.mkdir(valpath+'/'+eachclass)
                    imgs = os.listdir(sourcepath+'/'+eachclass)
                    idx = 0
                    for im in imgs:
                        if idx%8 == 0:
                            shutil.copyfile(sourcepath+'/'+eachclass+
                                            '/'+im, valpath+'/'+eachclass+'/'+im)
                        else:
                            shutil.copyfile(sourcepath+'/'+eachclass+
                                            '/'+im, trainpath+'/'+eachclass+'/'+im)
                        idx += 1

    # Train and validation ImageDataGenerator
    batchsize = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        trainpath,
        target_size=(224, 224),
        batch_size=batchsize)

    valid_generator = test_datagen.flow_from_directory(
        valpath,
        target_size=(224, 224),
        batch_size=batchsize)

    base_model = VGG16(include_top=False, weights='imagenet')
    i=0
    for layer in base_model.layers:
        layer.trainable = False
        i = i+1
        print(i, layer.name)

    x = base_model.output
    x = Dense(128)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(classNum[superclass[0]], activation='softmax')(x)

    # Train Model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics=["accuracy"])
    # model.compile(optimizer=SGD(lr=lr, momentum=0.9),
    #               loss='categorical_crossentropy', metrics=['accuracy'])

    steps_per_epoch = int(train_generator.n/batchsize)
    validation_steps = int(valid_generator.n/batchsize)

    visual = TensorBoard(log_dir="model/", histogram_freq=0,
                batch_size=batchsize, write_graph=True)
    weightname = 'model/mobile_'+superclass+'_wgt_fin-{epoch:02d}-{val_loss:.4f}-{val_acc:.3f}.h5'

    checkpointer = ModelCheckpoint(weightname, monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='auto', period=1)

    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[checkpointer, visual])


if __name__ == "__main__":
    main()
