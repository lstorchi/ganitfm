import math
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from IPython.core.display import Image

from ann_visualizer.visualize import ann_viz

from typing import List, Tuple

##########################################################################################
# Print iterations progress
#https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, \
                      length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

##########################################################################################

def create_binary_list_from_int(number: int) -> List[int]:

    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]

##########################################################################################

def generate_even_data(max_int: int, batch_size: int=16) -> Tuple[int, List[int], \
                                                                  List[List[int]], \
                                                                  List[int]]:
    
    gennum = 0
    realvalue = []
    labels = []
    data = []
    while gennum < batch_size:
        s = random.randint(0, max_int)
        if s % 2 == 0:
            gennum += 1
            realvalue.append(s)
            labels.append(1)
            data.append(create_binary_list_from_int(s))
            #print(s, create_binary_list_from_int(s))
        else:
            continue

    maxlen = 0
    for d in data:
        if maxlen < len(d):
            maxlen = len(d)
    
    # nomalize to maxlen
    retdata = []
    for idx, d in enumerate(data):
        if len(d) < maxlen:
            rd = [0] * (maxlen - len(d)) + d
            retdata.append(rd)
        else:
            retdata.append(d)

    return maxlen, labels, retdata, realvalue

##########################################################################################

def discriminator_model (input_length: int):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_length, activation='relu', \
                                    input_dim=input_length))
    #model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model    

##########################################################################################

def generator_model (input_length: int):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_length, activation='relu', \
                                    input_dim=input_length))
    #model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(input_length, activation='sigmoid'))

    return model

##########################################################################################

def discriminator_loss (real_output, fake_output):

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss

##########################################################################################

def generator_loss (fake_output):

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)

##########################################################################################

def train_step (real_data, batch_size, inputsize, generator, discriminator, 
                generator_loss, discriminator_loss ):
    noise = tf.random.uniform([batch_size, inputsize], 0, 1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_data = generator(noise, training=True)

        #print("Real Data: ", real_data)
        #print("Fake Data: ", fake_data)

        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,\
                                                generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, \
                                                    discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator,\
                                             generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,\
                                                 discriminator.trainable_variables))

    return 

##########################################################################################

if __name__ == "__main__":
    
    debug = False

    #val = 1
    #lval = create_binary_list_from_int(val)
    #print(f"Binary representation of {val} is {lval}")

    inputsize, labels, data, realvalue = generate_even_data(1000, 100)

    if debug:
        for idx, d in enumerate(data):
            print("Labels: %2d"%(labels[idx]), \
                  "RealValue: %5d"%(realvalue[idx]), d)

    gen_model = generator_model(inputsize)
    dis_model = discriminator_model(inputsize)

    if debug:
        #plot_model(gen_model, show_shapes=True)
        #plot_model(dis_model, show_shapes=True)
        ann_viz(gen_model, title="Generator Model", view=True, filename="gen_model.gv")
        ann_viz(dis_model, title="Discriminator Model", view=True, filename="dis_model.gv")


        # before training
        for i in range(100):
            noise = tf.random.uniform([1, inputsize], 0, 1)
            print("Noise: ", np.array(noise)[0])
            generated_data = gen_model(noise, training=False)
            print(generated_data)
            print("Generated: ", np.array(generated_data)[0])
            disout = dis_model(generated_data, training=False)
            print("Discriminator: ", np.array(disout)[0])

    # training
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # before training
    for i in range(5):
        noise = tf.random.uniform([1, inputsize], 0, 1)
        #print("Noise: ", np.array(noise)[0])
        generated_data = gen_model(noise, training=False)
        #print("Generated: ", np.array(generated_data)[0])
        disout = dis_model.predict(generated_data)
        print("Discriminator: ", disout)

    #for d in data:
    #    t_d = tf.convert_to_tensor([d], dtype=tf.float32)
    #    disout = dis_model.predict(t_d)
    #    print("Discriminator: ", disout)

    batchsize = 5
    epochs = 100

    printProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for epoch in range(epochs):
        real_data = []
        idx = 0
        for d in data:
            real_data.append(d)
            idx += 1
            if idx == batchsize:
                idx = 0
                #print(tf.convert_to_tensor(real_data))

                t_real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)

                train_step(t_real_data, batchsize, inputsize, \
                           gen_model, dis_model, \
                           generator_loss, discriminator_loss)
                
                real_data.clear()
        
        printProgressBar(epoch + 1, epochs, prefix = 'Progress:', \
                         suffix = 'Complete', length = 50)

    #for d in data:
    #    t_d = tf.convert_to_tensor([d], dtype=tf.float32)
    #    disout = dis_model.predict(t_d)
    #    print("Discriminator: ", disout)

    # after training
    for i in range(5):
        noise = tf.random.uniform([1, inputsize], 0, 1)
        #print("Noise: ", np.array(noise)[0])
        generated_data = gen_model(noise, training=False)
        #print("Generated: ", np.array(generated_data)[0])
        disout = dis_model.predict(generated_data)
        print("Discriminator: ", disout)

