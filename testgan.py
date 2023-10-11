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

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss

##########################################################################################

def generator_loss (fake_output):

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)

##########################################################################################
"""
class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))
    

class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))

def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
"""

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
        noise = tf.random.uniform([1, inputsize], 0, 2)
        print("Noise: ", np.array(noise)[0])
        generated_data = gen_model(noise, training=False)
        print("Generated: ", np.array(generated_data)[0])
        disout = dis_model(generated_data, training=False)
        print("Discriminator: ", np.array(disout)[0])

    # training
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


