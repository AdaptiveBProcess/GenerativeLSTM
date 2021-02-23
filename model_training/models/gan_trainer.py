# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:26:50 2020

@author: Manuel Camargo
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input

import sys
import time

class GAN(keras.Model):
    
    def __init__(self, discriminator, generator, ngram):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.ngram = ngram
        # self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # TODO: check
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.build_model()
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.ngram, ), name='ac_serie'))
        # add generator
        self.model.add(self.generator)
    	# add the discriminator
        self.model.add(self.discriminator)
    	# compile model
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))
        # self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.summary()

    def train(self, dataset, epochs, batch_size):
        dataset[:batch_size]
        for epoch in range(epochs):
            start = time.time()
            # for image_batch in dataset:
            self.train_step({'ac_input': dataset[:batch_size]})
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    def gan_loss(self, d_real_logits, d_fake_logits):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        categ_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        d_loss_real = cross_entropy(tf.ones_like(d_real_logits), d_real_logits)
        d_loss_fake = cross_entropy(tf.zeros_like(d_fake_logits), d_fake_logits)
        d_loss = d_loss_real + d_loss_fake
    
        g_loss = cross_entropy(tf.ones_like(d_fake_logits), d_fake_logits)
        return d_loss, g_loss
    

    @tf.function
    def train_step(self, events):
        if isinstance(events, tuple):
            events = events[0]
        batch_size = tf.shape(events['ac_input'])[0]

        noise = tf.random.uniform(
            shape=(batch_size, self.ngram),
            minval=0, maxval=events['ac_input'].shape[1],
            dtype=tf.int32)
        
        # with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        #     fake_events = self.model.get_layer('generator')(noise, training=True)
    
        #     d_real_logits = self.model.get_layer('discriminator')(events, training=True)
        #     d_fake_logits = self.model.get_layer('discriminator')(fake_events, training=True)
    
        #     d_loss, g_loss = self.gan_loss(d_real_logits, d_fake_logits)
        #     tf.print(d_loss, output_stream=sys.stderr)
        #     tf.print(g_loss, output_stream=sys.stderr)
        with tf.GradientTape() as g_tape:
            fake_events = self.model.get_layer('generator')(noise, training=True)
        
        with tf.GradientTape() as d_tape:
            d_real_logits = self.model.get_layer('discriminator')(events, training=True)
            d_fake_logits = self.model.get_layer('discriminator')(fake_events, training=True)
            tf.print(d_fake_logits, output_stream=sys.stderr)
            d_loss, g_loss = self.gan_loss(d_real_logits, d_fake_logits)
            # tf.print(d_loss, output_stream=sys.stderr)
            # tf.print(g_loss, output_stream=sys.stderr)

    
        g_gradients = g_tape.gradient(g_loss, self.model.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.model.trainable_variables)
        
        # tf.print(g_gradients, output_stream=sys.stderr)
        # tf.print(d_gradients, output_stream=sys.stderr)
        # self.generator_optimizer.apply_gradients(zip(g_gradients, self.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(d_gradients, self.model.trainable_variables))
        return {"d_loss": d_loss, "g_loss": 0.01} 
    
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # @tf.function
    # def train_step(self, events):
    #     if isinstance(events, tuple):
    #         events = events[0]
    #     batch_size = tf.shape(events['ac_input'])[0]

    #     noise = tf.random.uniform(
    #         shape=(batch_size, self.ngram),
    #         minval=0, maxval=events['ac_input'].shape[1],
    #         dtype=tf.int32)
        
    #     with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape() as d_tape:
    #         fake_events = self.generator(noise, training=True)
    
    #         d_real_logits = self.discriminator(events, training=True)
    #         d_fake_logits = self.discriminator(fake_events, training=True)
    
    #         d_loss, g_loss = self.gan_loss(d_real_logits, d_fake_logits)
    #         tf.print(d_loss, output_stream=sys.stderr)
    #         tf.print(g_loss, output_stream=sys.stderr)

    
    #     g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
    #     d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
    
    #     self.generator_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
    #     self.discriminator_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
    #     return {"d_loss": d_loss, "g_loss": 0.01} 
        
    # @tf.autograph.experimental.do_not_convert    
    # def train_step(self, real_events):
    #     if isinstance(real_events, tuple):
    #         real_events = real_events[0]
    #     # Sample random points in the latent space
    #     # tf.print(real_events['ac_input'].shape, output_stream=sys.stderr)
    #     batch_size = tf.shape(real_events['ac_input'])[0]
    #     # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_activities = tf.random.uniform(
    #         shape=(batch_size, self.ngram), 
    #         minval=0, maxval=real_events['ac_input'].shape[1],
    #         dtype=tf.int32)
    #     # random_roles = tf.random.uniform(
    #     #     shape=(batch_size, self.ngram), 
    #     #     minval=0, maxval=real_events['rl_input'].shape[1],
    #     #     dtype=tf.int32)

    #     # tf.print(random_activities.shape, output_stream=sys.stderr)


    #     # Decode them to fake events
    #     # generated_events = self.generator(
    #     #     {'ac_serie': random_activities, 'rl_serie': random_roles})

    #     generated_events = self.generator({'ac_serie': random_activities})

    #     # Combine them with real images
    #     # combined_activities = tf.concat([generated_events[0],
    #     #                                  real_events['ac_input']], axis=0)
    #     combined_activities = tf.concat([generated_events,
    #                                      real_events['ac_input']], axis=0)
    #     # combined_roles = tf.concat([generated_events[1],
    #     #                                  real_events['rl_input']], axis=0)

    #     # Assemble labels discriminating real from fake events
    #     labels = tf.concat(
    #         [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
    #     )
    #     # Add random noise to the labels - important trick!
    #     labels += 0.05 * tf.random.uniform(tf.shape(labels))

    #     # Train the discriminator
    #     # with tf.GradientTape() as tape:
    #     #     predictions = self.discriminator(
    #     #         {'ac_input': combined_activities, 'rl_input': combined_roles})
    #     #     d_loss = self.loss_fn(labels, predictions)
    #     # grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     # self.d_optimizer.apply_gradients(
    #     #     zip(grads, self.discriminator.trainable_weights)
    #     # )
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator({'ac_input': combined_activities})
    #         d_loss = self.loss_fn(labels, predictions)
    #     grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     self.d_optimizer.apply_gradients(
    #         zip(grads, self.discriminator.trainable_weights)
    #     )
    #     # Sample random points in the latent space
    #     # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_activities = tf.random.uniform(
    #         shape=(batch_size, self.ngram), 
    #         minval=0, maxval=real_events['ac_input'].shape[1],
    #         dtype=tf.int32)
    #     # random_roles = tf.random.uniform(
    #     #     shape=(batch_size, self.ngram), 
    #     #     minval=0, maxval=real_events['rl_input'].shape[1],
    #     #     dtype=tf.int32)

    #     # Assemble labels that say "all real images"
    #     misleading_labels = tf.zeros((batch_size, 1))

    #     # Train the generator (note that we should *not* update the weights
    #     # of the discriminator)!
    #     with tf.GradientTape(persistent=True) as tape:
    #         g_predictions = self.generator(
    #             {'ac_serie': random_activities})
    #         tape.watch(g_predictions)
    #         d_predictions = self.discriminator(
    #             {'ac_input': g_predictions})
    #         tape.watch(self.generator.trainable_weights)
    #         # d_predictions = self.discriminator(self.generator({'ac_serie': random_activities}))
    #         # tf.print(d_predictions, output_stream=sys.stderr)
    #         tape.watch(d_predictions)
    #         g_loss = self.loss_fn(misleading_labels, d_predictions)
    #         tape.watch(g_loss)
    #         # g_loss = [l(t, o) for l,o,t in zip([self.loss_fn], d_predictions, misleading_labels)]
    #         # losses = [l(t, o) for l,o,t in zip(loss_objects, outputs, targets)]
    #         # tf.print(tf.shape(self.generator.trainable_weights), output_stream=sys.stderr)
    #     grads = tape.gradient(g_loss, self.generator.trainable_weights)
    #     print(grads)
    #     # self.g_optimizer.apply_gradients(
    #     #     zip(grads, self.generator.trainable_weights)) 
    #     return {"d_loss": d_loss, "g_loss": 0.01} 

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
 
    # @tf.function
    # def train_step(self, events):
    #     if isinstance(events, tuple):
    #         events = events[0]
    #     batch_size = tf.shape(events['ac_input'])[0]

    #     # noise = tf.random.normal([BATCH_SIZE, noise_dim])

    #     noise = tf.random.uniform(
    #         shape=(batch_size, self.ngram),
    #         minval=0, maxval=events['ac_input'].shape[1],
    #         dtype=tf.int32)

    #     print(noise)
    #     with tf.GradientTape() as gen_tape:
    #         # generated_events = self.generator(noise, training=True)
    #         print('hello')
    #         generated_events = self.generator(noise, training=True)
    #         print('generated_events')
        
    #     with tf.GradientTape() as disc_tape:
    #         disc_tape.watch(generated_events)
    #         real_output = self.discriminator(events, training=True)
    #         fake_output = self.discriminator(generated_events, training=True)
    #         print('discriminate')
              
    #         gen_loss = self.generator_loss(fake_output)
    #         disc_loss = self.discriminator_loss(real_output, fake_output)
    #         print('loss')
        
    #     print('aca!!!!')
    #     gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    #     gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    
    #     print('aca2!!!!')
    #     self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    #     self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
    #     return {"d_loss": disc_loss, "g_loss": gen_loss} 

# =============================================================================
#     Implementation 1
# =============================================================================
    
    # @tf.function
    # def train_step(self, events):
    #     if isinstance(events, tuple):
    #         events = events[0]
    #     batch_size = tf.shape(events['ac_input'])[0]

    #     # noise = tf.random.normal([BATCH_SIZE, noise_dim])

    #     noise = tf.random.uniform(
    #         shape=(batch_size, self.ngram),
    #         minval=0, maxval=events['ac_input'].shape[1],
    #         dtype=tf.int32)

    #     with tf.GradientTape(persistent=True) as tape:
    #         # Compute the losses (note: I changed the method's signature; you
    #         # can use self.generator and self.discriminator in _gan_loss_fn)
    #         g_loss, d_loss = self._gan_loss_fn(noise, events['ac_input'])
    #     # Compute and apply the first set of gradients.
    #     g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
    #     self.generator_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
    #     # Compute and apply the second set of gradients.
    #     d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
    #     self.discriminator_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
    #     # Delete the tape to ensure proper garbage collection and variables release.
    #     del tape        
    #     return {"d_loss": d_loss, "g_loss": g_loss} 

    
    # def _get_loss(self, logits, label):
    #     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #     return cross_entropy(logits, label)
    
    # def _gan_loss_fn(self, input_noise, real_event):
    #     fake_event = self.generator(input_noise)
    #     # real_logits = self.discriminator(fake_event)
    #     # fake_logits = self.discriminator(real_event)
    #     real_logits = self.discriminator(real_event)
    #     fake_logits = self.discriminator(fake_event)

    #     g_loss = self._get_loss(fake_logits, tf.ones_like(fake_logits))
    #     d_loss = self._get_loss(real_logits, tf.ones_like(real_logits)) + \
    #               self._get_loss(fake_logits, tf.zeros_like(fake_logits))
    #     # d_loss = self._get_loss(fake_logits, tf.zeros_like(real_logits))
    #     print(g_loss)
    #     return g_loss, d_loss
