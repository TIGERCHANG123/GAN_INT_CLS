import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim, gp):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.gp = gp

        self.fake_loss = 0
        self.real_loss = 0
        self.grad_penalty = 0

    def discriminator_loss(self, real_output, fake_output1, fake_output2):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss1 = cross_entropy(tf.zeros_like(fake_output1), fake_output1)
        fake_loss2 = cross_entropy(tf.zeros_like(fake_output2), fake_output2)
        total_loss = real_loss + fake_loss1 + fake_loss2
        return total_loss, real_loss, fake_loss1, fake_loss2

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_g_step(self, noise, text):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(text, noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            self.fake_loss = self.generator_loss(fake_output)
            gen_loss = -self.fake_loss
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_loss(gen_loss)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    def train_d_step(self, noise, images, text, text_generator):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(text, noise, training=True)
            real_output = self.discriminator(text, images, training=True)
            fake_output1 = self.discriminator(text, generated_images, training=True)
            fake_text = text_generator()
            fake_output2 = self.discriminator(fake_text, images, training=True)

            disc_loss, real_loss, fake_loss1, fake_loss2 = self.discriminator_loss(real_output, fake_output1,fake_output2)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_loss(disc_loss)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train_step(self, noise, images, text, text_generator):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(text, noise, training=True)
            real_output = self.discriminator(text, images, training=True)
            fake_output1 = self.discriminator(text, generated_images, training=True)
            fake_text = text_generator(images.shape[0])
            fake_output2 = self.discriminator(fake_text, images, training=True)

            disc_loss, real_loss, fake_loss1, fake_loss2 = self.discriminator_loss(real_output, fake_output1, fake_output2)
            gen_loss = self.generator_loss(fake_output1)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_loss(gen_loss)
        self.disc_loss(disc_loss)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epoch,  pic, text_generator):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (image, text)) in enumerate(self.train_dataset):
            noise = tf.random.normal([image.shape[0], self.noise_dim], dtype=tf.float32)
            self.train_step(noise, image, text, text_generator)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}, grad penalty: {}, real loss: {}, fake loss: {}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result(), self.grad_penalty, self.real_loss, self.fake_loss))