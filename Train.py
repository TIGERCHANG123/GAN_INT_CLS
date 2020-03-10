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
    def get_loss(self, output):
        loss = tf.reduce_mean(output)
        return loss
    def train_g_step(self, noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            self.fake_loss = self.get_loss(fake_output)
            gen_loss = -self.fake_loss
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_loss(gen_loss)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    def train_d_step(self, noise, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            self.real_loss = self.get_loss(real_output)
            self.fake_loss = self.get_loss(fake_output)

            rate = np.random.rand()
            mixed_pic = rate * images + (1 - rate) * generated_images
            with tf.GradientTape() as mixed_tape:
                mixed_tape.watch(mixed_pic)
                mixed_output = self.discriminator(mixed_pic)
            grad_mixed = mixed_tape.gradient(mixed_output, mixed_pic)
            norm_grad_mixed = tf.sqrt(tf.reduce_sum(tf.square(grad_mixed), axis=[1, 2, 3]))
            self.grad_penalty = tf.reduce_mean(tf.square(norm_grad_mixed - 1))
            disc_loss = self.fake_loss - self.real_loss + self.gp * self.grad_penalty
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_loss(disc_loss)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train_step(self, noise, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            self.real_loss = self.get_loss(real_output)
            self.fake_loss = self.get_loss(fake_output)

            rate = np.random.rand()
            mixed_pic = rate * images + (1 - rate) * generated_images
            with tf.GradientTape() as mixed_tape:
                mixed_tape.watch(mixed_pic)
                mixed_output = self.discriminator(mixed_pic)
            grad_mixed = mixed_tape.gradient(mixed_output, mixed_pic)
            norm_grad_mixed = tf.sqrt(tf.reduce_sum(tf.square(grad_mixed), axis=[1, 2, 3]))
            self.grad_penalty = tf.reduce_mean(tf.square(norm_grad_mixed - 1))
            # print('gp: {}'.format(self.grad_penalty))
            disc_loss = self.fake_loss - self.real_loss + self.gp * self.grad_penalty
            gen_loss = -self.fake_loss
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_loss(gen_loss)
        self.disc_loss(disc_loss)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        i = 0
        for (batch, images) in enumerate(self.train_dataset):
            noise = tf.random.normal([images.shape[0], self.noise_dim])
            self.train_step(noise, images)
            # i = i + 1
            # if (i+1) % 5 != 0:
            #     noise = tf.random.normal([images.shape[0], self.noise_dim])
            #     self.train_d_step(noise, images)
            # else:
            #     noise = tf.random.normal([images.shape[0], self.noise_dim])
            #     self.train_g_step(noise)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}, grad penalty: {}, real loss: {}, fake loss: {}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result(), self.grad_penalty, self.real_loss, self.fake_loss))