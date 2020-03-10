from GAN_cls_Block import *

class generator(tf.keras.Model):
  def __init__(self, tokens):
    super(generator, self).__init__()
    self.encoder = Encoder(num_encoder_tokens=tokens, latent_dim=128, embedding_dim=256)
    self.input_layer = generator_Input(shape=[4, 4, 1024])

    self.middle_layer_list = [
      generator_Middle(filters=512, strides=2, padding='same'),#1024*4*4
      generator_Middle(filters=256, strides=2, padding='same'),#512*8*8
      generator_Middle(filters=128, strides=2, padding='same'),#256*16*16
    ]

    self.output_layer = generator_Output(image_depth=3, strides=2, padding='same')#3*32*32
  def call(self, text, noise):
    x = self.encoder(text)
    x = tf.concat([noise, x], axis=-1)
    x = self.input_layer(x)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
    x = self.output_layer(x)
    return x

class discriminator(tf.keras.Model):
  def __init__(self):
    super(discriminator, self).__init__()
    self.input_layer = discriminator_Input(filters=128, strides=1)
    self.middle_layer_list = [
      discriminator_Middle(filters=256, strides=2, padding='valid'),
      discriminator_Middle(filters=512, strides=2, padding='valid'),
      discriminator_Middle(filters=1024, strides=2, padding='valid'),
    ]
    self.output_layer = discriminator_Output(with_activation=False)

  def call(self, text, x):
    ones = tf.ones(shape=[1, x.shape[1], x.shape[2]], dtype=x.dtype)
    text = ones*text
    x = tf.concat([x, text], axis=-1)
    x = self.input_layer(x)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
    x = self.output_layer(x)
    return x

def get_gan(noise_dim):
  Generator = generator(noise_dim)
  Discriminator = discriminator()
  gen_name = 'WGAN'
  return Generator, Discriminator, gen_name


