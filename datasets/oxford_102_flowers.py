import tensorflow as tf
import numpy as np
import os
import cv2
import re

class oxford_102_flowers_dataset():
    def __init__(self, root, batch_size):
        self.file_path = root + '/datasets/oxford_102_flowers/jpg'
        self.image_width = 64
        self.batch_size = batch_size
        self.file_list = os.listdir(self.file_path)
        self.total_pic_num = len(self.file_list)
        print('total images: {}'.format(len(self.file_list)))
        self.name = 'oxford-102-flowers'

        self.text_file_name = []
        for parent, dirnames, filenames in os.walk(root + '/datasets/oxford_102_flowers/text_c10'):
            for filename in filenames:
                if ".txt" in filename:
                    self.text_file_name.append(parent+'/'+filename)
        lines = []
        self.index_sentences = [None]*len(self.text_file_name)
        for file_path in self.text_file_name:
            with open(file_path, 'r', encoding='utf-8') as f:
                temp_lines = f.read().split('\n')
                clear_lines = []
                for sentence in temp_lines:
                    line = re.sub(r'[^A-Za-z]+', ' ',sentence)
                    if line != '':
                        clear_lines.append(line)
                index = int(file_path.split('_')[-1].split('.')[0])
                self.index_sentences[index-1] = clear_lines
                lines= lines+clear_lines
        characters = set()

        for sentence in lines:
            for char in sentence.split(' '):
                if char not in characters:
                    characters.add(char)
        self.num_tokens = len(characters)
        self.max_seq_length = max([len(txt.split(' ')) for txt in lines])
        characters = sorted(list(characters))
        self.token_index = dict(
                    [(char, i) for i, char in enumerate(characters)])
        self.index_token = characters
    def generator(self):
        for name in self.file_list:
          img = cv2.imread('{}/{}'.format(self.file_path, name), 1)
          if not img is None:
            index=int(name.split('_')[1].split('.')[0])
            img = cv2.resize(img, (self.image_width, self.image_width), interpolation=cv2.INTER_AREA)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            print('index: {}, len sentences: {}'.format(index, len(self.index_sentences)))
            n = np.random.randint(len(self.index_sentences[index]))
            text = self.index_sentences[index][n]
            text_code = np.zeros((self.max_seq_length,), dtype='float32')
#             print(text)
            for i, token in enumerate(text.split(' ')):
                text_code[i]=self.token_index[token]
#                 print(text_code[i])
            yield img, text_code
    def parse(self, x, y):
        x = tf.cast(x, tf.float32)
        x = x/255 * 2 - 1
        return x, y
    def get_train_dataset(self):
        train = tf.data.Dataset.from_generator(self.generator, output_types=(tf.int64, tf.float32))
        train = train.map(self.parse).shuffle(1000).batch(self.batch_size)
        return train
    def get_random_text(self):
        text_list = []
        for i in range(self.batch_size):
            index = np.random.randint(len(self.file_list))
            n = np.random.randint(len(self.index_sentences[index]))
            text = self.index_sentences[index][n]
            text_code = np.zeros((self.max_seq_length,), dtype='float32')
            for i, token in enumerate(text.split(' ')):
                text_code[i]=self.token_index[token]
            text_list.append(text_code)
        text_list = np.asarray(text_list)
        return text_list
    def text_decoder(self, code):
        s = []
        for c in code:
            s.append(self.index_token[c])
        s = ' '.join(s)
        return s
class noise_generator():
    def __init__(self, noise_dim, digit_dim, batch_size, iter_num):
        self.noise_dim = noise_dim
        self.digit_dim = digit_dim
        self.batch_size = batch_size
        self.iter_num = iter_num
    def __call__(self):
        for i in range(self.iter_num):
            noise = tf.random.normal([self.batch_size, self.noise_dim])
            noise = tf.cast(noise, tf.float32)
            yield noise
    def get_noise(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.digit_dim * [float(1.0 / self.digit_dim)],size=[self.batch_size])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        return noise, auxi_dict

    def get_fixed_noise(self, num):
        noise = tf.random.normal([1, self.noise_dim])
        noise = tf.cast(noise, tf.float32)

        auxi_dict = np.array([num])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.one_hot(auxi_dict, depth=self.digit_dim)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        return noise, auxi_dict