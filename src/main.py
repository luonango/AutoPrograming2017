# encoding=utf8

from gensim.models.word2vec import Word2Vec

import numpy as np
import model_D_G_Conbine
from utils.preprocess import singleSegEngine
from model_D_G_Conbine import build_discriminator, build_combined, build_generator
import sys

from keras.optimizers import RMSprop, Adam, SGD
'''
TD: 将code和line和API和Demand对应，并且将每个demand的code限制为20行，一行为一个词向量。
一个demand就有20行内的code词向量。
'''

description_file = '/dataset/processed_data/random_description'
description_seg_file = '/dataset/processed_data/random_description_seg'
code_file = '/dataset/processed_data/random_code'
code_seg_file = '/dataset/processed_data/random_code_seg'
wordvec_description_file = '/dataset/processed_data/wordvec_description'
api_label_file = '/dataset/processed_data/random_api_label'
wordvec_code_file = '/dataset/processed_data/wordvec_code'

file_model = 'model/model.json'
file_model_weight = 'model/model.h5'
D_model_file = 'model/model_D.json'
G_model_file = 'model/model_G.json'
D_model_weight_file = 'model/model_D.h5'
G_model_weight_file = 'model/model_G.h5'


sentence_max_len=200 # max vector
code_max_len = 20  # code lines


def get_data():
    
    with open(description_seg_file, 'r') as f:
        description_seg = f.readlines()
    
    with open(api_label_file, 'r') as f:
        api_list = f.readlines()
    
    with open(code_seg_file, 'r') as f:
        code_seg = f.readlines()
    
    # sentence_max_len = 0
    for i, line in enumerate(description_seg):
        line = line.strip()
        line_list = line.split('||')
        # sentence_max_len = len(line_list) if len(line_list) > sentence_max_len else sentence_max_len
        description_seg[i] = line[:sentence_max_len]
    
    api_max_num = 0
    for i, api in enumerate(api_list):
        num = int(api.split(u'||')[0])
        api_max_num = num if num > api_max_num else api_max_num
    
    
    for i, line in enumerate(code_seg):
        line = line.strip()
        line_list = line.split('||')
        # code_max_len = len(line_list) if len(line_list) > code_max_len else code_max_len
        code_seg[i] = line[:code_max_len]
    
    model = Word2Vec.load(wordvec_description_file)
    code_model = Word2Vec.load(wordvec_code_file)
    vocab = model.wv.vocab.keys()
    code_vocab = model.wv.vocab.keys()
    
    x = np.zeros((len(description_seg), sentence_max_len, model.vector_size), dtype=np.float32)
    y = np.zeros((len(description_seg), int(api_max_num)), dtype=np.bool)
    codes = np.zeros((len(code_seg), code_max_len, code_model.vector_size), dtype=np.float32)
    
    for i, sentence in enumerate(description_seg):
        for j, word in enumerate(sentence.split(u' ')):
            if word in vocab:
                vocab_vector = model[word]
            else:
                vocab_vector = np.zeros((model.vector_size), dtype=np.float32)
            x[i, j] = vocab_vector
        api = api_list[i]
        num = int(api.split(u'||')[0])
        y[i, num - 1] = 1
    
    for i, code in enumerate(code_seg):
        for j, word in enumerate(code.split(u'||')):
            if word in vocab:
                vocab_vector = model[word]
            else:
                vocab_vector = np.zeros((model.vector_size), dtype=np.float32)
            codes[i, j] = vocab_vector
    
    input_shape = (sentence_max_len, model.vector_size)
    output_length = int(api_max_num)
    # return x, y, input_shape, output_length, description, api_list
    demand_train = x
    code_train = codes
    api_train=y
    return (demand_train, code_train, y, input_shape,code_model,code_vocab)


def Train():
    (demand_train, code_train, api_train, input_shape, code_model,code_vocab) = get_data()
    
    generator = build_generator(input_shape, output_length=20 * 100)
    discriminator = build_discriminator(input_shape=(20, 100))
    combined = build_combined(input_shape, generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    #
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    combined.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    #
    from keras.utils.vis_utils import plot_model
    plot_model(combined, to_file='./combined_model.png', show_shapes=True)
    #
    # demand_train.shape=[None,1016,100]
    # code_train.shape=[None,20,100]
    #
    BATCH_SIZE = 8
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(code_train.shape[0] / BATCH_SIZE))
        for index in range(int(code_train.shape[0] / BATCH_SIZE)):
            # noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            demand_batch = demand_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            code_batch = code_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_code_batch = generator.predict(demand_batch, verbose=0)
            #
            X = np.concatenate((code_batch, generated_code_batch))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            discriminator.trainable = False
            g_loss = combined.train_on_batch(demand_batch, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def get_similar_in_word2vec(src_code, code_model, code_vocab):
    '''
    cosine
    '''
    total_num = len(code_vocab)
    similar_code = ''
    max_num = 0.
    for key in code_vocab:
        vec = code_model[key]
        # cosine
        d1 = np.dot(src_code, vec) / (np.linalg.norm(src_code) * np.linalg.norm(vec) + 1e-07)
        similar_code = vec if d1 > max_num else similar_code
    return similar_code


def generate_code(input_shape, demands, code_model, code_vocab):
    g = build_generator(input_shape, output_length=20 * 100)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    # generated_code.shape=[len(demands),20,100]
    generated_code_vecs = g.predict(demands, verbose=1)
    similar_codes = []
    for each in generated_code_vecs:
        similar_code = get_similar_in_word2vec(each, code_model, code_vocab)
        similar_codes.append(similar_code)
    return similar_codes


if __name__ == '__main__':
    Train()
