#encoding=utf8

import xlrd
import jieba
import time
import multiprocessing
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec

import random

import sys
reload(sys)

case_file_name = 'dataset/original_data/case.xlsx'
description_file_name = 'dataset/processed_data/description'
description_seg_file_name = 'dataset/processed_data/description_seg'
wordvec_description_file_name = 'dataset/processed_data/wordvec_description'
api_label_file_name = 'dataset/processed_data/api_label'

random_description_file = 'dataset/processed_data/random_description'
random_description_seg_file = 'dataset/processed_data/random_description_seg'
random_api_label_file = 'dataset/processed_data/random_api_label'


def singleSegEngine(segStr, segMode='e', userDictPath=None):
    '''
    分词
    :param segStr:
    :param segMode:
    :param userDictPath:
    :return:
    '''
    if not userDictPath == None:
        jieba.load_userdict(userDictPath)

    if segMode == 'a':
        wordGenList = jieba.cut(segStr, cut_all=True)
    elif segMode == 's':
        wordGenList = jieba.cut_for_search(segStr)
    else:
        wordGenList = jieba.cut(segStr, cut_all=False)

    wordStr = '_'.join(wordGenList)
    wordList = wordStr.split('_')

    return wordList



def descriptionTextSeg():
    '''
    将case里的描述分词并保存文件，将对应的api保存文件
    :return:
    '''
    case = xlrd.open_workbook(case_file_name).sheet_by_name('Sheet1')
    description_file = open(description_file_name,'w')
    seg_file = open(description_seg_file_name, 'w')
    label_file = open(api_label_file_name, 'w')

    api_set = {}
    api_num = 1
    api_list = []
    sentence_max_len = 0
    for i in range(1,case.nrows):
        api = case.cell(i,1).value
        description = case.cell(i, 2).value

        api = api.replace('\n',' ').strip()
        description = description.replace('\n',' ').strip()
        if(api == '' or description == '' or api == None or description == None):
            continue

        if(api not in api_set):
            api_set[api] = api_num
            api_num += 1
        api_list.append(str(api_set[api]) + '||' + api)
        label_file.write(str(api_set[api]) + '||' + api + '\n')

        description_file.write(description.encode('utf-8') + '\n')
        seg_words = singleSegEngine(description)
        if sentence_max_len < len(seg_words):
            sentence_max_len = len(seg_words)
        seg_patient_str = '||'.join(seg_words)
        seg_file.write(seg_patient_str.encode('utf-8') + '\n')

    description_file.close()
    seg_file.close()
    label_file.close()

    print 'max_length of description:{}'.format(sentence_max_len)
    return sentence_max_len, api_list, api_num-1


def descriptionTextWordvec():
    '''
    用分好词的数据训练词向量
    :return:
    '''
    description_sentences = list(word2vec.LineSentence(description_seg_file_name))
    print('sentences num: {0}'.format(len(description_sentences)))

    start = time.clock()
    model = Word2Vec(description_sentences, size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save(wordvec_description_file_name)
    end = time.clock()
    print('train gensim word2vec model finish, use time {}'.format(end - start))

    return description_sentences,model


def random_shuffle():
    '''
    随机打乱数据
    :return:
    '''
    with open(description_file_name,'r') as f:
        description = f.readlines()

    with open(description_seg_file_name,'r') as f:
        description_seg = f.readlines()

    with open(api_label_file_name,'r') as f:
        api_label = f.readlines()

    # 随机打乱
    data_set = zip(description, description_seg, api_label)
    random.shuffle(data_set)
    description[:], description_seg[:], api_label[:] = zip(*data_set)

    with open(random_description_file,'w') as f:
        f.writelines(description)

    with open(random_description_seg_file,'w') as f:
        f.writelines(description_seg)

    with open(random_api_label_file,'w') as f:
        f.writelines(api_label)

