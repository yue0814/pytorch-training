import sys
import time
import numpy as np
import codecs

args = sys.argv

start = time.time()
with codecs.open(args[1], "r", encoding="utf-8") as train:
    words = [word for line in train for word in line.strip().split()]

    #vocabCount = {}
    #for i in words:
    #    if i not in vocabCount.keys():
    #        vocabCount[i] = 1
    #    else:
    #        vocabCount[i] += 1
    #rareword = []
    #for word, count in vocabCount.items():
    #    if count <= 10:
    #        rareword.append(word)
    #for i,word in enumerate(words):
    #    if word in rareword:
    #        words[i] = word.replace(word, '*UNK*')
    vocab = set(words)
    print('Number of words: %d' % len(words))
    print('Size of vocabulary: %d' % len(vocab))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    # id_to_word = {i: w for i, w in enumerate(vocab)}
    data = [word_to_id[w] for w in words]


with codecs.open(args[2], "r", encoding="utf-8") as development:
    dev_words = [word for line in development for word in line.strip().split()]
    for i, v in enumerate(dev_words):
        if v not in vocab:
            dev_words[i] = v.replace(v, '*UNK*')
    print('Number of words: %d' % len(dev_words))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    dev_data = [word_to_id[w] for w in dev_words]


def get_mini_batch(batch_size, step):
    ws, ts = [], []
    i = step * batch_size
    for _ in range(batch_size):
        while i < (step+1)*batch_size:
            w, t = (data[i:i+2], data[i+2])
            ws.append(w)
            ts.append(t)
            i += 1
    return np.array(ws), np.array(ts)


def dev_get_mini_batch(batch_size, dev_step):
    ws, ts = [], []
    i = dev_step * batch_size
    for _ in range(batch_size):
        while i < (dev_step+1)*batch_size:
            w, t = (dev_data[i:i+2], dev_data[i+2])
            ws.append(w)
            ts.append(t)
            i += 1
    return np.array(ws), np.array(ts)