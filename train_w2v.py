#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import multiprocessing
import logging
"""
 python train_w2v --input=wiki.txt --window=5 --size=50 --negative=5 --output=w2v.model
"""
#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
workers = multiprocessing.cpu_count()

def load_text(filename):
    data = gensim.models.word2vec.Text8Corpus(filename)
    return data

def train_w2v(args):
    data = load_text(args.input)
    sg = 1 if args.model == "skipgram" else 0
    model = gensim.models.word2vec.Word2Vec(data, sg=sg, size=args.size, window=args.window, min_count=args.min_count, workers=workers, negative=args.negative, hs=0, sample=1e-4, iter=args.iteration)
    return model

def load_w2v_model(model_filename):
    model = gensim.models.Word2Vec.load(model_filename)
    return model

def main(args):
    model = train_w2v(args)
    model.save(args.output)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, dest='model', default='skipgram', help='model')
    parser.add_argument('--negative', type=int, dest='negative', default=5, help='negative')
    parser.add_argument('--min_count', type=int, dest='min_count', default=10, help='min_count')
    parser.add_argument('--iteration', type=int, dest='iteration', default=1, help='iteration')
    parser.add_argument('--window', type=int, dest='window', default=5, help='window')
    parser.add_argument('--size', dest='size', default=200, type=int, help='size')
    parser.add_argument('--input', dest='input', type=str, default='', help='input')
    parser.add_argument('--output', dest='output', type=str, default='w2v.model', help='output')

    args = parser.parse_args()

    main(args)
