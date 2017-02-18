#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
from gensim.models import Doc2Vec
import multiprocessing
import logging
from collections import namedtuple
Document = namedtuple('Document', 'words tags split')

"""
 python train_pv.py --input=wiki.txt --window=5 --size=50 --negative=5 --output=w2v.model
"""
#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
workers = multiprocessing.cpu_count()

def load_text(filename):
    docs_data = [l.strip().split(' ') for l in open(filename)]
    alldocs = []
    for line_no, line in enumerate(docs_data):
        # tokens = gensim.utils.to_unicode(line)
        # tokens = line
        words = line
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = 'train'
        alldocs.append(Document(words, tags, split))

    return alldocs

def train_w2v(args):
    alldocs = load_text(args.input)
    dm = 0 if args.model == "dbowword" else 1


    model = Doc2Vec(dm=dm, size=args.size, negative=args.negative, hs=0, window=args.window, min_count=args.min_count, workers=workers, dbow_words=True, sample=1e-4)
    
    def trim_rule(word, count, min_count):
        RULE_DEFAULT = 0
        # RULE_DISCARD = 1
        RULE_KEEP = 2
        if '-' in word and len(word) >= 2:
            # phraseの場合
            return RULE_KEEP
        return RULE_DEFAULT

    model.build_vocab(alldocs, trim_rule=trim_rule)
    for i in xrange(args.iteration):
        model.train(alldocs)

    return model

# def load_w2v_model(model_filename):
#     model = gensim.models.Word2Vec.load(model_filename)
#     return model

def main(args):
    model = train_w2v(args)
    model.save(args.output)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, dest='model', default='dbowword', help='model')
    parser.add_argument('--negative', type=int, dest='negative', default=5, help='negative')
    parser.add_argument('--min_count', type=int, dest='min_count', default=10, help='min_count')
    parser.add_argument('--window', type=int, dest='window', default=5, help='window')
    parser.add_argument('--iteration', type=int, dest='iteration', default=5, help='iteration')
    parser.add_argument('--size', dest='size', default=200, type=int, help='size')
    parser.add_argument('--input', dest='input', type=str, default='', help='input')
    parser.add_argument('--output', dest='output', type=str, default='w2v.model', help='output')

    args = parser.parse_args()

    main(args)

