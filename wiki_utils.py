#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bz2
import functools
import gensim
import logging
import sys
import re
import mwparserfromhell
import multiprocessing
N_CPU = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)

from prettyprint import pp

def _wiki_to_raw(text):
    return gensim.corpora.wikicorpus.filter_wiki(text)

def _process_page(page):
    wiki_id, title, text = page
    if is_redirect_page(text):
        return None
    text = _wiki_to_raw(text)
    return wiki_id, title, text

def _process_page_keepmarkup(page):
    wiki_id, title, text = page
    if is_redirect_page(text):
        return None
    # text = _wiki_to_raw(text)
    return wiki_id, title, text

def _process_page2paragrahs(page):
    wiki_id, title, text = page
    sections = []
    section = {"section_name": u"", "text": u""}
    for node in mwparserfromhell.parse(text).nodes:
        if isinstance(node, mwparserfromhell.nodes.Heading):
            section["text"] = _wiki_to_raw(section["text"])
            sections.append(section)
            section = {"section_name": u"", "text": u""}
            section["section_name"] = unicode(node)
        else:
            section["text"] += unicode(node)

    if section["section_name"] != u"":
        sections.append(section)
    return wiki_id, title, text, sections

def extract_pages(dump_file, flag_keepmarkup=False):
    logger.info('Starting to read dump file...')
    reader = WikiDumpIter(dump_file)
    wiki_pages = []
    pool = multiprocessing.pool.Pool(N_CPU)
    imap_func = functools.partial(pool.imap_unordered, chunksize=100)

    if flag_keepmarkup:
        func_name = _process_page_keepmarkup
    else:
        func_name = _process_page
    
    for page in imap_func(func_name, reader):
        if page is None:
            continue
        wiki_pages.append(page)

        if len(wiki_pages) % 1000 == 0:
            logger.info("[read] " + str(len(wiki_pages)))
    return wiki_pages

def extract_paragraphs(dump_file):
    logger.info('Starting to read dump file...')
    reader = WikiDumpIter(dump_file)
    wiki_pages = []
    pool = multiprocessing.pool.Pool(N_CPU)
    imap_func = functools.partial(pool.imap_unordered, chunksize=100)

    for page in imap_func(_process_page2paragrahs, reader):
        wiki_id, title, text, paragraphs = page

        wiki_pages.append(page)

        if len(wiki_pages) % 1000 == 0:
            logger.info(len(wiki_pages))
    return wiki_pages




''' class of Wiki Dump Iterator'''
class WikiDumpIter(object):
    def __init__(self, dump_file):
        self.dump_file = dump_file
        self.IGNORE_PAGE = ['wikipedia:', 'category:', 'file:', 'portal:', 'template:', 'mediawiki:',
        'user:', 'help:', 'book:', 'draft:']

    def __iter__(self):
        with bz2.BZ2File(self.dump_file) as f:
            for (title, wiki_text, wiki_page_id) in gensim.corpora.wikicorpus.extract_pages(f):
                if any([title.lower().startswith(namespace) for namespace in self.IGNORE_PAGE]):
                    continue
                yield [wiki_page_id, unicode(title), unicode(wiki_text)]


def is_redirect_page(text):
    REDIRECT_REGEXP = re.compile(
        ur"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))", re.IGNORECASE
    )
    redirect_match = REDIRECT_REGEXP.match(text)
    if not redirect_match:
        return False
    return True


def output_raw_text(wiki_pages, output_file):
    f = open(output_file, "w")
    for i, page in enumerate(wiki_pages):
        wiki_id, title, text = page
        one_line = title.encode("utf-8") + "\n"
        one_line += text.encode("utf-8")
        f.write(one_line + "\n")
        if i % 1000 == 0:
            logging.info("[write] " + str(i))
    f.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_file', dest='dump_file', type=str, default='jawiki-20160920-pages-articles.xml.bz2', help='Wikipedia dump_file. See : https://dumps.wikimedia.org/jawiki/')
    # Example: 
    # Download https://dumps.wikimedia.org/jawiki/20160920/jawiki-20160920-pages-articles.xml.bz2
    parser.add_argument('--mode', dest='mode', type=str, default="raw_text", help='mode = [raw_text]')
    parser.add_argument('--output_file', dest='output_file', type=str, default="", help='output_file')
    args = parser.parse_args()
    print args

    dump_file = args.dump_file
    # dump_file = sys.argv[1]
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    logging.info(dump_file)


    if args.mode == "raw_text":
        output_file = dump_file.replace(".xml.bz2", "_raw.txt")
        wiki_pages = extract_pages(dump_file=dump_file)
        output_raw_text(wiki_pages, output_file)


    elif args.mode == "raw_text_keep_markup":
        output_file = dump_file.replace(".xml.bz2", "_raw_keepmarkup.txt")
        wiki_pages = extract_pages(dump_file=dump_file)
        output_raw_text(wiki_pages, output_file)

    elif args.mode == "paragraphs":
        extract_paragraphs(dump_file=dump_file)