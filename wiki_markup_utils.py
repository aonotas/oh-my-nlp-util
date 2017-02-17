#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bz2
import logging
import re
from xml.etree.cElementTree import iterparse  # LXML isn't faster, so let's go with the built-in solution
import multiprocessing

from gensim import utils

# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger('gensim.corpora.wikicorpus')

# ignore articles shorter than ARTICLE_MIN_WORDS characters (after full preprocessing)
ARTICLE_MIN_WORDS = 50


RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE)  # comments
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)  # footnotes
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE)  # links to languages
RE_P3 = re.compile("{{([^}{]*)}}", re.DOTALL | re.UNICODE)  # template
RE_P4 = re.compile("{{([^}]*)}}", re.DOTALL | re.UNICODE)  # template
RE_P5 = re.compile('\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)  # remove URL, keep description
RE_P6 = re.compile("\[([^][]*)\|([^][]*)\]", re.DOTALL | re.UNICODE)  # simplify links, keep description
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)  # keep description of images
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)  # keep description of files
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)  # outside links
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)  # math content
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE)  # all other tags
RE_P12 = re.compile('\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE)  # table formatting
RE_P13 = re.compile('\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE)  # table cell formatting
RE_P14 = re.compile('\[\[Category:[^][]*\]\]', re.UNICODE)  # categories
# Remove File and Image template
RE_P15 = re.compile('\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)

RE_MY_LINK = re.compile('\[\[([^|]+\|)?([^]]*)\]\]', re.UNICODE)

# MediaWiki namespaces (https://www.mediawiki.org/wiki/Manual:Namespace) that
# ought to be ignored
IGNORED_NAMESPACES = ['Wikipedia', 'Category', 'File', 'Portal', 'Template',
                      'MediaWiki', 'User', 'Help', 'Book', 'Draft',
                      'WikiProject', 'Special', 'Talk']


def filter_wiki(raw):
    """
    Filter out wiki mark-up from `raw`, leaving only text. `raw` is either unicode
    or utf-8 encoded string.
    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text)


def remove_markup(text):
    text = re.sub(RE_P2, "", text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    text = replace_link_to_phrase(text)
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, "", text)  # remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        # text = re.sub(RE_P3, '', text)  # remove template
        # text = re.sub(RE_P4, '', text)  # remove template
        text = re.sub(RE_P9, "", text)  # remove outside links
        text = re.sub(RE_P10, "", text)  # remove math content
        text = re.sub(RE_P11, "", text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description
        text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # text = re.sub(RE_P7, '\\3', text)  
        # text = re.sub(RE_P8, '\\3', text)  
        # remove table markup
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text)  # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        if old == text or iters > 2:  # stop if nothing changed between two iterations or after a fixed number of iterations
            break

    # the following is needed to make the tokenizer see '[[socialist]]s' as a single word 'socialists'
    # TODO is this really desirable?
    # text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text

    return text


def remove_template(s):
    """Remove template wikimedia markup.
    Return a copy of `s` with all the wikimedia markup template removed. See
    http://meta.wikimedia.org/wiki/Help:Template for wikimedia templates
    details.
    Note: Since template can be nested, it is difficult remove them using
    regular expresssions.
    """

    # Find the start and end position of each template by finding the opening
    # '{{' and closing '}}'
    n_open, n_close = 0, 0
    starts, ends = [], []
    in_template = False
    prev_c = None
    for i, c in enumerate(iter(s)):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                n_open, n_close = 0, 0
        prev_c = c

    # Remove all the templates
    s = ''.join([s[end + 1:start] for start, end in
                 zip(starts + [None], [-1] + ends)])

    return s


def remove_file(s):
    """Remove the 'File:' and 'Image:' markup, keeping the file caption.
    Return a copy of `s` with all the 'File:' and 'Image:' markup replaced by
    their corresponding captions. See http://www.mediawiki.org/wiki/Help:Images
    for the markup details.
    """
    # The regex RE_P15 match a File: or Image: markup
    for match in re.finditer(RE_P15, s):
        m = match.group(0)
        caption = m[:-2].split('|')[-1]
        s = s.replace(m, caption, 1)
    return s

def replace_link_to_phrase(s):
    for match in re.finditer(RE_MY_LINK, s):
        m = match.group(0)
        phrase = u'-'.join(match.groups()[-1].split(u' '))
        s = s.replace(m, phrase, 1)
    return s


if __name__ == '__main__':
    words = "'''Anarchism''' is a [[political philosophy]] that advocates [self-governed] societies based on voluntary institutions. These are often described as [stateless societies], although several authors have defined them more specifically as institutions based on non-[hierarchical] [free associations]. Anarchism considers the [state] to be undesirable, unnecessary, and harmful, because anarchists generally believe that human beings are capable of managing their own affairs on the basis of creativity, cooperation, and mutual respect, and when making individual decisions they are taking into account the concerns of others and the well-being of society."
    print replace_link_to_phrase(words)