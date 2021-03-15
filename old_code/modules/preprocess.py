from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
import warnings

from gensim.utils import keep_vocab_item, call_on_class_only, SaveLoad
from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.test.utils import datapath

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import range

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

import numpy as np
from numpy import exp, dot, zeros, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones, logaddexp, log, outer

from nltk import sent_tokenize
import string

import time
import random
import unidecode

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import (  # noqa: F401
        MAX_WORDS_IN_BATCH
    )
except ImportError:
    raise utils.NO_CYTHON

class vocabulary(utils.SaveLoad):
    """Build vocabulary from a sequence of documents (can be a once-only generator stream).
            Parameters
            ----------
            documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
                Can be simply a list of :class:`~gensim.models.doc2vec.TaggedDocument` elements, but for larger corpora,
                consider an iterable that streams the documents directly from disk/network.
                See :class:`~gensim.models.doc2vec.TaggedBrownCorpus` or :class:`~gensim.models.doc2vec.TaggedLineDocument`
            corpus_file : str, optional
                Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
                You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
                `corpus_file` arguments need to be passed (not both of them). Documents' tags are assigned automatically
                and are equal to a line number, as in :class:`~gensim.models.doc2vec.TaggedLineDocument`.
            update : bool
                If true, the new words in `documents` will be added to model's vocab.
            progress_per : int
                Indicates how many words to process before showing/updating the progress.
            keep_raw_vocab : bool
                If not true, delete the raw vocabulary after the scaling is done and free up RAM.
            trim_rule : function, optional
                Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
                be trimmed away, or handled using the default (discard if word count < min_count).
                Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
                or a callable that accepts parameters (word, count, min_count) and returns either
                :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
                The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
                of the model.
                The input parameters are of the following types:
                    * `word` (str) - the word we are examining
                    * `count` (int) - the word's frequency count in the corpus
                    * `min_count` (int) - the minimum count threshold.
    """
    def __init__(
            self, sentences=None, corpus_file=None, update=False, progress_per=500, keep_raw_vocab=False,
            trim_rule=None, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0,
            max_final_vocab=None, ns_exponent=0.75, embedding_dim = 30, letter_embedding = False):
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        # self.sample = sample
        # self.sorted_vocab = sorted_vocab
        # self.cum_table = None  # for negative sampling
        self.raw_vocab = None
        self.paths = None
        self.wv = Word2VecKeyedVectors(embedding_dim)   ##Word2VecKeyedVectors(size)
        self.max_final_vocab = max_final_vocab
        # self.ns_exponent = ns_exponent
        self.size = 0 # size of the vocabulary (number of words)
        self.letter_embedding = letter_embedding

        total_words, corpus_count = self.scan_vocab(
            sentences=sentences, corpus_file=corpus_file, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count    # number of sentences/paths in the corpus
        self.corpus_total_words = total_words
        # report_values = self.prepare_vocab( self.wv, update=update, keep_raw_vocab=keep_raw_vocab,
        #     trim_rule=trim_rule, **kwargs)
        # report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        # self.trainables.prepare_weights(self.hs, self.negative, self.wv, update=update, vocabulary=self.vocabulary)


    def scan_vocab(self, sentences=None, corpus_file=None, progress_per=1000, workers=None, trim_rule=None):
        logger.info("collecting all words and their counts")
        if corpus_file:
            sentences = LineSentence(corpus_file)

        total_words, corpus_count = self._scan_vocab(sentences, progress_per, trim_rule)
        self.size = len(self.wv.index2word)

        if self.letter_embedding:
            logger.info(
                "collected %i letter types from a corpus of %i raw letters and %i words",
                self.size, total_words, corpus_count
            )
        else:
            logger.info(
                "collected %i word types from a corpus of %i raw words and %i sentences",
                self.size, total_words, corpus_count
            )

        return total_words, corpus_count

    def _scan_vocab(self, sentences, progress_per, trim_rule):
        if self.letter_embedding:
            words = sentences
            word_no = -1
            total_letters = 0
            min_reduce = 1
            checked_string_types = 0
            lettersIta = 'abcdefghilmnopqrstuvz'
            index2letter = list(lettersIta)
            # index2letter = list(string.ascii_lowercase)
            vocab = {}
            # for index, letter in enumerate(string.ascii_lowercase):
            for index, letter in enumerate(lettersIta):
                vocab[letter] = Vocab(count=0, sentences_no=[], index=index)
            paths = []
            for word_no, word in enumerate(words):
                if not checked_string_types:
                    if isinstance(word, string_types):
                        logger.warning(
                            "Each 'words' item should be a list of letters (usually unicode strings). "
                            "First item here is instead plain %s.",
                            type(word)
                        )
                    checked_string_types += 1
                if word_no % progress_per == 0:
                    logger.info(
                        "PROGRESS: at word #%i, processed %i letters, keeping %i word types",
                        word_no, total_letters, 26
                    )
                path = []
                for letter in word:
                    vocab[letter].count += 1
                    vocab[letter].sentences_no.append(word_no)
                    path.append(vocab[letter].index)
                assert len(path) == len(word)
                paths.append(path)
                total_letters += len(word)

                if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                    ### SHOULD CHECK WHAT prune.vocab DOES IF I WANT TO USE THIS FUNCTIONALITY!!!
                    utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                    min_reduce += 1
            corpus_count = word_no + 1
            # self.raw_vocab = raw_vocab
            assert len(paths) == corpus_count
            self.wv.index2word = index2letter
            self.wv.vocab = vocab
            self.paths = paths
            return total_letters, corpus_count
        else:
            sentence_no = -1
            total_words = 0
            min_reduce = 1
            raw_vocab = defaultdict(list)
            checked_string_types = 0
            index2word = []
            vocab = {}
            paths = []

            for sentence_no, sentence in enumerate(sentences):
                if not checked_string_types:
                    if isinstance(sentence, string_types):
                        logger.warning(
                            "Each 'sentences' item should be a list of words (usually unicode strings). "
                            "First item here is instead plain %s.",
                            type(sentence)
                        )
                    checked_string_types += 1
                if sentence_no % progress_per == 0:
                    logger.info(
                        "PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                        sentence_no, total_words, len(raw_vocab)
                    )
                path = []
                for word in sentence:
                    raw_vocab[word].append(sentence_no)
                    if len(raw_vocab[word]) == 1:
                        vocab[word] = Vocab(count=1, sentences_no=raw_vocab[word], index =len(index2word))
                        index2word.append(word)
                    else:
                        keep_index = vocab[word].index
                        vocab[word] = Vocab(count=len(raw_vocab[word]), sentences_no=raw_vocab[word],
                                            index=keep_index)
                    path.append(vocab[word].index)
                assert len(path) == len(sentence)
                paths.append(path)
                total_words += len(sentence)

                if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                    ### SHOULD CHECK WHAT prune.vocab DOES IF I WANT TO USE THIS FUNCTIONALITY!!!
                    utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                    min_reduce += 1
            corpus_count = sentence_no + 1
            # self.raw_vocab = raw_vocab
            assert len(paths) == corpus_count
            self.wv.index2word = index2word
            self.wv.vocab = vocab
            self.paths = paths
            return total_words, corpus_count

    ### UNCOMMENT FOR prepare_vocab FUNCTION

    # def prepare_vocab(
    #         self, wv, update=False, keep_raw_vocab=False, trim_rule=None,
    #         min_count=None, sample=None, dry_run=False):
    #     """Apply vocabulary settings for `min_count` (discarding less-frequent words)
    #     and `sample` (controlling the downsampling of more-frequent words).
    #     Calling with `dry_run=True` will only simulate the provided settings and
    #     report the size of the retained vocabulary, effective corpus length, and
    #     estimated memory requirements. Results are both printed via logging and
    #     returned as a dict.
    #     Delete the raw vocabulary after the scaling is done to free up RAM,
    #     unless `keep_raw_vocab` is set.
    #     """
    #     if self.letter_embedding:  ####IMPLEMENT!!!!####IMPLEMENT!!!!
    #         return
    #     else:
    #         min_count = min_count or self.min_count
    #         sample = sample or self.sample
    #         drop_total = drop_unique = 0
    #
    #         # set effective_min_count to min_count in case max_final_vocab isn't set
    #         self.effective_min_count = min_count
    #
    #         # if max_final_vocab is specified instead of min_count
    #         # pick a min_count which satisfies max_final_vocab as well as possible
    #         if self.max_final_vocab is not None:
    #             sorted_vocab = sorted(self.raw_vocab.keys(), key=lambda word: len(self.raw_vocab[word]), reverse=True)
    #             calc_min_count = 1
    #
    #             if self.max_final_vocab < len(sorted_vocab):
    #                 calc_min_count = self.raw_vocab[sorted_vocab[self.max_final_vocab]] + 1
    #
    #             self.effective_min_count = max(calc_min_count, min_count)
    #             logger.info(
    #                 "max_final_vocab=%d and min_count=%d resulted in calc_min_count=%d, effective_min_count=%d",
    #                 self.max_final_vocab, min_count, calc_min_count, self.effective_min_count
    #             )
    #
    #         if not update:
    #             logger.info("Loading a fresh vocabulary")
    #             retain_total, retain_words = 0, []
    #             # Discard words less-frequent than min_count
    #             if not dry_run:
    #                 wv.index2word = []
    #                 # make stored settings match these applied settings
    #                 self.min_count = min_count
    #                 self.sample = sample
    #                 wv.vocab = {}
    #
    #             for word, v in iteritems(self.raw_vocab):
    #                 if keep_vocab_item(word, len(v), self.effective_min_count, trim_rule=trim_rule):
    #                     retain_words.append(word)
    #                     retain_total += len(v)
    #                     if not dry_run:
    #                         wv.vocab[word] = Vocab(count=len(v), sentences_no = v, index=len(wv.index2word))
    #                         wv.index2word.append(word)
    #                 else:
    #                     drop_unique += 1
    #                     drop_total += v
    #             original_unique_total = len(retain_words) + drop_unique
    #             retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
    #             logger.info(
    #                 "effective_min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
    #                 self.effective_min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique
    #             )
    #             original_total = retain_total + drop_total
    #             retain_pct = retain_total * 100 / max(original_total, 1)
    #             logger.info(
    #                 "effective_min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
    #                 self.effective_min_count, retain_total, retain_pct, original_total, drop_total
    #             )
    #         else:
    #             logger.info("Updating model with new vocabulary")
    #             new_total = pre_exist_total = 0
    #             new_words = pre_exist_words = []
    #             for word, v in iteritems(self.raw_vocab):
    #                 if keep_vocab_item(word, v, self.effective_min_count, trim_rule=trim_rule):
    #                     if word in wv.vocab:
    #                         pre_exist_words.append(word)
    #                         pre_exist_total += v
    #                         if not dry_run:
    #                             wv.vocab[word].count += v
    #                     else:
    #                         new_words.append(word)
    #                         new_total += v
    #                         if not dry_run:
    #                             wv.vocab[word] = Vocab(count=v, index=len(wv.index2word))
    #                             wv.index2word.append(word)
    #                 else:
    #                     drop_unique += 1
    #                     drop_total += v
    #             original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
    #             pre_exist_unique_pct = len(pre_exist_words) * 100 / max(original_unique_total, 1)
    #             new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
    #             logger.info(
    #                 "New added %i unique words (%i%% of original %i) "
    #                 "and increased the count of %i pre-existing words (%i%% of original %i)",
    #                 len(new_words), new_unique_pct, original_unique_total, len(pre_exist_words),
    #                 pre_exist_unique_pct, original_unique_total
    #             )
    #             retain_words = new_words + pre_exist_words
    #             retain_total = new_total + pre_exist_total
    #
    #         # Precalculate each vocabulary item's threshold for sampling
    #         if not sample:
    #             # no words downsampled
    #             threshold_count = retain_total
    #         elif sample < 1.0:
    #             # traditional meaning: set parameter as proportion of total
    #             threshold_count = sample * retain_total
    #         else:
    #             # new shorthand: sample >= 1 means downsample all words with higher count than sample
    #             threshold_count = int(sample * (3 + sqrt(5)) / 2)
    #
    #         downsample_total, downsample_unique = 0, 0
    #         for w in retain_words:
    #             v = self.raw_vocab[w]
    #             word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
    #             if word_probability < 1.0:
    #                 downsample_unique += 1
    #                 downsample_total += word_probability * v
    #             else:
    #                 word_probability = 1.0
    #                 downsample_total += v
    #             if not dry_run:
    #                 wv.vocab[w].sample_int = int(round(word_probability * 2 ** 32))
    #
    #         if not dry_run and not keep_raw_vocab:
    #             logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
    #             self.raw_vocab = defaultdict(int)
    #
    #         logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
    #         logger.info(
    #             "downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
    #             downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total
    #         )
    #
    #         # return from each step: words-affected, resulting-corpus-size, extra memory estimates
    #         report_values = {
    #             'drop_unique': drop_unique, 'retain_total': retain_total, 'downsample_unique': downsample_unique,
    #             'downsample_total': int(downsample_total), 'num_retained_words': len(retain_words)
    #         }
    #
    #         if self.sorted_vocab and not update:
    #             self.sort_vocab(wv)
    #
    #         return report_values

class newLineSentence(object):
    """Iterate over a file that contains sentences: one line = one sentence.
    Words must be already preprocessed and separated by whitespace.
    """
    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None, letter_embedding = False):
        """
        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        limit : int or None
            Clip the file to the first `limit` lines. Do no clipping if `limit is None` (the default).
        Examples
        --------
        .. sourcecode:: pycon
            >>> from gensim.test.utils import datapath
            >>> sentences = LineSentence(datapath('lee_background.cor'))
            >>> for sentence in sentences:
            ...     pass
        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.letter_embedding = letter_embedding

    def __iter__(self):
        """Iterate through the lines in the source."""
        if self.letter_embedding:
            try:
                self.source.seek(0)
                for line in itertools.islice(self.source, self.limit):
                    if line[:-1].isalpha():
                        word = list(unidecode.unidecode(line))[:-1]
                        i = 0
                        while i < len(word):
                            yield word[i: i + self.max_sentence_length]
                            i += self.max_sentence_length
            except:
                ### SHOULD IMPLEMENT EXCEPTION ERROR ###
                print('Couldn\'t split words into letters')
                return
        else:
            try:
                # Assume it is a file-like object and try treating it as such
                # Things that don't have seek will trigger an exception
                self.source.seek(0)
                table = str.maketrans('', '', string.punctuation)
                for line in itertools.islice(self.source, self.limit):
                    for sentence in sent_tokenize(line):
                        sentence = utils.to_unicode(sentence).split()
                        sentence = [word.translate(table).lower() for word in sentence]
                        i = 0
                        while i < len(sentence):
                            yield sentence[i: i + self.max_sentence_length]
                            i += self.max_sentence_length
            except AttributeError:
                # If it didn't work like a file, use it as a string filename
                table = str.maketrans('', '', string.punctuation)
                with utils.open(self.source, 'rb') as fin:
                    for line in itertools.islice(fin, self.limit):
                        for sentence in sent_tokenize(line):
                            sentence = utils.to_unicode(sentence).split()
                            sentence = [word.translate(table).lower() for word in sentence]
                            i = 0
                            while i < len(sentence):
                                yield sentence[i: i + self.max_sentence_length]
                                i += self.max_sentence_length