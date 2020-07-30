# -*- coding: utf-8 -*-
from sample.helpers import *

aminatou_decklists, aminatou_corpus = corpus_maker('') # Deck files are somewhat proprietary, so this field has been left blank.

#lda = create_lda(k=9, corpus=aminatou_corpus)

hdp = create_hdp(min_cf=0,min_df=5,initial_k=9,corpus=aminatou_corpus)

hdp_results = hdp_param_checker(corpus=aminatou_corpus, word_list=aminatou_decklists, card_count=60)

print('Done')
