from deck_themer.helpers import *

aminatou_corpus = corpus_maker('C:\\Users\\Jamesson\\Documents\\GitHub\\deck_themer_v2\\CSV_files\\aminatou_tdm.csv')

lda = create_lda(k=9, corpus=aminatou_corpus)

hdp = create_hdp(initial_k=9, corpus=aminatou_corpus)

print('Done')
