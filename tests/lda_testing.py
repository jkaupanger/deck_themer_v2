from deck_themer.helpers import *

decklists, corpus, df = corpus_maker('CSV_files\\obfuscated_tdm.csv')
# Deck files are somewhat proprietary, so you'll have to make your own.

lda = create_lda(k=11, corpus=corpus)

for i in range(0, 1001, 100):
    lda.burn_in = 100
    lda.train(0)
    lda.train(100)

topics = lda_topic_outputter(lda, card_count=60, to_excel=True, fname='k11_lda_topics.xlsx')

print('Done')
