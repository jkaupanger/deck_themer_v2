from deck_themer.helpers import *

decklists, corpus, df = corpus_maker('..\\CSV_files\\obfuscated_tdm.csv')
decklist = decklists[0000]

lda = tp.LDAModel.load('..\\tests\\obfuscated_lda_model')
#lda = create_lda(k=9, corpus=corpus)
#lda.burn_in = 100
#lda.train(0)
#for i in range(0,1001,100):
#    lda.train(100)
#lda.save(filename='..\\tests\\obfuscated_lda_model')

measured_decks = decks_measurer(lda, decklists=df,
                                to_excel=True, fname="..\\tests\\TestResults\\obfuscated_measured_decks.xlsx")

measured_deck0000 = deck_measurer(decklist, lda)

outlier = id_outlier(lda, decklist, 'all')
missing_common = id_missing_common(lda, decklist, 'all')

deck_removed_improvement = remove_improvement(lda, decklist, outlier[0])
deck_added_improvement = add_improvement(lda, decklist, missing_common[0])

print("Done")
