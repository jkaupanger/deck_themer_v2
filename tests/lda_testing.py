from deck_themer.helpers import *

# Test for experimenting with HDP and with HDP hyperparameters.

# The 'corpus' created below is necessary for both the lda_param_checker() method and the create_lda() method. The
# 'decklists' created below is only used for the lda_param_checker().
decklists, corpus, df = corpus_maker('..\\CSV_files\\obfuscated_tdm.csv')
# Deck files are somewhat proprietary, so you'll have to make your own.

# Use this for getting a general feel for what the different hyperparameters do.
lda_test_results = lda_param_checker(tw=tp.TermWeight.IDF, min_df_0=5, min_df_f=6, k_0=8, k_f=11, k_s=1, alpha_0=-1,
                                     alpha_f=1, eta_0=0, eta_f=2, corpus=corpus, word_list=decklists, to_excel=True,
                                     fname='..\\tests\\TestResults\\obfuscated_tdm_lda_param_checking_results_tw-IDF.xlsx')

## Use this for testing specific hyperparameters.
# lda = create_lda(min_df=5, k=9, corpus=corpus)
