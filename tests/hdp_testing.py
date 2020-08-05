from deck_themer.helpers import *

# Test for experimenting with HDP and with HDP hyperparameters.

# The 'corpus' created below is necessary for both the hdp_param_checker() method and the create_hdp() method. The
# 'decklists' created below is only used for the hdp_param_checker().
decklists, corpus, df = corpus_maker('..\\CSV_files\\obfuscated_tdm.csv')
# Deck files are somewhat proprietary, so you'll have to make your own.

# Use this for getting a general feel for what the different hyperparameters do.
hdp_test_results = hdp_param_checker(tw=tp.TermWeight.IDF, min_df_0=5, min_df_f=6, k0_0=2, k0_f=10, k0_s=7, alpha_0=-1,
                                     alpha_f=1, eta_0=-1, eta_f=1, gamma_0=-1, gamma_f=1, corpus=corpus,
                                     word_list=decklists, to_excel=True,
                                     fname='..\\tests\\TestResults\\obfuscated_tdm_param_checking_results_tw-ONE.xlsx')

## Use this for testing specific hyperparameters.
# hdp = create_hdp(min_df=5, initial_k=9, corpus=corpus)
