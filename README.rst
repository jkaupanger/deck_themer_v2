Deck_themer
    Deck_themer is a python... package, I guess, that attempts to automatically extract deck archetypes from EDH/Commander decklists.

Contributing
    Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

    Please make sure to update tests as appropriate.

Updates
    - 2020-07-29: Initial code commit. None of this code is tested as of yet; it's my first attempt at converting my Jupyter notebook into "proper" Python code.

    - 2020-07-31: Making methods' docstrings better.

Testing
    There are three general groups of testing:

    - LDA testing - Testing to get a feel for latent Dirichlet allocation, which is for when you "know" how many topics that there are
    - HDP testing - Testing to get a feel for hierarchical Dirichlet processing, which is for when you are unsure of how many topics that there are
    - Site utility testing - Testing to demonstrate the various methods that might more directly benefit a user of the site

    Unfortunately, the data, itself, it proprietary, so I cannot supply an example CSV file with real data that this package will work with. The best I can do is to take real data and obfuscate it, hence 'CSV_files/obfuscated_tdm.csv'.

    LDA Testing
        If you run the lda_param_checking() method with the following parameters, you should get somewhere close to the following results:

        Hyperparameter Values:
            lda_param_checker(tw=tp.TermWeight.IDF, min_df_0=5, min_df_f=6, k_0=8, k_f=11, k_s=1, alpha_0=-1, alpha_f=1, eta_0=0, eta_f=2, corpus=corpus, word_list=<decklist_file>, to_excel=True, fname=<you_choose_filename.xlsx>)

        Average Results:
            - Average Average Log Likelihood: -12.46 +/- 0.1452
            - Average Perplexity: 247358 +/- 32378
            - Average Coherence: 0.8104 +/- 0.0576


    HDP Testing
        If you run the hdp_param_checking() method with the following parameters, you should get somewhere close to the following results:

        Hyperparameter Values:
            hdp_param_checker(tw=tp.TermWeight.ONE, min_df_0=5, min_df_f=6, k0_0=2, k0_f=10, k0_s=7, alpha_0=-1, alpha_f=1, eta_0=-1, eta_f=1, gamma_0=-1, gamma_f=1, corpus=corpus, word_list=<decklist_file>, to_excel=True, fname=<you_choose_filename>)

        Average Results:
            - Average Live_k: ~14 +/- ~7
            - Average Average Log Likelihood: -12.03 +/- 0.14
            - Average Perplexity: 167618 +/- 23600
            - Average Coherence: 0.884 +/- 0.0611

    Site Utility Testing
        Using the included tomotopy.LDAModel(), you should get the following results:

        Measured deck (measured_deck0000):
            - Topic0 - 0.8090473
            - Topic1 - 0.00049178087
            - Topic2 - 5.3064232e-05
            - Topic3 - 0.00014593646
            - Topic4 - 0.00018057834
            - Topic5 - 0.00043903533
            - Topic6 - 3.1631585e-05
            - Topic7 - 4.6155532e-05
            - Topic8 - 0.14680731
        Cards removed (outlier):
            - Topic0 - Card2203
            - Topic1 - Card1832
            - Topic2 - Card0294
            - Topic3 - Card3359
            - Topic4 - Card2478
            - Topic5 - Card1071
            - Topic6 - Card3359
            - Topic7 - Card0305
            - Topic8 - Card0831
        Cards missing (missing_common):
            - Topic0 - Card3179
            - Topic1 - Card0979
            - Topic2 - Card1187
            - Topic3 - Card0543
            - Topic4 - Card2620
            - Topic5 - Card2049
            - Topic6 - Card1725
            - Topic7 - Card2962
            - Topic8 - Card2120
        Remove improvement (deck_removed_improvement): # You're removing the same, specific card to each deck, so some will get better and some will get worse.
            - Topic0 - 0.8090473		->	0.954171
            - Topic1 - 0.00049178087	->	0.0005082074
            - Topic2 - 5.3064232e-05	->	5.4813823e-05
            - Topic3 - 0.00014593646	->	0.00015074816
            - Topic4 - 0.00018057834	->	0.00018654538
            - Topic5 - 0.00043903533	->	0.0004535354
            - Topic6 - 3.1631585e-05	->	3.2679727e-05
            - Topic7 - 4.6155532e-05	->	4.767514e-05
            - Topic8 - 0.14680731		->	0.00022770253
        Add improvement (deck_added_improvement): # You're adding the same, specific card to each deck, so some will get better and some will get worse.
            - Topic0 - 0.8090473		->	0.5478563
            - Topic1 - 0.00049178087	->	0.00048721273
            - Topic2 - 5.3064232e-05	->	5.2571544e-05
            - Topic3 - 0.00014593646	->	0.00014458569
            - Topic4 - 0.00018057834	->	0.00017888573
            - Topic5 - 0.00043903533	->	0.0004349573
            - Topic6 - 3.1631585e-05	->	3.1343916e-05
            - Topic7 - 4.6155532e-05	->	4.5724886e-05
            - Topic8 - 0.14680731		->	0.40840778