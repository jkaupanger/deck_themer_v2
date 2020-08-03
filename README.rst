Deck_themer

Deck_themer is a python... package, I guess, that attempts to automatically extract deck archetypes from EDH/Commander decklists.

Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

- 2020-07-29: Initial code commit. None of this code is tested as of yet; it's my first attempt at converting my Jupyter notebook into "proper" Python code.

- 2020-07-31: Making methods' docstrings better.

Testing

There are three general groups of testing:

- LDA testing - Testing to get a feel for latent Dirichlet allocation, which is for when you "know" how many topics that there are
- HDP testing - Testing to get a feel for hierarchical Dirichlet processing, which is for when you are unsure of how many topics that there are
- Site utility testing - Testing to demonstrate the various methods that might more directly benefit a user of the site

Unfortunately, the data, itself, it proprietary, so I cannot supply an example CSV file with real data that this package will work with. The best I can do is to take real data and obfuscate it, hence 'CSV_files/obfuscated_tdm.csv'.

HDP Testing
    If you run the hdp_param_checking() method with the following parameters, you should good somewhere close to the following results:

    Hyperparameter Values:
        hdp_param_checker(tw=tp.TermWeight.ONE, min_df_0=5, min_df_f=6, k0_0=2, k0_f=10, k0_s=7, alpha_0=-1, alpha_f=1, eta_0=-1, eta_f=1, gamma_0=-1, gamma_f=1, corpus=corpus, word_list=<decklist_file>, to_excel=True, fname=<you_choose_filename>)

    Average Results:
        Average Live_k: ~14 +/- ~7

        Average Average Log Likelihood: -12.03 +/- 0.14

        Average Perplexity: 167618 +/- 23600

        Average Coherence: .884 +/- .0611