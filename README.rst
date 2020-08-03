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