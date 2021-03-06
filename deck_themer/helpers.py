import csv
import pandas as pd
import tomotopy as tp
from deck_themer.model_funcs import *


# Jack's note: Eduardo Coronado's (@ecoronado92 on Twitter, author of the model_funcs.py file
# referenced above) assistance was invaluable, and I will be forever grateful for his patience,
# his expertise, and his willingness to share both with me, a mostly unlearned pleb.


def corpus_maker(f_name):
    """
    Takes a csv file and creates a set of decklists in list of lists,
        DataFrame, and tomotopy Corpus objects.
    Parameters:
        f_name: .csv file
            Term-document matrix: columns are card names, rows are decks,
                and each cell is the count of the given card in the given deck.
    :return:
        corpus: list of list of str
            All decklists represented as list of lists of strings
        processed_corpus: tomotopy.Corpus() object
            Decklist corpus in tomotopy Corpus format.
        df: pandas.DataFrame() object
            DataFrame version of the corpus object.
    """
    with open(file=f_name, encoding='utf-8', mode='r') as f:
        reader = csv.reader(f)
        card_names = next(reader)
        card_names[0] = 'Deck Numbers'
        data = list(list(deck) for deck in csv.reader(f, delimiter=','))
    corpus = []
    for row in data:
        deck = []
        for col in range(1, len(card_names)):
            if int(col) >= 1:
                for x in range(0, int(row[col])):
                    deck.append(card_names[col])
        corpus.append(deck)
    df = pd.DataFrame(data=data, columns=card_names)
    processed_corpus = tp.utils.Corpus()
    for decklist in corpus:
        processed_corpus.add_doc(decklist)
    return corpus, processed_corpus, df


def create_lda(tw=tp.TermWeight.IDF, min_cf=0, min_df=5, rm_top=0, k=2, alpha=0.1, eta=1, seed=101, corpus=None):
    """
    Creates a tomotopy LDAModel()
    Parameters:
        tw: Union[int, TermWeight]
            term weighting scheme in https://bab2min.github.io/tomotopy/v0.8.0/en/#tomotopy.TermWeight ;
            I chose the default to be inverse document frequency, which means that cards that appear in
            almost all decks are weighted lower than cards that appear in very few decks.
        min_cf: int
            Unless I'm mistaken, this is the minimum number of times that a card must appear at all in
            any deck to be included. However, since the vast majority of cards can be included at most
            once, this is almost always going to be the same as min_df.
        min_df: int
            Minimum number of times that a card must appear in a deck to be included in the analysis;
            default is set to 5.
        rm_top: int
            When ranking the most popular cards that are included in a given commander's decks, this
            parameter will remove the top n of them. Default is 0.
        k: int
            Number of themes/archetypes to sort decks into from 1 ~ 32,767. The default value is 2.
        alpha: float
            "hyperparameter of Dirichlet distribution for document-topic". Increasing alpha ... Based
            on advice from Eduardo Coronado (@ecoronado92 on Twitter), default for alpha is set to 0.1.
        eta: float
            "hyperparameter of Dirichlet distribution for topic-word". Increasing eta ... Based on
            experimentation, default for eta is 1.
        seed: int
            Random seed. Set to 101 as default in an attempt to duplicate results; however, said
            duplication has proven to be... elusive.
        corpus: tomotopy Corpus
            A list of documents to be added into the model. If None, documents have to be added
            after the model is created through LDAModel.add_doc() before the model can be trained.
    :return:
        tomotopy LDA model object
    """

    lda = tp.LDAModel(tw=tw, min_cf=min_cf, min_df=min_df, rm_top=rm_top, k=k, alpha=alpha, eta=eta,
                      seed=seed, corpus=corpus)
    return lda


def create_hdp(tw=tp.TermWeight.IDF, min_cf=0, min_df=5, rm_top=0, initial_k=2, alpha=0.1, eta=1,
               gamma=1, seed=101, corpus=None):
    """
    Creates a tomotopy HDPModel()
    Parameters:
        tw: Union[int, TermWeight]
            term weighting scheme in https://bab2min.github.io/tomotopy/v0.8.0/en/#tomotopy.TermWeight ;
            I chose the default to be inverse document frequency, which means that cards that appear in
            almost all decks are weighted lower than cards that appear in very few decks.
        min_cf: int
            Unless I'm mistaken, this is the minimum number of times that a card must appear at all in
            any deck to be included. However, since the vast majority of cards can be included at most
            once, this is almost always going to be the same as min_df.
        min_df: int
            Minimum number of times that a card must appear in a deck to be included in the analysis;
            default is set to 5.
        rm_top: int
            When ranking the most popular cards that are included in a given commander's decks, this
            parameter will remove the top n of them. Default is 0.
        initial_k: int
            Number of themes/archetypes that you THINK this commander's decks can be sorted into.
            This number does not dictate, per se, how many themes will be identified once the analysis
            is over. Perhaps a good place to start would be with how many themes have been currently
            identified for this commander already on EDHREC? The default value is 2.
        alpha: float
            "concentration coe[f]ficient of Dirichlet Process for document-table". Increasing alpha ... Based
            on advice from Eduardo Coronado (@ecoronado92 on Twitter), default for alpha is set to 0.1.
        eta: float
            "hyperparameter of Dirichlet distribution for topic-word". Increasing eta ... Based on
            experimentation, default for eta is 1.
        gamma: float
            "concentration coef[f]icient of Dirichlet [p]rocess for table-topic". Sets the overall number
            of themes/archetypes that the decks can share. Increasing gamma increases the number of
            themes that can be identified. Based on advice from Eduardo Coronado (@ecoronado92 on
            Twitter), default for alpha is set to 1.
        seed: int
            Random seed. Set to 101 as default in an attempt to duplicate results; however, said
            duplication has proven to be... elusive.
        corpus: tomotopy Corpus
            A list of documents to be added into the model. If None, documents have to be added
            after the model is created through HDPModel.add_doc() before the model can be trained.

    :return:
        tomotopy HDP model object
    """

    hdp = tp.HDPModel(tw=tw, min_cf=min_cf, min_df=min_df, rm_top=rm_top, initial_k=initial_k,
                      alpha=alpha, eta=eta, gamma=gamma, seed=seed, corpus=corpus)
    return hdp


def lda_param_checker(tw=tp.TermWeight.IDF, min_cf_0=0, min_cf_f=1, min_cf_s=1, min_df_0=0,
                      min_df_f=1, min_df_s=1, rm_top_0=0, rm_top_f=1, rm_top_s=1, k_0=2,
                      k_f=12, k_s=3, alpha_0=-1, alpha_f=0, alpha_s=1, eta_0=0, eta_f=1,
                      eta_s=1, seed=101, corpus=None, burn=100, train=1001, word_list=None,
                      card_count=30, to_excel=False, fname='param_checking.xlsx'):
    """
    Method to automatically iterate through different LDA parameters to compare results
    Parameters
        tw: Union[int, TermWeight]
            term weighting scheme in https://bab2min.github.io/tomotopy/v0.8.0/en/#tomotopy.TermWeight ;
            I chose the default to be inverse document frequency, which means that cards that appear in
            almost all decks are weighted lower than cards that appear in very few decks.
        min_cf_0: int
            Starting minimum card collection frequency
        min_cf_f: int
            Ending minimum card collection frequency
        min_cf_s: int
            Minimum card collection frequency step size
        min_df_0: int
            Starting minimum deck collection frequency
        min_df_f: int
            Ending minimum deck collection frequency
        min_df_s: int
            Minimum deck collection frequency step size
        rm_top_0: int
            Starting number of top cards to exclude
        rm_top_f: int
            Ending number of top cards to exclude
        rm_top_s: int
            Top cards to exclude step size
        k_0: int
            Starting number of topics
        k_f: int
            Ending number of topics
        k_s: int
            Number of topics to increase by per iteration
        alpha_0: int
            Starting number for the alpha hyperparameter as a power of ten, i.e. alpha = 10^(alpha_0)
        alpha_f: int
            Ending number for the alpha hyperparameter as a power of ten, i.e. alpha = 10^(alpha_f)
        alpha_s: int
            Step size for the powers of ten of the alpha hyperparameter
        eta_0: int
            Starting number for the eta hyperparameter as a power of ten, i.e. eta = 10^(eta_0)
        eta_f: int
            Ending number for the eta hyperparameter as a power of ten, i.e. eta = 10^(eta_f)
        eta_s: int
            Step size for the powers of ten of the eta hyperparameter
        seed: int
            Random seed. Set to 101 as default in an attempt to duplicate results; however, said
            duplication has proven to be... elusive.
        corpus: tomotopy Corpus
            A list of documents to be added into the model. Method will not function without corpus.
        burn: int
            Number of initial training iterations to discard the results of?
        train: int
            Number of iterations to train over
        word_list: list of lists of strings
            Collection of decklists with each card name represented as a string.
        card_count: int
            Number of cards used to evaluate card coherence.
        to_excel: boolean
            Output the resulting DataFrame to Excel spreadsheet?
        fname: string ending in '.xlsx'
            If to_excel == True, filename of the resulting Excel spreadsheet.
    :return:
        DataFrame that lists the results of the preceding iterations. Contains the following columns:
            k - number of topics
            Avg. LL - Average log likelihood per word (not really sure what this means,
                but I think that lower is better)
            LL Std. Dev. - Log Likelihood standard deviation
            LL CV - Log Likelihood coefficient of variance (Std. Dev./Average)
            Perplexity - Perplexity of the model (don't know what this means,
                but pretty sure that lower is better
            Coherence - (C_V) Coherence of the model. Shooting for ... 0.65? Or between
                0.7 and 0.8? I'm honestly not sure. I think that you'll get better
                results shooting for the latter.
    """

    results_lists = [['tw', 'Min. f_col', 'Min. f_doc', 'Top n Terms Removed',
                      'k', 'alpha', 'eta', 'Avg. LL', 'LL Std. Dev.', 'LL CV',
                      'Perplexity']]
    average_coherences = []
    coh_std_dev = []
    coh_cv = []
    for cf in range(min_cf_0, min_cf_f, min_cf_s):
        print("Collection Frequency = " + str(cf))
        for df in range(min_df_0, min_df_f, min_df_s):
            print("Document Frequency = " + str(df))
            for rm in range(rm_top_0, rm_top_f, rm_top_s):
                print("Remove Top " + str(rm) + " Words")
                for k in range(k_0, k_f, k_s):
                    print(str(k) + " Topics")
                    for a in range(alpha_0, alpha_f, alpha_s):
                        print("alpha = " + str(10 ** a))
                        for e in range(eta_0, eta_f, eta_s):
                            print("eta = " + str(10 ** e))
                            ll_list = []
                            lda = tp.LDAModel(tw=tw, min_cf=cf, min_df=df, rm_top=rm, k=k,
                                              alpha=10 ** a, eta=10 ** e, seed=seed, corpus=corpus)
                            lda.burn_in = burn
                            lda.train(0)
                            for i in range(0, train, 100):
                                lda.train(100)
                                ll_list.append(lda.ll_per_word)
                            lda_mean = sum(ll_list) / len(ll_list)
                            lda_variance = sum([((x - lda_mean) ** 2) for x in ll_list]) / len(ll_list)
                            lda_std_dev = lda_variance ** 0.5
                            lda_cv = lda_std_dev / lda_mean
                            # I believe that the following method can be used even though it was designed for HDP
                            lda_topics = get_lda_topics(lda, card_count)
                            # I believe that the following method can be used even though it was designed for HDP
                            results_list = [str(tw), cf, df, rm, k, 10 ** a, 10 ** e,
                                            lda_mean, lda_std_dev, lda_cv,
                                            lda.perplexity]
                            topic_coherences = eval_coherence_by_topic(lda, deck_lists=word_list)
                            results_list.extend(topic_coherences)
                            average_coh = eval_coherence(lda_topics, word_list)
                            average_coherences.append(average_coh)
                            coh_variance = sum([((x - average_coh) ** 2) for x in topic_coherences]) / len(
                                topic_coherences)
                            coh_std_dev.append(coh_variance ** 2)
                            coh_cv.append((coh_variance ** 2) / average_coh)
                            results_lists.append(results_list)
    for num_top in range(0, lda.k):
        results_lists[0].append('Top ' + str(num_top) + ' Coherence')
    df = pd.DataFrame(data=results_lists[1:], columns=results_lists[0])
    df['Average Coherence'] = average_coherences
    df['Coherence Std Dev'] = coh_std_dev
    df['Coherence CV'] = coh_cv
    if to_excel:
        df.to_excel(fname, encoding='utf-8')
    return df


def hdp_param_checker(tw=tp.TermWeight.IDF, min_cf_0=0, min_cf_f=1, min_cf_s=1, min_df_0=0,
                      min_df_f=1, min_df_s=1, rm_top_0=0, rm_top_f=1, rm_top_s=1, k0_0=2,
                      k0_f=12, k0_s=3, alpha_0=-1, alpha_f=0, alpha_s=1, eta_0=0, eta_f=1,
                      eta_s=1, gamma_0=0, gamma_f=1, gamma_s=1, seed=101, corpus=None, burn=100,
                      train=1001, word_list=None, card_count=30, to_excel=False, fname='param_checking.xlsx'):
    """
    Method to automatically iterate through different HDP parameters to compare results
    Parameters
        tw: Union[int, TermWeight]
            term weighting scheme in https://bab2min.github.io/tomotopy/v0.8.0/en/#tomotopy.TermWeight ;
            I chose the default to be inverse document frequency, which means that cards that appear in
            almost all decks are weighted lower than cards that appear in very few decks.
        min_cf_0: int
            Starting minimum card collection frequency
        min_cf_f: int
            Ending minimum card collection frequency
        min_cf_s: int
            Minimum card collection frequency step size
        min_df_0: int
            Starting minimum deck collection frequency
        min_df_f: int
            Ending minimum deck collection frequency
        min_df_s: int
            Minimum deck collection frequency step size
        rm_top_0: int
            Starting number of top cards to exclude
        rm_top_f: int
            Ending number of top cards to exclude
        rm_top_s: int
            Top cards to exclude step size
        k0_0: int
            Starting number of initial topics
        k0_f: int
            Ending number of initial topics
        k0_s: int
            Number of initial topics step size
        alpha_0: int
            Starting number for the alpha hyperparameter as a power of ten, i.e. alpha = 10^(alpha_0)
        alpha_f: int
            Ending number for the alpha hyperparameter as a power of ten, i.e. alpha = 10^(alpha_f)
        alpha_s: int
            Step size for the powers of ten of the alpha hyperparameter
        eta_0: int
            Starting number for the eta hyperparameter as a power of ten, i.e. eta = 10^(eta_0)
        eta_f: int
            Ending number for the eta hyperparameter as a power of ten, i.e. eta = 10^(eta_f)
        eta_s: int
            Step size for the powers of ten of the eta hyperparameter
        gamma_0: int
            Starting number for the gamma hyperparameter as a power of ten, i.e. gamma = 10^(gamma_0)
        gamma_f: int
            Ending number for the gamma hyperparameter as a power of ten, i.e. gamma = 10^(gamma_f)
        gamma_s: int
            Step size for the powers of ten of the gamma hyperparameter
        seed: int
            Random seed. Set to 101 as default in an attempt to duplicate results; however, said
            duplication has proven to be... elusive.
        corpus: tomotopy Corpus
            A list of documents to be added into the model. Method will not function without model.
        burn: int
            Number of initial training iterations to discard the results of?
        train: int
            Number of iterations to train over
        word_list: list of lists of strings
            Collection of decklists with each card name represented as a string.
        card_count: int
            Number of cards used to evaluate card coherence.
        to_excel: boolean
            Output the resulting DataFrame to Excel spreadsheet?
        fname: string ending in '.xlsx'
            If to_excel == True, filename of the resulting Excel spreadsheet.
    :return:
        DataFrame that lists the results of the preceding iterations. Contains the following columns:
            k - number of topics (not all of which are live; not sure why this is relevant)
            Live k - number of topics that are actually viable
            Avg. LL - Average log likelihood per word (not really sure what this means,
                but I think that lower is better)
            LL Std. Dev. - Log Likelihood standard deviation
            LL CV - Log Likelihood coefficient of variance (Std. Dev./Average)
            Perplexity - Perplexity of the model (don't know what this means,
                but pretty sure that lower is better
            Coherence - (C_V) Coherence of the model. Shooting for ... 0.65? Or between
                0.7 and 0.8? I'm honestly not sure
    """

    results_lists = [['tw', 'Min. f_col', 'Min. f_doc', 'Top n Terms Removed', 'Initial k',
                      'alpha', 'eta', 'gamma', 'k', 'Live k', 'Avg. LL', 'LL Std. Dev.', 'LL CV',
                      'Perplexity']]
    average_coherences = []
    coh_std_dev = []
    coh_cv = []
    max_live_top = 0
    for cf in range(min_cf_0, min_cf_f, min_cf_s):
        print("Collection Frequency = " + str(cf))
        for df in range(min_df_0, min_df_f, min_df_s):
            print("Document Frequency = " + str(df))
            for rm in range(rm_top_0, rm_top_f, rm_top_s):
                print("Remove Top " + str(rm) + " Words")
                for k in range(k0_0, k0_f, k0_s):
                    print(str(k) + " Initial Topics")
                    for a in range(alpha_0, alpha_f, alpha_s):
                        print("alpha = " + str(10 ** a))
                        for e in range(eta_0, eta_f, eta_s):
                            print("eta = " + str(10 ** e))
                            for g in range(gamma_0, gamma_f, gamma_s):
                                print("gamma = " + str(10 ** g))
                                ll_list = []
                                hdp = tp.HDPModel(tw=tw, min_cf=cf, min_df=df, rm_top=rm, initial_k=k,
                                                  alpha=10 ** a, eta=10 ** e, gamma=10 ** g, seed=seed, corpus=corpus)
                                hdp.burn_in = burn
                                hdp.train(0)
                                for i in range(0, train, 100):
                                    hdp.train(100)
                                    ll_list.append(hdp.ll_per_word)
                                hdp_mean = sum(ll_list) / len(ll_list)
                                hdp_variance = sum([((x - hdp_mean) ** 2) for x in ll_list]) / len(ll_list)
                                hdp_std_dev = hdp_variance ** 0.5
                                hdp_cv = hdp_std_dev / hdp_mean
                                hdp_topics = get_hdp_topics(hdp, card_count)
                                # hdp_coh = eval_coherence(hdp_topics, word_list=word_list)
                                results_list = [str(tw), cf, df, rm, k, 10 ** a, 10 ** e, 10 ** g, hdp.k,
                                                hdp.live_k, hdp_mean, hdp_std_dev, hdp_cv,
                                                hdp.perplexity]
                                topic_coherences = eval_coherence_by_topic(hdp, deck_lists=word_list)
                                results_list.extend(topic_coherences)
                                average_coh = eval_coherence(hdp_topics, word_list)
                                average_coherences.append(average_coh)
                                coh_variance = sum([((x - average_coh) ** 2) for x in topic_coherences]) / len(
                                    topic_coherences)
                                coh_std_dev.append(coh_variance ** 2)
                                coh_cv.append((coh_variance ** 2) / average_coh)
                                results_lists.append(results_list)
                                if hdp.live_k > max_live_top:
                                    max_live_top = hdp.live_k
    for num_top in range(0, max_live_top):
        results_lists[0].append('Top ' + str(num_top) + ' Coherence')
    df = pd.DataFrame(data=results_lists[1:], columns=results_lists[0])
    df['Average Coherence'] = average_coherences
    df['Coherence Std Dev'] = coh_std_dev
    df['Coherence CV'] = coh_cv
    if to_excel:
        df.to_excel(fname, encoding='utf-8')
    return df


def lda_topic_outputter(lda_model, card_count=30, to_excel=False, fname='lda_output.xlsx'):
    """
    Given a trained tomotopy LDAModel(), outputs the top n = card_count cards for all topics
        with the option to also output to Excel spreadsheet.
    Parameters:
        lda_model: tomotopy.LDAModel()
            Trained tomotopy LDA model
        to_excel: boolean
            Should the resulting DataFrame also be saved as an Excel spreadsheet?
        fname: str ending in ".xlsx"
            If to_excel = True, the filename for the resulting Excel spreadsheet
    :return:
        topics: DataFrame
            DataFrame for the given LDA model. DataFrame has three columns:
            Topic Number - topic number that the LDA model randomly associated with the topic
            Card Name - name of card that is present in a given topic
            Weight - how prevalent a given card is in a given topic
    """
    topics = []
    for topic in get_lda_topics(lda_model, top_n=card_count).keys():
        for card in get_lda_topics(lda_model, card_count)[topic]:
            topics.append([topic, card[0], card[1]])
    df = pd.DataFrame(data=topics, columns=['Topic Number', 'Card Name', 'Weight'])
    if to_excel:
        df.to_excel(fname, encoding="utf-8")
    return df


def hdp_topic_outputter(hdp_model, card_count=30, to_excel=False, fname='hdp_output.xlsx'):
    """
    Given a trained tomotopy HDPModel(), outputs the top n = card_count cards for all topics
        with the option to also output to Excel spreadsheet.
    Parameters:
        hdp_model: tomotopy.HDPModel()
            Trained tomotopy HDP model
        to_excel: boolean
            Should the resulting DataFrame also be saved as an Excel spreadsheet?
        fname: str ending in ".xlsx"
            If to_excel = True, the filename for the resulting Excel spreadsheet
    :return:
        topics: DataFrame
            DataFrame for the given HDP model. DataFrame has three columns:
            Topic Number - topic number that the HDP model randomly associated with the topic
            Card Name - name of card that is present in a given topic
            Weight - how prevalent a given card is in a given topic
    """
    topics = []
    for topic in get_hdp_topics(hdp_model, top_n=card_count).keys():
        for card in get_hdp_topics(hdp_model, card_count)[topic]:
            topics.append([topic, card[0], card[1]])
    df = pd.DataFrame(data=topics, columns=['Topic Number', 'Card Name', 'Weight'])
    if to_excel:
        df.to_excel(fname, encoding="utf-8")
    return df


def decks_measurer(lda, decklists, to_excel=False, fname="decks_infer.xlsx"):
    """
    Given a tomotopy LDAModel() and a list of decklists, returns a DataFrame that
        shows how much each decklist aligns with each identified topic, with the
        option to also output to an Excel spreadsheet.
    Parameters:
        lda: tomotopy.LDAModel() object or tomotopy.HDPModel() object
            A trained tomotopy LDAModel() object or a trained tomotopy HDPModel() object. If an
            HDPModel() object is passed in, an LDAModel() object will be generated from the
            HDPModel().convert_to_lda() method.
        decklists: pd.DataFrame()
            Pandas DataFrame of decks, with columns as card names and rows as decks
        to_excel: boolean
            Whether or not to output the DataFrame to an Excel spreadsheet
        fname: str ending in ".xlsx"
            If to_excel = True, the filename of the results Excel spreadsheet
    :return:
        DataFrame with a column for each identified topic and a row for each deck where each cell is
            the "amount" that that deck is associated with that topic.
    """
    if type(lda) == tp.HDPModel:
        lda_model = lda.convert_to_lda()[0]
    else:
        lda_model = lda
    new_docs = []
    for decklist in decklists.index:
        deck = list(decklists.loc[decklist][decklists.loc[decklist] != str(0)][1:].index)
        infer = [decklists.loc[decklist][0], lda_model.infer(lda_model.make_doc(deck))[0]]
        new_docs.append(infer)
    df = pd.DataFrame(data=[item[1] for item in new_docs], index=[item[0] for item in new_docs])
    if to_excel:
        df.to_excel(fname, encoding="utf-8")
    return df


def get_lda_topics(lda, top_n=30):
    """Wrapper function to extract topics from trained tomotopy LDA model (adapted from
        @ecoronado's get_hdp_topics() method)

    ** Inputs **
    lda:obj -> LDAModel trained model
    top_n: int -> top n words in topic based on frequencies

    ** Returns **
    topics: dict -> per topic, an array with top words and associated frequencies
    """

    # Get most important topics by # of times they were assigned (i.e. counts)
    sorted_topics = [k for k, v in sorted(enumerate(lda.get_count_by_topics()), key=lambda x: x[1], reverse=True)]

    topics = dict()

    # For topics found, extract only those that are still assigned
    for k in sorted_topics:
        topic_wp = []
        for word, prob in lda.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp  # store topic word/frequency array

    return topics


def get_lda_word_topic_dist(lda, to_excel=False, fname='lda_topic_dist.xlsx'):
    """
    Given a trained tomotopy LDAModel(), returns a DataFrame with a column for
        each topic and a row for each card name that gives that card's likelihood
        of appearing in a deck of that theme.
    Parameters:
        lda: tomotopy.LDAModel() object
            Trained tomotopy LDA model.
        to_excel: boolean
            Should the resulting DataFrame also be saved to an Excel file?
        fname:
            If to_excel==True, filename of the resulting Excel spreadsheet.
    :return:
        DataFrame
            DataFrame, with columns equal to the number of LDA topics and rows
            equal to the size of the LDA's used vocabulary, that gives each
            card's likelihood of being included in that topic.
    """
    df = pd.DataFrame(data=(100 * lda.get_topic_word_dist(0)), columns=[0], index=lda.used_vocabs)
    for topic in range(1, lda.k):
        df[topic] = 100 * lda.get_topic_word_dist(topic)
    if to_excel:
        df.to_excel(fname, encoding="utf-8")
    return df


def deck_measurer(decklist, lda):
    """
    Given a decklist and a trained tomotopy LDAModel(), returns the likelihood
        that the given deck is associated with the model's topics.
    Parameters:
        decklist: list of str
            A single decklist represented as a list of card names (strings).
        lda: tomotopy.LDAModel object
            A trained tomotopy LDA model.
    :return:
        measured_deck: ndarray
            An array with length equal to the number of topics that shows how
                much the deck aligns with each topic.
    """

    return lda.infer(lda.make_doc(decklist))[0]


def id_outlier(lda, decklist, wtopic='max'):
    """
    Given a decklist and a trained tomotopy LDAModel(), returns the card in the
        decklist that least belongs in the given topic(s).
    Parameters:
        lda: tomotopy.LDAModel() object
            Trained tomotopy LDAModel.
        decklist: list of str
            Decklist represented as a list of strings of card names
        wtopic: int [0,lda.k) or str ('min','max', or 'all')
            If an integer between 0 and the LDAModel's k value, evaluates for the
                topic with that index.
            If 'min', evaluates for the topic that the deck aligns with the least.
            If 'max', evaluates for the topic that the deck aligns with the most.
            If 'all', evaluates for all identified topics.
    :return:
        str or dictionary
            If 'all', returns a dictionary where each topic index is a key associated
                with the card from the deck that's the outlier for that topic.
            Else, returns a string for the card name for the card in that deck least
                associated with that topic.
    """
    word_topic_dist = get_lda_word_topic_dist(lda)
    deck_themes = deck_measurer(decklist, lda)
    if wtopic == 'all':
        outliers = {}
        for topic in range(0, lda.k):
            card_weights = word_topic_dist[topic][word_topic_dist[topic].index.isin(decklist)].sort_values()
            outliers[topic] = card_weights.index[0]
        return outliers
    elif wtopic == 'min':
        chtopic = list(deck_themes).index(deck_themes.min())
    elif wtopic == 'max':
        chtopic = list(deck_themes).index(deck_themes.max())
    else:
        chtopic = wtopic
    card_weights = word_topic_dist[chtopic][word_topic_dist[chtopic].index.isin(decklist)].sort_values()
    outlier = card_weights.index[0]
    return outlier


def card_remover(decklist, card):
    """
    Given a decklist and a card name, returns the decklist with the card removed.
    Parameters:
        decklist: list of str
            A decklist represented by a list of strings of card names.
        card: str
            The card to be removed from the decklist
    :return:
        list of str
            Returns the same decklist, minus the card that was removed, as a list
                of strings
    """
    return [x for x in decklist if x != card]


def id_missing_common(lda, decklist, wtopic='max'):
    """
    Given a decklist and a trained tomotopy LDAModel(), returns the most-included
        card in a given topic or topics that isn't in the supplied decklist.
    Parameters:
        lda: tomotopy.LDAModel() object
            Trained tomotopy LDAModel.
        decklist: list of str
            Decklist represented as a list of strings of card names
        wtopic: int [0,lda.k) or str ('min','max', or 'all')
            If an integer between 0 and the LDAModel's k value, evaluates for the
                topic with that index.
            If 'min', evaluates for the topic that the deck aligns with the least.
            If 'max', evaluates for the topic that the deck aligns with the most.
            If 'all', evaluates for all identified topics.
    :return:
        str or dictionary
            If 'all', returns a dictionary where each topic index is a key associated
                with the most popular card from that topic that isn't in the given decklist
            Else, returns a string for the card name for the most popular card in
                that topic that's not in the given decklist.
    """
    word_topic_dist = get_lda_word_topic_dist(lda)
    deck_themes = deck_measurer(decklist, lda)
    if wtopic == 'all':
        missing = {}
        for topic in range(0, lda.k):
            card_index = 0
            missing_card = 0
            while missing_card == 0:
                if word_topic_dist[topic].sort_values(ascending=False).index[card_index] not in decklist:
                    missing_card = word_topic_dist[topic].sort_values(ascending=False).index[card_index]
                else:
                    card_index += 1
            missing[topic] = missing_card
        return missing
    elif wtopic == 'min':
        chtopic = list(deck_themes).index(deck_themes.min())
    elif wtopic == 'max':
        chtopic = list(deck_themes).index(deck_themes.max())
    else:
        chtopic = wtopic
    missing = 0
    card_index = 0
    while missing == 0:
        if word_topic_dist[chtopic].sort_values(ascending=False).index[card_index] not in decklist:
            missing = word_topic_dist[chtopic].sort_values(ascending=False).index[card_index]
        else:
            card_index += 1
    return missing


def card_adder(decklist, card):
    """
    Given a decklist and a card name, returns the decklist with the supplied card added in.
    Parameters:
        decklist: list of str
            Decklist represented by a list of strings of card names.
        card: str
            Card name to be added to the deck.
    :return:
        list of str
            Decklist with added card
    """
    new_decklist = decklist.copy()
    new_decklist.append(card)
    return new_decklist


def remove_improvement(lda=None, decklist=None, card_name=None):
    """
    Given a decklist, shows the deck's theme breakdown by theme before and after
        a specific card is removed. If no card name is provided, least common
        card for given theme is removed.
    Parameters
        lda: tomotopy.LDAModel() object
            Trained tomotopy LDAModel object.
        decklist: list of str
            A decklist, represented by a list of strings of card names.
        card_name: str or None
            Name of card to remove. If not provided, the least common card for
                each theme that is in the deck is used.
    :return:
        before_remove: ndarray
            Array of length lda.k + 1 that shows the given decklist's
                alignments with each topic.
        after_remove: ndarray
            Array of length lda.k + 1 that shows the given decklist's
                alignments with each topic after a specific card has been removed.
    """
    c_decklist = decklist.copy()
    outliers = {}
    for topic in range(0, lda.k):
        if card_name is None:
            outliers[topic] = id_outlier(lda=lda, decklist=c_decklist, wtopic=topic)
        else:
            outliers[topic] = card_name
        if outliers[0] not in c_decklist:
            print('Error: ' + outliers[0] + ' is not in provided decklist.')
            return None
    before_remove = {}
    after_remove = {}
    for topic in range(0, lda.k):
        c_decklist_r = card_remover(decklist=c_decklist.copy(), card=outliers[topic])
        before_remove[topic] = deck_measurer(decklist=c_decklist, lda=lda)[topic]
        after_remove[topic] = deck_measurer(decklist=c_decklist_r, lda=lda)[topic]
    return before_remove, after_remove


def add_improvement(lda=None, decklist=None, card_name=None):
    """
    Given a decklist, shows the deck's theme breakdown by theme before and after
        a specific card is added. If no card name is provided, most common card
        for a given theme that isn't already in the deck is added.
    Parameters
        lda: tomotopy.LDAModel() object
            Trained tomotopy LDAModel object.
        decklist: list of str
            A decklist, represented by a list of strings of card names.
        card_name: str or None
            Name of card to add. If not provided, the most common card for
            that theme that isn't already in the deck is used.
    :return:
        before_add: ndarray
            Array of length lda.k + 1 that shows the given decklist's
                alignments with each topic.
        after_add: ndarray
            Array of length lda.k + 1 that shows the given decklist's
                alignments with each topic after a specific card has been added.
    """
    c_decklist = decklist.copy()
    missing = {}
    for topic in range(0, lda.k):
        if card_name is None:
            missing[topic] = id_missing_common(lda=lda, decklist=c_decklist, wtopic=topic)
        else:
            missing[topic] = card_name
        if missing[0] in c_decklist:
            print('Error: ' + missing[0] + ' is already in provided decklist.')
            return None
    before_add = {}
    after_add = {}
    for topic in range(0, lda.k):
        c_decklist_a = card_adder(decklist=c_decklist.copy(), card=missing[topic])
        before_add[topic] = deck_measurer(decklist=c_decklist, lda=lda)[topic]
        after_add[topic] = deck_measurer(decklist=c_decklist_a, lda=lda)[topic]
    return before_add, after_add


def eval_coherence_by_topic(lda, deck_lists, card_count=30, coherence_type='c_v', with_std=False, with_support=False):
    '''
    Adapted from @ecoronado92's eval_coherence method in the model_funcs.py file.

    Wrapper function that uses gensim Coherence Model to compute topic coherence scores,
        separated by topic.

    ** Inputs **
    lda: trained tomotopy.LDAModel() or trained tomotopy.HDPModel()
    word_list: list -> decklists as list of lists of strings of card names
    coherence_typ: str -> type of coherence value to compute (see gensim for options)

    ** Returns **
    score: float -> coherence value
    '''

    if type(lda) == tp.HDPModel:
        lda_model = lda.convert_to_lda()[0]
    else:
        lda_model = lda

    # Build gensim objects
    vocab = corpora.Dictionary(deck_lists)
    corpus = [vocab.doc2bow(words) for words in deck_lists]

    # Build topic list from dictionary
    topic_list = get_lda_topics(lda_model, card_count)

    topic_dict = []
    for k, tups in topic_list.items():
        topic_tokens = []
        for w, p in tups:
            topic_tokens.append(w)

        topic_dict.append(topic_tokens)

    # Build Coherence model
    cm = CoherenceModel(topics=topic_dict, corpus=corpus, dictionary=vocab, texts=deck_lists,
                        coherence=coherence_type)

    score = cm.get_coherence_per_topic(segmented_topics=cm.segment_topics(), with_std=with_std,
                                       with_support=with_support)

    return score
