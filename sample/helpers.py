import tomotopy as tp
import pandas as pd
from sample.model_funcs import *
import gensim.corpora as corpora
from gensim.models import CoherenceModel


def corpus_maker(csv_file):
    """

    Starts with a .csv file and outputs a tomotopy Corpus, a necessary input for both the LDA and the HDP models
    Parameters:
        csv_file: file that holds the decklists. .csv file needs to be in the following format:
            Columns = card names
            Rows = decks
            Cells = number of card (column) included in deck (row), which, other than basic lands and
                other cards that an EDH can have more than one copy of, will either be a 0 or a 1.

    :return:
        decklists: list of lists of strings
            Each deck as a list of card names included (strings)
        corpus: tomtoopy.utils.Corpus() object
            Returns a tomotopy corpus object

    """
    df = pd.read_csv(csv_file, header=0, index_col=0, encoding='utf-8')
    decklists = []
    for deck in df.iloc:
        decklist = []
        for card in deck.index:
            if deck[card] == 0:
                pass
            else:
                for x in range(0, deck[card]):
                    decklist.append(card)
        decklists.append(decklist)

    corpus = tp.utils.Corpus()
    for decklist in decklists:
        corpus.add_doc(decklist)
    return decklists, corpus


def create_lda(tw=tp.TermWeight('IDF'), min_cf=0, min_df=5, rm_top=0, k=2, alpha=0.1, eta=1, seed=101, corpus=None):
    """
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


def create_hdp(tw=tp.TermWeight('IDF'), min_cf=0, min_df=5, rm_top=0, initial_k=2, alpha=0.1, eta=1,
               gamma=1, seed=101, corpus=None):
    """
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
            "concentration coeficient of Dirichlet Process for document-table". Increasing alpha ... Based
            on advice from Eduardo Coronado (@ecoronado92 on Twitter), default for alpha is set to 0.1.
        eta: float
            "hyperparameter of Dirichlet distribution for topic-word". Increasing eta ... Based on
            experimentation, default for eta is 1.
        gamma: float
            "concentration coeficient of Dirichlet [p]rocess for table-topic". Sets the overall number
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


def hdp_param_checker(tw=tp.TermWeight('IDF'), min_cf_0=0, min_cf_f=1, min_cf_s=1, min_df_0=0,
                      min_df_f=1, min_df_s=1, rm_top_0=0, rm_top_f=1, rm_top_s=1, k0_0=2,
                      k0_f=11, k0_s=1, alpha_0=-1, alpha_f=0, alpha_s=1, eta_0=0, eta_f=1,
                      eta_s=1, gamma_0=0, gamma_f=1, gamma_s=1, seed=101, corpus=None, burn=100,
                      train=1001, word_list=None, card_count=30):
    """

    Method to automatically iterate through different HDP parameters to compare results

    tw: Union[int, TermWeight]
        term weighting scheme in https://bab2min.github.io/tomotopy/v0.8.0/en/#tomotopy.TermWeight ;
        I chose the default to be inverse document frequency, which means that cards that appear in
        almost all decks are weighted lower than cards that appear in very few decks.
    min_cf_0: int
        Starting minimum card collection frequency
    min_cf_f: int
        Ending minimum card collection frequency
    min_cf_s: int
        Minmum card collection frequency step size
    min_df_0: int
        Starting minimum deck collection frequency
    min_df_f: int
        Ending miniumum deck collection frequency
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
    :return:
        Dataframe that lists the results of the preceding iterations. Contains the following columns:
            k - number of topics (not all of which are live; not sure why this is relevant)
            Live k - number of topics that are actually viable
            Avg. LL - Average log likelihood per word (not really sure what this means,
                but I think that lower is better)
            LL Std. Dev. - Log Likelihood standard deviation
            LL CV - Log Likelihood coefficient of variance (Std. Dev./Average)
            Perplexity - Perplexity of the model (don't know what this means,
                but pretty sure that lower is better
            Coherence - (C_V) Coherence of the model. Shooting for between 0.7 and 0.8.
    """

    results_lists = [['tw', 'Min. f_collect', 'Min. f_doc', 'Top n Terms Removed', 'Initial k',
                      'alpha', 'eta', 'gamma', 'k', 'Live k', 'Avg. LL', 'LL Std. Dev.', 'LL CV',
                      'Perplexity', 'Coherence']]
    for cf in range(min_cf_0, min_cf_f, min_cf_s):
        for df in range(min_df_0, min_df_f, min_df_s):
            for rm in range(rm_top_0, rm_top_f, rm_top_s):
                for k in range(k0_0, k0_f, k0_s):
                    for a in range(alpha_0, alpha_f, alpha_s):
                        for e in range(eta_0, eta_f, eta_s):
                            for g in range(gamma_0, gamma_f, gamma_s):
                                ll_list = []
                                hdp = tp.HDPModel(tw=tw, min_cf=cf, min_df=df, rm_top=rm, initial_k=k,
                                                  alpha=a, eta=e, gamma=g, seed=seed, corpus=corpus)
                                for i in range(0, train, 1):
                                    hdp.burn_in = burn
                                    hdp.train(0)
                                    hdp.train(100)
                                    ll_list.append(hdp.ll_per_word)
                                hdp_mean = sum(ll_list) / len(ll_list)
                                hdp_variance = sum([((x - hdp_mean) ** 2) for x in ll_list]) / len(ll_list)
                                hdp_std_dev = hdp_variance ** 0.5
                                hdp_cv = hdp_std_dev / hdp_mean
                                hdp_topics = get_hdp_topics(hdp, card_count)
                                hdp_coh = eval_coherence(hdp_topics, word_list=word_list)
                                results_list = [str(tw), cf, df, rm, k, a, e, g, hdp.k,
                                                hdp.live_k, hdp_mean, hdp_std_dev, hdp_cv,
                                                hdp.perplexity, hdp_coh]
                                results_lists.append(results_list)
    df = pd.DataFrame(data=results_lists[1:], columns=results_lists[0])
    return df


def hdp_topic_outputter(hdp_model, card_count=30, to_excel=False, fname='hdp_output.xlsx'):
    """
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
    for topic in hdp_model.keys():
        for card in get_hdp_topics(hdp_model, card_count)[topic]:
            topics.append([topic, card[0], card[1]])
    df = pd.DataFrame(data=topics, columns=['Topic Number', 'Card Name', 'Weight'])
    if to_excel:
        df.to_excel(fname, encoding="utf-8")
    return df


def hdp_deck_measure(lda, decklists, to_excel=False, fname="decks_infer.xlsx"):
    """
    Parameters:
        lda: tomotopy LDA model THAT WAS GENERATED FROM A TRAINED HDP MODEL USING hdp.convert_to_lda()
        decklists: list of list of strings
            list of decks, each of which is a list of strings that represent card names in a given deck
        to_excel: boolean
            Whether or not to output the DataFrame to an Excel spreadsheet
        fname: str ending in ".xlsx"
            If to_excel = True, the filename of the results Excel spreadsheet
    :return:
        DataFrame with a column for each identified topic and a row for each deck where each cell is
            the "amount" that that deck is associated with that topic.
    """
    new_docs = []
    for decklist in decklists:
        infer = [decklist[0], lda[0].infer(lda[0].make_doc(decklist[1:]))[0]]
        new_docs.append(infer)
    df = pd.DataFrame(data=[item[1] for item in new_docs], index=[item[0] for item in new_docs])
    if to_excel:
        df.to_excel(fname, encoding="utf-8")
    return df
