"""
This script is copied from https://github.com/CMGeval/Evaluating-CMG,
the replication package for "Evaluating Commit Message Generation: To BLEU Or Not To BLEU?"
accepted to ICSE NIER 2022.
"""

#!/usr/bin/env python
# coding: utf-8

# # The Log-MNEXT metric

# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from itertools import chain, product


def _generate_enums(hypothesis, reference, preprocess=str.lower):
    """
    Takes in string inputs for hypothesis and reference and returns
    enumerated word lists for each of them
    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :preprocess: preprocessing method (default str.lower)
    :type preprocess: method
    :return: enumerated words list
    :rtype: list of 2D tuples, list of 2D tuples
    """
    hypothesis_list = list(enumerate(preprocess(hypothesis).split()))
    reference_list = list(enumerate(preprocess(reference).split()))
    return hypothesis_list, reference_list


def exact_match(hypothesis, reference):
    """
    matches exact words in hypothesis and reference
    and returns a word mapping based on the enumerated
    word id between hypothesis and reference
    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    hypothesis_list, reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(hypothesis_list, reference_list)


def _match_enums(enum_hypothesis_list, enum_reference_list):
    """
    matches exact words in hypothesis and reference and returns
    a word mapping between enum_hypothesis_list and enum_reference_list
    based on the enumerated word id.
    :param enum_hypothesis_list: enumerated hypothesis list
    :type enum_hypothesis_list: list of tuples
    :param enum_reference_list: enumerated reference list
    :type enum_reference_list: list of 2D tuples
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                (enum_hypothesis_list.pop(i)[1], enum_reference_list.pop(j)[1])
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer()
):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.
    :param enum_hypothesis_list:
    :type enum_hypothesis_list:
    :param enum_reference_list:
    :type enum_reference_list:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    stemmed_enum_list1 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list
    ]

    stemmed_enum_list2 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
    ]

    word_match, enum_unmat_hypo_list, enum_unmat_ref_list = _match_enums(
        stemmed_enum_list1, stemmed_enum_list2
    )

    enum_unmat_hypo_list = (
        list(zip(*enum_unmat_hypo_list)) if len(enum_unmat_hypo_list) > 0 else []
    )

    enum_unmat_ref_list = (
        list(zip(*enum_unmat_ref_list)) if len(enum_unmat_ref_list) > 0 else []
    )

    enum_hypothesis_list = list(
        filter(lambda x: x[0] not in enum_unmat_hypo_list, enum_hypothesis_list)
    )

    enum_reference_list = list(
        filter(lambda x: x[0] not in enum_unmat_ref_list, enum_reference_list)
    )

    return word_match, enum_hypothesis_list, enum_reference_list


def stem_match(hypothesis, reference, stemmer=PorterStemmer()):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference
    :param hypothesis:
    :type hypothesis:
    :param reference:
    :type reference:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that
                   implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)


def _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.
    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype:  list of tuples, list of tuples, list of tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(
            chain.from_iterable(
                (
                    lemma.name()
                    for lemma in synset.lemmas()
                    if lemma.name().find("_") < 0
                )
                for synset in wordnet.synsets(enum_hypothesis_list[i][1])
            )
        ).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                enum_hypothesis_list.pop(i), enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def wordnetsyn_match(hypothesis, reference, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.
    :param hypothesis: hypothesis string
    :param reference: reference string
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of mapped tuples
    :rtype: list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )


def _enum_allign_words(
        enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer(), wordnet=wordnet
):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input
    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
        enum_hypothesis_list, enum_reference_list
    )

    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer
    )

    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )

    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_hypothesis_list,
        enum_reference_list,
    )


def allign_words(hypothesis, reference, stemmer=PorterStemmer(), wordnet=wordnet):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.
    :param hypothesis: hypothesis string
    :param reference: reference string
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_allign_words(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )


def _count_chunks(matches):
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to caluclate the
    fragmentation part of the metric.
    :param matches: list containing a mapping of matched words (output of allign_words)
    :return: Number of chunks a sentence is divided into post allignment
    :rtype: int
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
                matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(
        reference,
        hypothesis,
        preprocess=str.lower,
        stemmer=PorterStemmer(),
        wordnet=wordnet,
        alpha=0.85,
        beta=2.35,
        gamma=0.45,
        w_1=1,
        w_2=0.8,
        w_3=0.6
):
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf
    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.7398
        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.
    >>> round(meteor_score('this is a cat', 'non matching hypothesis'),4)
    0.0
    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    print(translation_length)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference, stemmer=stemmer)
    # print(matches)
    exact_m, _, _ = exact_match(hypothesis, reference)
    print(exact_m)
    stem_m, _, _ = _enum_stem_match(enum_hypothesis, enum_reference, stemmer=PorterStemmer())
    print(stem_m)
    syn_m, _, _ = _enum_wordnetsyn_match(enum_hypothesis, enum_reference, wordnet=wordnet)
    print(syn_m)

    exact_count = len(exact_m)
    stem_matches = set(stem_m).difference(exact_m)
    stem_count = len((set(stem_m).difference(exact_m)))
    syn_matches = set(syn_m).difference(stem_m)
    syn_count = len((set(syn_m).difference(stem_m)))
    tot_matches = list(exact_m) + list(stem_matches) + list(syn_matches)
    tot_alligned_matches = sorted(
        tot_matches, key=lambda wordpair: wordpair[0]
    )
    tot_alligned_count = len(tot_alligned_matches)
    # print(tot_alligned_matches)
    print(exact_count)
    print(stem_count)
    print(syn_count)
    # print(tot_alligned_count)
    matches_count = len(matches)
    print(matches_count)
    try:
        precision = float(w_1 * exact_count + w_2 * stem_count + w_3 * syn_count) / translation_length
        recall = float(w_1 * exact_count + w_2 * stem_count + w_3 * syn_count) / reference_length
        # precision = float(matches_count) / translation_length
        # recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    if tot_alligned_count == translation_length and tot_alligned_count == reference_length:
        penalty = 0
    else:
        penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean


def log_mnext_score(
        references,
        hypothesis,
        preprocess=str.lower,
        stemmer=PorterStemmer(),
        wordnet=wordnet,
        alpha=0.85,
        beta=2.35,
        gamma=0.45,
        w_1=1,
        w_2=0.8,
        w_3=0.6
):
    """
    Calculates METEOR score for hypothesis with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf
    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given hypothesis
    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    >>> hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'
    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    >>> reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    >>> reference3 = 'It is the practical guide for the army always to heed the directions of the party'
    >>> round(meteor_score([reference1, reference2, reference3], hypothesis1),4)
    0.7398
        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.
    >>> round(meteor_score(['this is a cat'], 'non matching hypothesis'),4)
    0.0
    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    return max(
        [
            single_meteor_score(
                reference,
                hypothesis,
                preprocess=preprocess,
                stemmer=stemmer,
                wordnet=wordnet,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            for reference in references
        ]
    )


# In[ ]:


import csv
import json

preds = []
refs = []
auth_1 = []
auth_2 = []
auth_3 = []
with open('human_annotations.csv') as csvfile:
    ader = csv.reader(csvfile)
    for row in ader:
        refs.append(row[1])
        preds.append(row[0])
        auth_1.append(row[2])
        auth_2.append(row[3])
        auth_3.append(row[4])

refs_n = refs[:100]
preds_n = preds[:100]

auth_1 = auth_1[:100]
for i in range(0, len(auth_1)):
    auth_1[i] = int(auth_1[i])
# print(auth_1)

auth_2 = auth_2[:100]
for i in range(0, len(auth_2)):
    auth_2[i] = int(auth_2[i])
# print(auth_2)

auth_3 = auth_3[:100]
for i in range(0, len(auth_3)):
    auth_3[i] = int(auth_3[i])
# print(auth_3)


# In[ ]:


# Removal of punctuation from predicted and reference texts
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

# refs_new = []
for i in range(len(auth_1)):
    for ele in refs_n[i]:
        if ele in punc:
            refs_n[i] = refs_n[i].replace(ele, "")

# preds_new = []
for i in range(len(auth_1)):
    for ele in preds_n[i]:
        if ele in punc:
            preds_n[i] = preds_n[i].replace(ele, "")
print(refs_n)
print(preds_n)

# In[ ]:


# Normalised author 1 scores
norm_auth_1 = []
max1 = max(auth_1)
min1 = min(auth_1)

for a in range(len(auth_1)):
    norm_auth_1.append((auth_1[a]) / (max1))
print(norm_auth_1)

# In[ ]:


# Normalised author 2 scores
norm_auth_2 = []
max1 = max(auth_2)
min1 = min(auth_2)

for a in range(len(auth_2)):
    norm_auth_2.append((auth_2[a]) / (max1))
print(norm_auth_2)

# In[ ]:


# Normalised author 3 scores
norm_auth_3 = []
max1 = max(auth_3)
min1 = min(auth_3)

for a in range(len(auth_3)):
    norm_auth_3.append((auth_3[a]) / (max1))
print(norm_auth_3)

# In[ ]:


# Average normalised author scores
avg_norm_score = []
for i in range(len(auth_1)):
    avg_norm_score.append(round((norm_auth_1[i] + norm_auth_2[i] + norm_auth_3[i]) / 3, 2))
print(avg_norm_score)

# In[ ]:


log_mnext = []

for i in range(len(auth_1)):
    log_mnext.append(round(log_mnext_score([refs_n[i]], preds_n[i]), 2))
print(log_mnext)

# In[ ]:


# Normalising the Log-MNEXT scores
norm_log_mnext = []

max_mn = max(log_mnext)
min_mn = min(log_mnext)

for a in range(len(log_mnext)):
    norm_log_mnext.append(round(((log_mnext[a]) / (max_mn)), 2))
print(norm_log_mnext)

# In[ ]:


from scipy.stats import spearmanr

# calculate spearman's correlation
coef, p = spearmanr(norm_log_mnext, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
    print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('correlated (reject H0) p=%.3f' % p)