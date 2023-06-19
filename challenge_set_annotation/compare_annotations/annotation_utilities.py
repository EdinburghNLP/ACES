#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# this is the list of phenomena and which option they need to be annotated with:
phenomena = {
    'addition':'add-omit',
    'ambiguous-translation-wrong-discourse-connective-since-causal':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-since-temporal':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-while-contrast':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-while-temporal':'diff_flexible',
    'ambiguous-translation-wrong-gender-female-anti':'diff_flexible',
    'ambiguous-translation-wrong-gender-female-pro':'diff_flexible',
    'ambiguous-translation-wrong-gender-male-anti':'diff_flexible',
    'ambiguous-translation-wrong-gender-male-pro':'diff_flexible',
    'ambiguous-translation-wrong-sense-frequent':'diff_flexible',
    'ambiguous-translation-wrong-sense-infrequent':'diff_flexible',
    'anaphoric_group_it-they:deletion':'annotate_word',
    'anaphoric_group_it-they:substitution':'annotate_word',
    'anaphoric_intra_non-subject_it:deletion':'annotate_word',
    'anaphoric_intra_non-subject_it:substitution':'annotate_word',
    'anaphoric_intra_subject_it:deletion':'annotate_word',
    'anaphoric_intra_subject_it:substitution':'annotate_word',
    'anaphoric_intra_they:deletion':'annotate_word',
    'anaphoric_intra_they:substitution':'annotate_word',
    'anaphoric_singular_they:deletion':'annotate_word',
    'anaphoric_singular_they:substitution':'annotate_word',
    'antonym-replacement':'REF_flexible',
    'commonsense-only-ref-ambiguous':'diff_flexible',
    'commonsense-src-and-ref-ambiguous':'diff_flexible',
    'copy-source':'whole_sentence',
    'coreference-based-on-commonsense':'mixed_flexible',
    'do-not-translate':'whole_sentence',
    'hallucination-date-time':'date',
    'hallucination-named-entity-level-1':'diff_flexible',
    'hallucination-named-entity-level-2':'REF_flexible',
    'hallucination-named-entity-level-3':'REF_flexible',
    'hallucination-number-level-1':'diff_flexible',
    'hallucination-number-level-2':'REF_flexible',
    'hallucination-number-level-3':'REF_flexible',
    'hallucination-real-data-vs-ref-word':'diff_flexible',
    'hallucination-real-data-vs-synonym':'diff_flexible',
    'hallucination-unit-conversion-amount-matches-ref':'units',
    'hallucination-unit-conversion-unit-matches-ref':'units',
    'hypernym-replacement':'REF_flexible',
    'hyponym-replacement':'REF_flexible',
    'lexical-overlap':'manual',
    'modal_verb:deletion':'add-omit',
    'modal_verb:substitution':'diff_flexible',
    'nonsense':'REF_flexible',
    'omission':'add-omit',
    'ordering-mismatch':'swap',
    'overly-literal-vs-correct-idiom':'diff_flexible',
    'overly-literal-vs-explanation':'diff_flexible',
    'overly-literal-vs-ref-word':'diff_flexible',
    'overly-literal-vs-synonym':'diff_flexible',
    'pleonastic_it:deletion':'annotate_word',
    'pleonastic_it:substitution':'annotate_word',
    'punctuation:deletion_all':'add-omit',
    'punctuation:deletion_commas':'add-omit',
    'punctuation:deletion_quotes':'add-omit',
    'punctuation:statement-to-question':'add-omit',
    'real-world-knowledge-entailment':'diff_flexible',
    'real-world-knowledge-hypernym-vs-distractor':'diff_flexible',
    'real-world-knowledge-hypernym-vs-hyponym':'diff_flexible',
    'real-world-knowledge-synonym-vs-antonym':'diff_flexible',
    'similar-language-high':'whole_sentence',
    'similar-language-low':'whole_sentence',
    'untranslated-vs-ref-word':'diff_flexible',   # here add-omit can be used for getting character level replacements too
    'untranslated-vs-synonym':'diff_flexible',
    'xnli-addition-contradiction':'manual',
    'xnli-addition-neutral':'manual',
    'xnli-omission-contradiction':'manual',
    'xnli-omission-neutral':'manual'
}

# given a sentence, return the tokens and their start and end indices
def tokenize(sentence):
    s0 = sentence.lower()
    s0 = re.sub('i̇', 'i', s0)
    s = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+\s", ' ', s0)
    s = re.sub(r"^[\"\[\]\.,!?:;'\(\)$“„”]+", ' ', s)
    s = re.sub(r"\s[\"\[\]\.,!?:;'\(\)$“„”]+",' ', s)
    s = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+$",' ', s)
    s = s.strip("\"[].,!?:;'\(\)$")
    tokenized = s.split()
    
    spans = []
    split = 0
    for token in tokenized:
        res = re.search(re.escape(token), s0[split:])
        start = res.start() + split
        end = res.end() + split
        spans.append((start, end))
        split = end
    return tokenized, spans

# choose whether the reference sentence or the good sentence was changed to create the incorrect translation.
def ref_or_good(ref, good, bad):
    g, g_spans = tokenize(good)
    b, b_spans = tokenize(bad)
    r, r_spans = tokenize(ref)
    g_change = diff_flexible(good, g, g_spans, bad, b, b_spans)
    r_change = diff_flexible(ref, r, r_spans, bad, b, b_spans)
    if len(r_change[0]["in_good"]["token"]) <= len(g_change[0]["in_good"]["token"]):
        return ref
    else:
        return good

# this finds if any number of words are replaced with any number of words - but only for one span
# if there is more than one span, for example as in
# good = "What does slip ring starter and a b starter mean"
# bad = "What does savory starter and c starter mean"
# then it finds it as one big span: ('slip ring starter and a b', [2, 3, 4, 5, 6, 7], 'savory starter and c', [2, 3, 4, 5])
def diff_flexible(good, g, g_spans, bad, b, b_spans, phenomena="default"):
    i = 0
    change = []
    while i < len(b) and i < len(g) and g[i] == b[i]:
        i += 1
    start = i
    if start < len(b) and start < len(g):
        i = len(g) - 1
        j = len(b) - 1
        while i > start and j > start and g[i] == b[j]:
            i -= 1
            j -= 1
        change.append({"in_good":{'token_index':list(range(start,i+1)), 
                    'character_span':(g_spans[start][0],g_spans[i][1]), 
                                  'token':good[g_spans[start][0]:g_spans[i][1]]}, 
                    "in_bad":{'token_index':list(range(start,j+1)), 
                    'character_span':(b_spans[start][0],b_spans[j][1]), 
                               'token':bad[b_spans[start][0]:b_spans[j][1]]}})          
        
    return change
