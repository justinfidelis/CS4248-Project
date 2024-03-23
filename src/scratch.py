from nltk.tag import pos_tag
from feature_extractor import regex_tokenize

sentence = "The dosages of L-NMMA and indomethacin infused were selected on the basis of previous experiments that found a reduction in muscle blood flow during exercise (28)."

tags = pos_tag(regex_tokenize(sentence), tagset="universal")

counts = {}
for _, t in tags:
    if t not in counts:
        counts[t] = 0
    counts[t] += 1

print(counts)