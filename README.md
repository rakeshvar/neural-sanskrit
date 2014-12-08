neural-sanskrit
===============

A Neural Network that learns to predict Sanskrit text and learns a vector representations for the Aksharas(syllables).

Use
-----
Build numerical corpus from text corpus like rAmAyaNa.txt 
	NOTE: needs to be run in python3
```sh
python3 corpus_builder.py [corpus.txt] # See Usage string for more options
```
input: [corpus.txt]
output: [corpus.pkl], [corpus.list]

Run n-gram model to assess training and test prob (Use python2)
```sh
python predictor_ngram.py [corpus.pkl] # See Usage string for more options
```

Run Neural Net to learn word representations 
(Needs you to install Theano. Available via github)

```sh
python predictor_cnn.py [corpus.pkl] # See Usage string for more options
```

input: [corpus.pkl]
output: [corpus_[error]_[default].pkl], 

Run Dimensional Reduction on output file
```sh
python dim_reduction.py [corpus_[error]_[default].pkl] 
```
input: [corpus_[error]_[default].pkl]
output: [corpus_[error]_[default]_pca.pkl], [corpus_[error]_[default]_tsne.pkl]

Visualize PCA and tSNE in 2D 
```sh
python visualize_lexicon.py [corpus_[error]_[default]_[pca/tsne].pkl] [corpus.list] 
```
input: [corpus_[error]_[default]_pca.pkl] or [corpus_[error]_[default]_tsne.pkl] 
	and [corpus.list] 
output: A bunch of plots that let you visualize the lexicon in 2/3 dimensions.
