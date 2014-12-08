
# neural-sanskrit

A Neural Network that learns to predict Sanskrit text and learns a vector representations for the Aksharas(syllables).

## Usage

1) Build numerical corpus from text corpus like rAmAyaNa.txt. (Needs __python3__)
```sh
python3 corpus_builder.py <corpus.txt> 
# See Usage string for more options
```
    output: <corpus.pkl>, <corpus.list>


2) Run n-gram model to assess training and test prob (Use python2)
```sh
python predictor_ngram.py <corpus.pkl> 
# See Usage string for more options
```


3) Run Neural Net to learn word representations. (Needs [Theano](https://github.com/Theano/Theano))

```sh
python predictor_cnn.py <corpus.pkl> 
# See Usage string for more options
```
	output: <corpus_<error>_<default>.pkl> 

4) Run Dimensionality Reduction on output file
```sh
python dim_reduction.py <corpus_<error>_<default>.pkl> 
```
	output: <corpus_<error>_<default>_pca.pkl>
	output: <corpus_<error>_<default>_tsne.pkl>

5) Visualize PCA and tSNE in 2D 
```sh
python visualize_lexicon.py <corpus_<error>_<default>_<pca/tsne>.pkl> <corpus.list> 
# See Usage string for more options
```

	output: A bunch of plots that let you visualize the lexicon in 2 or 3 dimensions.

## Sample Visualization
![](https://github.com/rakeshvar/neural-sanskrit/blob/master/pca_vowel.png)
