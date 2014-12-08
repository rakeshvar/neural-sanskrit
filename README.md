neural-sanskrit
===============

A Neural Network that learns to predict Sanskrit text and learns a vector representations for the Aksharas(syllables).

Use
-----
Build Corpus [corpus_builder.py] - NOTE: needs to be run in python3
```sh
python3 corpus_builder.py <path_to_file.txt> # See Usage string for more options
```
input: <path_to_file.txt>
output: <path_to_file.pkl>, <path_to_file.list>

Run n-gram model to assess training and test prob (Use python2)
```sh
python predictor_ngram.py <path_to_file.pkl> # See Usage string for more options
```

Run Neural Net to learn word representations 
(Needs you to install Theano. Available via github)
```sh
python predictor_cnn.py <path_to_file.pkl> # See Usage string for more options
```
input: <path_to_file.pkl>
output: <path_to_file_<test error>_<param_flag>.pkl>, 

Run Dimensional Reduction on output file
```sh
python dim_reduction.py <path_to_file_<test error>_<param_flag>.pkl> 
```
input: <path_to_file_<test error>_<param_flag>.pkl>
output: <path_to_file_<test error>_<param_flag>_pca.pkl>, <path_to_file_<test error>_<param_flag>_tsne.pkl>

Visualize PCA and tSNE in 2D 
```sh
python visualize_lexicon.py <path_to_file_<test error>_<param_flag>_<pca/tsne>.pkl> <path_to_file.list> 
```
input: <path_to_file_<test error>_<param_flag>_pca.pkl> or <path_to_file_<test error>_<param_flag>_tsne.pkl> 
	and <path_to_file.list> 
output: A bunch of plots that let you visualize the lexicon in 2/3 dimensions.
