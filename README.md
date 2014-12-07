neural-sanskrit
===============

A Neural Network that learns to predict Sanskrit text and learns a vector representations for the Aksharas(syllables).

Use
-----
Build Corpus [corpus_builder.py] - NOTE: needs to be run in python3
```sh
python corpus_builder.py <path_to_file.txt> # See Usage string for more options
```
input: <path_to_file.txt>
output: <path_to_file.pkl>, <path_to_file_ak.pkl>, <path_to_file.list>

Run n-gram model to assess training and test prob
```sh
python predictor_ngram.py <path_to_file.pkl> # See Usage string for more options
```

Run Neural Net to learn word representations [predictor_cnn.py]
```sh
python predictor_cnn.py <path_to_file.pkl> # See Usage string for more options
```
input: <path_to_file.pkl>
output: <path_to_file_<test error>.pkl>, 

Run Dimensional Reduction on output file
```sh
python dim_reduction.py <path_to_file_<test error>.pkl> 
```
input: <path_to_file_<test error>.pkl>
output: <path_to_file_<test error>_pca.pkl>, <path_to_file_<test error>_tsne.pkl>

Visualize PCA and tSNE in 2D 
```sh
python visual_output.py <int DIMENSIONS> <#occurences below which words not displayed> <path_to_file_<test error>_pca.pkl> <path_to_file_<test error>_tsne.pkl> <path_to_file_ak.pkl> <path_to_file.list> 
```
input: <int DIMENSIONS>. <int #occurences below which words not displayed>, <path_to_file_<test error>_pca.pkl>, <path_to_file_<test error>_tsne.pkl>, <path_to_file_ak.pkl>, <path_to_file.list> 
output: 2 graphs - one for PCA then after exit another for tSNE

NOTE: PCA results are colored based on the final character in the akshara, while tSNE is colored based on hierarchical clustering (agglomerative).Also, depending on your OS and support for Unicode characters you may or may not be able to view the Devanagri aksharas.
