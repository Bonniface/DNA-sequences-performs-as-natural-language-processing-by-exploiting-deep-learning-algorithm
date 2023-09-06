# README #

immune2vec is an embedding model for embedding gentic/protein sequences of immnue receptors.
This repository includes the source code for creating and using the embedding model and 2 examples of classifier which uses the immune2vec.

Immune repertoire classification:
The model uses immune2vec for embedding the cdr3 amino acid sequences to vectors with fixed size. 
It is then finds clusters using euclidean distance and complete linkage hierarchical clustering and builds feature tables based on each repertoire 
frequency in the selected clusters. Finally a logistic regression with l2 regularisation penalty is used for the classification.

Immune sequence classification:
The model uses immune2vec for embedding trimmed cdr3 sequences to vectors with fixed size. The embedding can be done in different configuration and datasets. 
After completing the embedding a classifier is trained (number of classifiers are available) with per sequence labeling. 

### To setup and run the repertoire model example ###

* Prerequisites: python3.* with gensim package version 3.8.3 installed (use "python3 -m pip install gensim=3.8.3") and ray package installed (use "pip3 install ray")
* Clone this repository
* Download the example datasets files celiac_igh.tsv.gz and hcv_bcr_prepared.tsv.gz from https://www.dropbox.com/s/zzju5azguwfqg4s/celiac_tg2.tar.gz?dl=0
  and https://www.dropbox.com/s/s175byiwvmk9nrx/hcv_bcr_prepared.tsv.gz?dl=0, unzip them and place them in the immune2vec_model folder
* Run "python3 -m examples.hcv_bcr_example"
* Each component of the model can run independently from command line (for example "python -m dataset.split_dataset_folds --help") 
* For using the prepare_dataset api installation of Bio and changeo packages is required.

### To setup and run the vfamily model example ###

* Prerequisites: python3.* with gensim package version 3.8.3 installed (use "python3 -m pip install gensim=3.8.3")  
* Clone this repository
* Download the example datasets files igh_celiac.tsv.gz, igh_hcv.tsv.gz and igh_flu.tsv.gz from https://www.dropbox.com/s/6xccbfskc2fdemx/celiac_igh.tsv.gz?dl=0, 
  https://www.dropbox.com/s/i84r5h2sbw6h0dd/hcv_igh.tsv.gz?dl=0 and https://www.dropbox.com/s/v0v1kyzmerpr0ki/flu_igh.tsv.gz?dl=0.
  unzip them and place them in the immune2vec_model folder.
* Run "python3 -m examples.vfamily_example.py"
* For using the prepare_dataset api installation of Bio and changeo packages is required.

### Model Components ###

* dataset -> prepare_dataset/split_dataset_folds/split_vectors_folds
* embedding -> generate_model/generate_vectors
* feature_engineering -> vec_hierarchical_clustering/build_vec_feature_list/build_vec_feature_table/rf_fe
* classification -> lg