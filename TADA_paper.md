# TADA: Phylogenetic augmentation of microbiome samples enhances phenotype classification 
### By Erfan Sayyari, Ban Kawas, and Siavash Mirarab

## Abstract

__Motivation:__ Learning associations of traits with the microbial composition of a set of samples is a fun- damental goal in microbiome studies. Recently, machine learning methods have been explored for this goal, with some promise. However, in comparison to other fields, microbiome data is high-dimensional and not abundant; leading to a high-dimensional low-sample-size under-determined system. Moreover, micro- biome data is often unbalanced and biased. Given such training data, machine learning methods often fail to perform a classification task with sufficient accuracy. Lack of signal is especially problematic when classes are represented in an unbalanced way in the training data; with some classes under-represented. The presence of inter-correlations among subsets of observations further compounds these issues. As a result, machine learning methods have had only limited success in predicting many traits from microbiome. Data augmentation consists of building synthetic samples and adding them to the training data and is a technique that has proved helpful for many machine learning tasks.

__Results:__ In this paper, we propose a new data augmentation technique for classifying phenotypes based on the microbiome. Our algorithm, called TADA, uses available data and a statistical generative model to create new samples augmenting existing ones, addressing issues of low-sample-size. In generating new samples, TADA takes into account phylogenetic relationships between microbial species. On two real datasets, we show that adding these synthetic samples to the training set improves the accuracy of down- stream classification, especially when the training data have an unbalanced representation of classes.

This repository is to describe the datasets and code introduced in this work.
### Installation

To install the code, you need the following python packages:

* [SEPP](https://github.com/smirarab/sepp/blob/master/sepp-package/README.md)
* [newick utility](http://cegg.unige.ch/newick_utils)
* [sciki-learn](https://scikit-learn.org/) 
* [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/)
* [Dendropy4](https://dendropy.org)
* [biom](http://biom-format.org/)

### Dataset preparation
The codes to prepare the AGP and Gevers datasets are avaiable on the repository at [AGP](src/utils/ipython/make_experiment_AGP.ipynb) and [Gevers](src/utils/ipython/make_experiment_Gevers.ipynb). 
### Tree placement
You could download and install [SEPP](https://github.com/smirarab/sepp/blob/master/sepp-package/README.md) from the github. Running SEPP is fairly easy, you need to provide a FASTA file which has all the fragments (16S) and a label. This will output 1) a JSON file which has information regarding the SEPP placements, and 2) a phylogeny, where the fragments inserted into the backbone tree. 
 

```
run-sepp.sh <input_seq> <label>
```

To remove backbone species from the SEPP output phylogeny, you could use the following commands using [newick utility](http://cegg.unige.ch/newick_utils)

``` bash
v=$(cat dna-sequences.fasta | grep  ">" | sed -e 's/>//' | tr '\n' ' ' | sed -e 's/ $/\n/') 

nw_topology -Ib <completed_tree.tre> > p

nw_prune -v <completed_tree.tre> $v > phylogeny.tre
```

__Note:__ If the number of fragments is huge, you could split FASTA files, run SEPP separately on each of them, and then merge the output JSON files (placements) using __merge\_placements.py__. Finally, you need to insert the fragments into the GG backbone (following commands), 

```
python merge_placements.py <comma-separated> <output_dir> ${name}_placement.json

gbin=$(dirname `grep -A1 "pplacer" $WS_HOME/sepp-package/sepp/.sepp/main.config |grep path|sed -e "s/^path=//g"` )

name=<label in sepp>

$gbin/guppy tog ${name}_placement.json

cat ${name}_placement.tog.tre | python ${name}_rename-json.py > ${name}_placement.tog.relabelled.tre

$gbin/guppy tog --xml ${name}_placement.json

cat ${name}_placement.tog.xml | python ${name}_rename-json.py > ${name}_placement.tog.relabelled.xml
```
## E1 (Augmentation)


### Augmentation with clustering (TADA-TVSV-C)

clustering (4, 8, 40 as the number of clusters). The seed numbers we used are 0, 42, 10126, and 24828 (5-fold, 4 seed numbers), and phenotypes be IBD or bmi:

```
python run_augmentation.py -t <phylogeny.tre> -s <dna-fragments.fasta> -b <feature-table.biom> -m <meta_data.csv> -p <options.csv> -o <output_dir> -c <phenotype> -e clustering -i <fold number> -f <number of folds> -r 1 --seed <seed_num> --num_clusters <number of clusters> --method braycurtis --rank 10 --final_column clusters_and_class_labels > <log> 2>&1
```

if the number of clusters is one, we run the code using:

```
python run_augmentation.py -t <phylogeny.tre> -s <dna-fragments.fasta> -b <feature-table.biom> -m <meta_data.csv> -p <options.csv> -o <output_dir> -c <phenotype> -e augmentation -i <fold-number> -f <numbder of folds> -r 1 --seed <seed_number> > <log> 2>&1
```


### Augmentation without clustering (TADA-SV and TADA-TVSV-m)


```
python run_augmentation.py -t <phylogeny.tre> -s <dna-fragments.fasta> -b <feature-table.biom> -m <meta_data.csv> -p <options.csv> -o <output_dir> -c <phenotype> -e augmentation -i <fold-number> -f <numbder of folds> -r 1 --seed <seed_number> > <log> 2>&1
```

### Classification
For classificaiton, we used [sciki-learn](https://scikit-learn.org/) implementation of [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) using 2000 trees (otherwise default values) and a two layers Mult-layer Perceptron classifier ([MLPC](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), 2000 and 1000 as dimensions for first and second layers respectively).

``` python
if method == "rf":
	clf = RandomForestClassifier(n_estimators=2000, n_jobs= n_jobs, random_state=42)
elif method == "nn":
	clf = MLPClassifier(hidden_layer_sizes=(2000, 1000, ), learning_rate='adaptive', random_state=42,  verbose=True, early_stopping=True)
```

The exact command to run classification for the input augmentation tar file (contains normalized augmented, training, and testing samples in three compressed [numpy npz format files](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) is as follow:

```
python run_classification_augmentation.py -z <input_augmentation.tar> -o <output_dir> -p <label> -m <classification_method (rf or nn)> -n <number of cpus> -d <seed_number fixed to 42>
```

### SMOTE and ADASYN
We used SMOTE and ADASYN implementation in [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/) package for python. We first use the training files to generate augmented samples with the following commands in python:

``` python
if method == "SMOTE":
	imbl = SMOTE(sampling_strategy=freq_dct, random_state=42, k_neighbors=5)
else:
	imbl = ADASYN(sampling_strategy=freq_dct, random_state=42, n_neighbors=5)
```
where __freq\_dct__ is a python dictionary explicitly setting the number of samples (to be generated) for each class label. We then run the classification (same as above) on the augmented plus original samples. The eact command to run SMOTE and then classification is

```bash
python run_classification_balance_by_orig_data.py -z <input_augmentation.tar> -o <output_dir> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42>  -e augmentation  --meta <meta_data.csv> --class_label <phenotype IBD or bmi> -u <0 for not running classification for augmented data> --num_clusters 5
```

### Balancing (TADA-SV and TADA-TVSV-m)
We could use TADA for balancing unbalance datasets. For the experiments we have done in the paper, which are called E2, E2-max, and E3, we used following commands to run TADA, classification, and SMOTE pipelines.

#### E2

To generate new samples for E2, we use the following command. This command will split the data into folds (let's say 5-folds). Let's assume we have two classes, with sizes 200 and 500. First it randomly selects a subset of larger class (in size), so that we have the same number of samples from both classes (both are 200). Then it splits the remaining into k-folds (e.g. 5-folds, each fold has 80 samples, 40 from each class). Then it keeps one part from smaller-size class and k-1 parts from larger-size class, and move the rest for validation (i.e. 40 from class1 and 160 from class2 in training). 
  
```
python run_augmentation.py -t <phylogeny.tre> -b <feature-table.biom> -s <dna-fragments.fasta> -m <meta_data.csv> -o <output_directory> -p <options.csv> -c <phenotype> -e unbalance -i <fold-number> -f <numbder of folds> -r 1 --seed <seed_number> --split_direction 0 > <log> 2>&1
```

To run classification, such that the testing ratio matches the one we used for training (e.g. 20 to 80 for the above example), we use a threshold (passed with __-t__, e.g. 0.2 for above example) as the split ratio between classes in test data.  

```
python run_classification_balances.py -z <input_augmentation.tar> -o <output_directory> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42> -t <The split ratio between classes in test data> -c 1 -e unbalance --meta <meta_data.csv> --class_label <phenotype IBD or bmi>
```

To run SMOTE (and classification), we use the following command:

```
python run_classification_balance_by_orig_data.py -z <input_augmentation.tar> -o <output_directory> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42> -t <The split ratio between classes in test data> -c 1 -e unbalance --meta <meta_data.csv> --class_label <phenotype IBD or bmi> -u 0 --num_clusters 5
```

#### E2-max

To generate new samples for E2-max, we use the following command (using __--split\_direction 1__). This command will split the data into training and testing in a fashion that maximum possible size of training data retained. The training ratio (passed with __--training\_ratio__) defines the ratio between less frequent class and most freq class (e.g. 0.1, or 0.2 etc.) in the training set. Note that in this scenario, we first split data for k-folds cross validation (we used 5-folds), and then keep k-1 folds for the training, and move 1 fold for test. Then we impose the ratio on top of the retained training samples (to have for example 10% from one class and 90% from the other class). 

```
python run_augmentation.py -t <phylogeny.tre> -b <feature-table.biom> -s <dna-fragments.fasta> -m <meta_data.csv> -o <output_directory> -p <options.csv> -c <phenotype> -e unbalance -i <fold-number> -f <numbder of folds> -r 1 --seed <seed_number> --split_direction 1 --training_ratio <unbalancedness ratio> > <log> 2>&1
```

To run classification, such that the testing ratio matches the one we used for training (e.g. 20 to 80 for the above example), we use a threshold (passed with __-t__, e.g. 0.2 for above example) as the split ratio between classes in test data.  

```
python run_classification_balances.py -z <input_augmentation.tar> -o <output_directory> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42> -t <The split ratio between classes in test data> -c 1 -e unbalance --meta <meta_data.csv> --class_label <phenotype IBD or bmi>
```

To run SMOTE (and classification), we use the following command:

```
python run_classification_balance_by_orig_data.py -z <input_augmentation.tar> -o <output_directory> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42> -t <The split ratio between classes in test data> -c 1 -e unbalance --meta <meta_data.csv> --class_label <phenotype IBD or bmi> -u 0 --num_clusters 5
```

#### E3
To run classification, such that the testing ratio doesn't matches the one we used for training (e.g. 20 to 80 for the above example), we use a threshold (passed with __-t__, e.g. 0.2 for above example). If the most frequent class in training should be the most frequent class in testing we use __-c 1__ otherwise use __-c 0__. For example let's say that you have an unbalance ratio of 20-80 in the training, and you want the same ratio in test. Then use ``` -t 0.2 -c 1```. Now, assume you want to impose bias in testing data (80-20), then use the code with ``` -t 0.2 -c 0```. 


```
python run_classification_balances.py -z <input_augmentation.tar> -o <output_directory> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42> -t <The split ratio between classes in test data> -c 0 -e unbalance --meta <meta_data.csv> --class_label <phenotype IBD or bmi>
```
To run SMOTE (and classification), we use the following command:

```
python run_classification_balance_by_orig_data.py -z <input_augmentation.tar> -o <output_directory> -p <label> -m <method nn or rf> -n <number of CPUs> -d <seed_number set to 42> -t <The split ratio between classes in test data> -c 0 -e unbalance --meta <meta_data.csv> --class_label <phenotype IBD or bmi> -u 0 --num_clusters 5
```

### Figures and R scripts in the paper 
The R scirpts to generate the figures we produced in the paper are avaialble [here](R/).

### Datasets

* The AGP dataset is available here [AGP](????), and the generated data are avaialble [TADA-AGP](????).
* The Gevers dataset is available here [Gevers](????), and the generated data are avaialble [TADA-Gevers](????).
* The option files are available here [options](????).

## References
1.  Junier, T. and Zdobnov, E.M., 2010. The Newick utilities: high-throughput phylogenetic tree processing in the UNIX shell. Bioinformatics, 26(13), pp.1669-1670.
2. Mirarab, S., Nguyen, N. and Warnow, T., 2012. SEPP: SATé-enabled phylogenetic placement. In Biocomputing 2012 (pp. 247-258).
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825-2830.
4. Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37
5. Travis E, Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).