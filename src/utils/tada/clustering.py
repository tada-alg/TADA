from __future__ import division
import pandas as pd
from deicode.optspace import OptSpace
from deicode.preprocessing import rclr
from skbio import DistanceMatrix
from biom import load_table
from scipy.spatial import distance
from sklearn.cluster import KMeans
import sys
import copy
import os
import tempfile
import logger
import shutil
from time import time
import numpy as np
from scipy.spatial.distance import pdist
from biom.util import biom_open

if 'TMPDIR' in os.environ:
	tmpdir = os.environ['TMPDIR']
else:
	tmpdir = tempfile.gettempdir()

def select_biom_table(samples, table):
	return table.filter(ids_to_keep=samples, axis='sample', inplace=False)

def select_samples_meta_data(samples, meta):
	return meta.loc[meta.index.isin(samples)]


def split_meta_data(meta, table, column):
	levels = meta[column].unique()
	meta_lst = list()
	biom_lst = list()
	label_strs = list()
	for level in levels:
		meta_tmp = meta.loc[meta[column] == level]
		meta_tmp['classes_within_clusters'] = meta_tmp['class_labels']
		meta_lst.append(meta_tmp)
		samples = meta_tmp.index
		biom_lst.append(select_biom_table(samples, table))
		label_strs.append('clustering_level-' + str(level))
	return meta_lst, biom_lst, label_strs

def get_distance_matrix(otutabledf, index, method, rank, logger_ins, training_index):

	if method == "deicode":
		logger_ins.info("method is deicode, and the rank is", rank)
		rclr_obj = rclr()
		table_norm = rclr_obj.fit_transform(copy.deepcopy(otutabledf))
		opt = OptSpace(rank=rank,iteration=10,tol=1e-5).fit(table_norm)
		sample_loading = pd.DataFrame(opt.sample_weights, index=index)
		rapca = DistanceMatrix(distance.cdist(sample_loading.values, sample_loading.values, 'euclidean'))

	else:
		table_norm = np.asarray(otutabledf / np.sum(otutabledf, 1))
		logger_ins.info("Using the braycurtis way of calculating distances between samples")
		logger_ins.info("the shape of the normalized table is", table_norm.shape)
		logger_ins.info("The sum of columns of the table is", set(list(table_norm.sum(1).squeeze())), "and the size is", table_norm.sum(1).shape)
		sample_loading = pd.DataFrame(table_norm, index=index)
		rapca = DistanceMatrix(pdist(sample_loading.values, 'braycurtis'))
	return rapca

def get_new_df(final_column, class_labels, index, cluster_labels, table):
	clustering_df = pd.DataFrame(np.column_stack([cluster_labels, class_labels]), index=index,
								 columns=['cluster_labels', 'class_labels'])
	clustering_df.index.name = '#SampleID'
	clustering_df['clusters_and_class_labels'] = "cluster-label-" + clustering_df['cluster_labels'].map(str) + "-class_label-" + \
								 clustering_df['class_labels'].map(str)

	clustering_df = map_string_class_labels_to_numerical(clustering_df, 'clusters_and_class_labels')
	clustering_df = map_string_class_labels_to_numerical(clustering_df, 'cluster_labels')

	if final_column == 'classes_within_clusters':
		meta_lst, biom_lst, label_strs = split_meta_data(clustering_df, table, 'cluster_labels')
		final_column = 'class_labels'
	else:
		meta_lst = [clustering_df]
		biom_lst = [table]
		label_strs = [final_column]
	if final_column ==  'clusters_and_class_labels':
		final_column = 'int_clusters_and_class_labels'
	if final_column == 'cluster_labels':
		final_column = 'int_cluster_labels'
	return meta_lst, biom_lst, clustering_df, label_strs, final_column

def map_string_class_labels_to_numerical(meta, column_label):
	mapping_dct = dict()
	c = 0
	for label in set(meta[column_label].values):
		mapping_dct[label] = c
		c += 1
	mapped_labels = list()
	for label in meta[column_label].values:
		mapped_labels.append(mapping_dct[label])
	meta["int_" + column_label] = np.asarray(mapped_labels)
	return meta

def clustering(X, num_cls, method, rank, class_labels,  index, table, out_fp=None, final_column='class_labels', logger_obj=None,training_index=None):
	'''
	:param X:
	:param num_cls:
	:param method:
	:param rank:
	:param class_labels:
	:param index:
	:param table:
	:param out_fp:
	:param final_column:  could be cluster_labels (make all clusters the same size), class_labels (make all class labels the same size),
	clusters_and_class_labels (make all clusters and classes the same size), classes_within_clusters (make class labels the same in the clusters)
	:return:
	'''
	t1 = time()
	if logger_obj is None:
		log_fp = tempfile.mktemp(dir=tmpdir)
		logger_obj = logger.LOG(log_fp)
		logger_ins = logger_obj.get_logger('CLUSTERING')
	else:
		logger_ins = logger_obj.get_logger('CLUSTERING')
	otutabledf = X
	index = index

	logger_ins.info("The size of the input data is", otutabledf.shape)
	logger_ins.info("The method to do clustering is", method)


	if final_column is not 'class_labels':
		rapca = get_distance_matrix(otutabledf, index, method, rank, logger_ins, training_index)

		t0 = time()
		kmeans = KMeans(n_clusters=num_cls)
		train_distance = rapca.data[np.ix_(training_index, training_index)]
		kmeans.fit(train_distance)
		cluster_labels = kmeans.predict(rapca.data[:, training_index])
		meta_lst, biom_lst, clustering_df, label_strs, final_column = get_new_df(final_column, class_labels, index, cluster_labels, table)
		logger_ins.info("Training of the kmean clustering finished in", time() - t0, "seconds")
		write_stats(clustering_df, logger_ins, 'clusters_and_class_labels')
		write_stats(clustering_df, logger_ins, 'class_labels')
		write_stats(clustering_df, logger_ins, 'cluster_labels')

	else:
		clustering_df = pd.DataFrame(class_labels, index=index, columns=['class_labels'])
		clustering_df.index.name = '#SampleID'
		meta_lst = [clustering_df]
		biom_lst = [table]
		label_strs = ['only_class_labels']
		kmeans = None

	logger_ins.info("The final classification column will be", final_column)

	write_df_to_file(clustering_df, out_fp)
	logger_ins.info("Clustering finished in", time()-t1, "seconds")
	return kmeans, meta_lst, biom_lst, label_strs, final_column

def write_stats(df, logger_ins, column):
	freq_dct = dict()
	for k in df[column]:
		freq_dct[k] = freq_dct.get(k, 0) + 1
	logger_ins.info(column, freq_dct)

def write_df_to_file(df, out_fp):
	if out_fp is not None:
		df.to_csv(out_fp, sep='\t')


if __name__ == "__main__":
	if len(sys.argv[1:]) < 7 or "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
		print("USAGE: [biom table] [number of clusters] [rank] [meta file path] [class_label column] [method] [output filepath]")
		sys.exit()
	table_fp = sys.argv[1]
	num_clusters = int(sys.argv[2])
	rank = int(sys.argv[3])
	meta_fp = sys.argv[4]
	label_col = sys.argv[5]
	method = sys.argv[6]
	out_fp = sys.argv[7]

	table = load_table(table_fp)
	meta = pd.read_csv(meta_fp,  sep='\t', low_memory=False, dtype={'#SampleID': str}, index_col='#SampleID')
	meta = meta.reindex(table.ids('sample'))
	labels = meta[label_col]
	otuX = table.matrix_data.transpose().todense()

	kmeans, meta_lst, biom_lst,label_strs, final_column = clustering(otuX, num_clusters, method, rank, labels, meta.index.values, table, out_fp=out_fp)
