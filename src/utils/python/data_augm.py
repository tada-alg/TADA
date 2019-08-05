import dendropy
import biom
from os import path
import unittest
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from read_parameters_augm import *
from time import time
import os
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

def isNotEmpty(s):
	return bool(s and s.strip())
def read_split_file(split_fp):
	split_lst = list()
	if split_fp is not None and os.path.exists(split_fp):
		with open(split_fp) as f:
			tmp_dct = json.load(f)
		for key in tmp_dct.keys():
			test_lst = list(map(int, tmp_dct[key]['TEST'].split()))
			train_lst = list(map(int, tmp_dct[key]['TRAIN'].split()))
			tmp = {'TRAIN': train_lst, 'TEST': test_lst}
			split_lst.append(tmp)
	else:
		split_lst = None
	return split_lst


class Data():
	def __init__(self, table_fp, tree_fp, fragmentary_fp, meta_fp, param_fp, out_dir, splits, split_id, class_label, logger=None, freq_classes=None, most_freq_class=None, main_class_label=None, augmentation_method=None):

		param_pre = read_params(param_fp, out_dir)
		self.param_pre = param_pre
		self.__set_atributes(param_pre)
		if logger is None:
			self.logger_ins = param_pre['logger'].get_logger('generator_' + str(split_id))
		else:
			self.logger_ins = logger.get_logger('generator_' + str(split_id))

		self.logger_ins.info("The seed number is set to", self.seed_num)

		self.class_label = class_label
		if main_class_label is not None:
			self.main_class_label = main_class_label
		else:
			self.main_class_label = self.class_label

		self.fragmentary_fp = fragmentary_fp
		self.table_fp = table_fp
		self.idx = split_id
		self.freq_classes = freq_classes

		self.most_freq_class = most_freq_class
		if augmentation_method is not None:
			self.generate_strategy = augmentation_method
		if self.generate_strategy == "individual":
			self.most_freq_class = None
			self.freq_classes = None
		self.meta_fp = meta_fp
		self.splits = splits
		self.final_output_path = out_dir
		self.split_idx =  split_id

		self.dirpath = self.tmp_dir

		self.__set_num_to_select(param_pre)
		self.__set_splits()


		self.fragmentaries_dict = dict()

		self.basename = path.basename(self.table_fp).replace(".biom", "")
		self.logger_ins.info("will write outputs on", self.dirpath + "/" + self.basename)


		t0 = time()
		self._load_table()
		self.logger_ins.info("biom table loaded in", time() - t0, "seconds")
		self.logger_ins.info("Number of sOTUs is ", len(self.obs))
		self.logger_ins.info("Number of samples is ", len(self.sample))

		t0 = time()
		self._load_fasta()
		self.logger_ins.info("fasta file loaded in", time() - t0, "seconds")

		t0 = time()
		self.__read_meta_fp()
		self.logger_ins.info("Meta data loaded in", time() - t0, "seconds")

		t0 = time()
		self._load_tree(tree_fp)
		self.logger_ins.info("loading the tree took", time() - t0, "seconds")
		self.__add_pseudo()


		t0 = time()
		self._comput_average_distances()
		self.logger_ins.info("computing the average tip-to-tip distances took", time()-t0, "seconds")
		self._init_gen()


	def __set_atributes(self, param_pre):
		for key in param_pre:
			setattr(self,key,param_pre[key])

	def __add_pseudo(self):
		self.pseudo_cnt /= self.num_leaves
		self.table += self.pseudo_cnt
		if self.test_samples is not None:
			self.table_ts += self.pseudo_cnt

	def __set_splits(self):
		if self.splits is not None:
			self.selected_samples = sorted(self.splits['TRAIN'])
			self.test_samples = sorted(self.splits['TEST'])
			self.logger_ins.info(self.selected_samples, self.test_samples)
		else:
			self.selected_samples = 'all'
			self.test_samples = None

	def __set_num_to_select(self, param_pre):
		if self.most_freq_class is None or self.freq_classes is None:
			self.n_binom_extra = {'all': 0}
		else:
			self.n_binom_extra = dict()
			self.n_samples_to_select = dict()
			for c in self.freq_classes:
				n_all = self.most_freq_class[0] * (param_pre['xgen'])
				if n_all + self.most_freq_class[0] - self.freq_classes[c] < 0:
					print("The n_all is negative!", n_all + self.most_freq_class[0] - self.freq_classes[c], c, n_all,
						  self.most_freq_class[0], self.freq_classes[c])
					exit()
				n_total_samples_to_gen = np.max(n_all + self.most_freq_class[0] - self.freq_classes[c], 0)
				self.n_samples_to_select[c] = np.max(
					(n_total_samples_to_gen) // (param_pre["n_binom"] * param_pre["n_beta"]), 0)

				self.n_binom_extra[c] = max(
					n_total_samples_to_gen - self.n_samples_to_select[c] * param_pre["n_binom"] * param_pre["n_beta"],
					0)

				self.logger_ins.info("class", c, "Generating", self.n_samples_to_select[c], "samples", "with",
									 "n_binom:",
									 param_pre["n_binom"], "and n_beta:", param_pre["n_beta"],
									 "and n_binom_extra", self.n_binom_extra[c])

	def __read_meta_fp(self):
		meta = pd.read_csv(self.meta_fp, sep='\t', index_col='#SampleID', low_memory=False,
							   dtype={'#SampleID': str})

		self.meta, self.labels, self.classes, le = self.__set_meta_clusters(meta, self.sample, None, 'training')

		if self.test_samples is not None:
			self.meta_ts, self.labels_ts, _, _ = self.__set_meta_clusters(meta, self.samples_ts, le, 'test')
		else:
			self.meta_ts = None
		self.sample = np.asarray(self.sample)
		self.__set_clusters()


	def __set_meta_clusters(self, meta, samples, le, label):
		meta = meta.loc[meta.index.isin(samples)]
		meta = meta.reindex(samples)
		meta[self.class_label] = meta[self.class_label].fillna(-1)

		if self.main_class_label is not None:
			meta[self.main_class_label] = meta[self.main_class_label].fillna(-1)

		labels = meta[self.class_label]
		classes, counts = np.unique(labels, return_counts=True)


		self.logger_ins.info("possible class labels are", classes)
		for i in range(len(classes)):
			self.logger_ins.info("The number of samples from the class", classes[i], "in the", label, "is", counts[i])

		return meta, labels, classes, le


	def __set_clusters(self):
		self.clusters = dict()
		for cls in self.classes:
			if pd.isnull(cls):
				self.clusters['NaN'] = self.sample[self.meta[self.class_label].isnull()]
				self.logger_ins.info("The number of samples with label", 'NaN', "is", len(self.clusters['NaN']))

			else:
				self.clusters[cls] = self.sample[self.labels == cls]
				self.logger_ins.info("The number of samples with label", cls, "is", len(self.clusters[cls]))

	def _init_gen(self):
		self.augmented_samples = list()
		self.orig_samples = self.table[:, self.sample_argsort].transpose()
		if self.test_samples is not None:
			self.orig_samples_ts = self.table_ts[:, self.sample_ts_argsort].transpose()

	def _load_table(self):
		'''
		loads the biom table, removes unseen observation, and normalizes it, and saves the observations and samples
		:return:
		'''

		table = biom.load_table(self.table_fp)
		self.logger_ins.info("size of table before filtering", table.shape)

		# filter_fn = lambda val, id_, md: val.sum() > 0
		# table.filter(axis='observation', ids_to_keep=filter_fn, inplace=True)

		if self.selected_samples != 'all' and self.test_samples is not None:
			self.table, self.sample_argsort, self.sample = self.__filter_table(table, self.selected_samples, 'training')
			self.table_ts, self.sample_ts_argsort, self.samples_ts = self.__filter_table(table, self.test_samples, 'test')
		else:
			self.table, self.sample_argsort, self.sample = self.__filter_table(table, table.ids('sample'), 'training')

		self.obs = set(self.table.ids('observation'))

		# To keep track of index of samples and sequences in the biom table
		self.biom_table = self.table
		self._sample_obs_dct()
		self.table = self.table.matrix_data.todense()
		self.table_ts = self.table_ts.matrix_data.todense()

		return


	def __filter_table(self, table, samples, label):
		new_table = table.filter(ids_to_keep=samples, axis='sample', inplace=False)
		sample_argsort = np.argsort(new_table.ids('sample'))
		samples_final = np.asarray(sorted(new_table.ids('sample')))
		self.logger_ins.info("The size of the", label, "table after selecting samples", new_table.shape)
		return new_table, sample_argsort, samples_final


	def _sample_obs_dct(self):
		'''
		 To keep track of index of samples and sequences in the biom table, we create two dictionaries
		 self.obs_dct: key: sequences, values: index
		 self.sample_dct: key: sample, values: index
		:return:
		'''
		sequences = self.table.ids('observation')
		samples = self.table.ids('sample')
		self.obs_dct = dict()
		for i, seq in enumerate(sequences):
			self.obs_dct[seq] = i

		self.sample_dct = dict()
		for i, sample in enumerate(samples):
			self.sample_dct[sample] = i
		return

	def _load_fasta(self):
		'''
		Builds a dictionary. Keys: species names, values: sequences
		:return:
		'''
		fragmentaries = open(self.fragmentary_fp).readlines()

		frg_nm = ""
		for frg in fragmentaries:
			frg = frg.strip("\n")
			if re.match(">", frg) is not None:
				frg = frg.strip(">")
				frg_nm = frg
			elif isNotEmpty(frg):
				if frg_nm not in self.fragmentaries_dict:
					self.fragmentaries_dict[frg_nm] = frg
				else:
					self.fragmentaries_dict[frg_nm] += frg
		return

	def _load_tree(self, tree_fp):

		self.tree = dendropy.Tree.get(path=tree_fp, preserve_underscores=True, schema="newick",
									  rooting='default-rooted')
		self.tree.resolve_polytomies()
		c = 0
		for nd in self.tree.postorder_node_iter():
			if nd == self.tree.seed_node and nd.label is None:
				nd.label = "root"
			if nd.is_leaf():
				nd.label = nd.taxon.label
			else:
				nd.label = "internal_node_" + str(c)
			c += 1

		self.num_leaves = len(self.tree.leaf_nodes())

		return

	def _comput_average_distances(self):
		'''
		:param tree: The phylogeny
		:return: Computes the average tip to tip distances bellow for each clade of the tree in one traverse of the trees
		'''
		for nd in self.tree.postorder_node_iter():
			if nd.is_leaf():
				nd.num = 1
				nd.avg = 0.0
				nd.sum_dist = 0.0
				nd.num_pairs = 1
			else:
				child_nodes = nd.child_nodes()
				nd.num_pairs = child_nodes[0].num * child_nodes[1].num
				total_dist = child_nodes[1].sum_dist * child_nodes[0].num + child_nodes[0].sum_dist * child_nodes[
					1].num + \
							 (child_nodes[1].edge.length + child_nodes[0].edge.length) * (nd.num_pairs)
				nd.avg = total_dist / (nd.num_pairs)

				nd.sum_dist = child_nodes[0].sum_dist + child_nodes[1].sum_dist + \
							  child_nodes[0].edge.length * child_nodes[0].num + \
							  child_nodes[1].edge.length * child_nodes[1].num

				nd.num = child_nodes[0].num + child_nodes[1].num
		return

if __name__ == '__main__':
	unittest.main()
