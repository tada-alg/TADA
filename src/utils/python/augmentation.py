import re
from time import time
from numpy import random
import tarfile
from data_augm import Data
import os
import numpy as np
from copy import deepcopy
import unittest
import shutil
import scipy
from scipy.stats import beta as beta_t

def make_tarfile(output_filename, source_dir):
	with tarfile.open(output_filename, "w:gz") as tar:
		tar.add(source_dir, arcname=os.path.basename(source_dir))

class SampleGenerator(Data, unittest.TestCase):

	def __init__(self, table_fp, tree_fp, fragmentary_fp, meta_fp, param_fp, out_dir, splits, split_id, class_label, logger=None, freq_classes=None, most_freq_class=None, main_class_label=None, augmentation_method=None, seed_num=0):
		Data.__init__(self, table_fp, tree_fp, fragmentary_fp, meta_fp, param_fp, out_dir, splits, split_id, class_label, logger=logger,
					  freq_classes=freq_classes, most_freq_class=most_freq_class, main_class_label=main_class_label)
		np.random.seed(seed_num)
		self.logger_ins.info("The seed number inside the sample generatory is set to", seed_num)

		print("The temp directory is", self.tmp_dir)

	def _load_counts_on_tree_class(self, class_id):
		samples = self.clusters[class_id]
		self.logger_ins.info(samples)
		smp_idxs = np.asarray([self.sample_dct[sample] for sample in samples])

		t0 = time()
		for nd in self.tree.postorder_node_iter():
			if nd.is_leaf():
				seq = self.fragmentaries_dict[nd.taxon.label]
				if seq in self.obs_dct:
					seq_idx = self.obs_dct[seq]

					if hasattr(nd, 'freq_class'):
						nd.freq_class[class_id] = self.table[seq_idx, smp_idxs].reshape((-1,1))
					else:
						nd.freq_class = dict()
						nd.freq_class[class_id] = self.table[seq_idx, smp_idxs].reshape((-1,1))
				else:
					if hasattr(nd, 'freq_class'):
						nd.freq_class[class_id] = np.zeros((len(smp_idxs),1))
					else:
						nd.freq_class = dict()
						nd.freq_class[class_id] = np.zeros((len(smp_idxs),1))
			else:
				child_nodes = nd.child_nodes()
				if hasattr(nd, 'freq_class'):
					nd.freq_class[class_id] = child_nodes[0].freq_class[class_id] + child_nodes[1].freq_class[class_id]
					nd.mu[class_id] = np.mean(child_nodes[0].freq_class[class_id])
					nd.S[class_id] = np.mean(np.power(child_nodes[0].freq_class[class_id], 2))
					nd.n1[class_id] = np.mean(nd.freq_class[class_id])
					nd.n2[class_id] = np.mean(np.power(nd.freq_class[class_id],2))
					nd.p[class_id] = nd.mu[class_id] / nd.n1[class_id]
					nd.probs[class_id] = child_nodes[0].freq_class[class_id]/nd.freq_class[class_id]
				else:
					nd.freq_class = dict()
					nd.mu = dict()
					nd.S = dict()
					nd.n1 = dict()
					nd.n2 = dict()
					nd.p = dict()
					nd.probs = dict()
					nd.freq_class[class_id] = child_nodes[0].freq_class[class_id] + child_nodes[1].freq_class[class_id]
					nd.mu[class_id] = np.mean(child_nodes[0].freq_class[class_id])
					nd.S[class_id] = np.mean(np.power(child_nodes[0].freq_class[class_id], 2))
					nd.n1[class_id] = np.mean(nd.freq_class[class_id])
					nd.n2[class_id] = np.mean(np.power(nd.freq_class[class_id], 2))
					nd.p[class_id] = nd.mu[class_id]/nd.n1[class_id]
					nd.probs[class_id] = child_nodes[0].freq_class[class_id] / nd.freq_class[class_id]


		self.logger_ins.info("traversing the tree for class", class_id, "Finished in", time() - t0, "seconds")
		return

	def _load_counts_on_tree_individual(self, sample_ids, class_id):
		samples = sample_ids
		smp_idxs = np.asarray([self.sample_dct[sample] for sample in samples])
		# tmp_samples = self.biom_table.ids('sample')
		# s = self.meta.loc[self.meta.index==sample_ids[0]]
		# tmp_class = s[self.class_label]
		# self.logger_ins.info("Testing if the sample is in the cluster:", class_id, tmp_class.values[0], smp_idxs[0], tmp_samples[smp_idxs[0]] == samples[0], samples[0] in self.clusters[class_id], (self.meta.index==tmp_samples[self.sample_argsort]).sum())
		tree = self.tree
		for nd in tree.postorder_node_iter():
			'''
			features_edge_pr: num samples x num features 
			self.edge_map: keys: edge_num values: index of them (for features), note that edge_num is fixed across different runs
			rev_placement_supports: keys: edge_num, values: list of tuples (sequence (not species name), likelihood, pendent_edge_length, index (0 means maximum likelihood postion))
			'''
			if nd.is_leaf():
				seq = self.fragmentaries_dict[nd.taxon.label]
				if seq in self.obs_dct:
					seq_idx = self.obs_dct[seq]

					nd.freq = self.table[seq_idx, smp_idxs]
					nd.prior = np.sum(nd.freq_class[class_id])
				else:
					nd.freq = 0
					nd.prior = np.sum(nd.freq_class[class_id])

			else:
				child_nodes = nd.child_nodes()
				nd.freq = child_nodes[0].freq + child_nodes[1].freq
				nd.prior = np.sum(nd.freq_class[class_id])
				if nd.prior - child_nodes[0].prior < nd.freq - child_nodes[0].freq:
					self.logger_ins.info("The length of the class_id", class_id, "is", len(self.clusters[class_id]), "And the size of the freq_class is", nd.freq_class[class_id].shape)
					self.logger_ins.info("The class id is", class_id, "smp_idxs:", smp_idxs, "sample_ids", sample_ids)
					self.logger_ins.info( "The label is", nd.label, "The prior of the node is", nd.prior, "The freq  for node is", nd.freq, "Child0 label:",
									 child_nodes[0].label, "prior child 0:", child_nodes[0].prior, "freq child 0:", child_nodes[0].freq, sample_ids, self.table[:, smp_idxs])
					exit()

		return tree

	def get_method_of_moments_estimates(self, n1, n2, S, mu, p):
		if self.stat_method != 'beta':
			alpha_mom = (n2 * mu - n1 * S)/((S/mu - 1)*np.power(n1,2) + mu * (n1-n2))
			beta_mom = (S/mu * n1 - n2) * (mu - n1 ) / ((S/mu - 1)*np.power(n1,2) + mu * (n1-n2))
			# if alpha_mom < 0 or beta_mom < 0:
				# self.logger_ins.info("WARNING: The alpha and beta estimates from method of moments are negative!", "n1:",
				# 				n1, "n2:", n2, "S:", S, "mu:", mu, "alpha_mom:", alpha_mom, "beta_mom:", beta_mom)
		else:
			alpha_mom, beta_mom = self._infer_alpha_and_beta_beta_distribution(p)

		return alpha_mom, beta_mom
	def _infer_alpha_and_beta_beta_distribution(self, p):
		mu = np.mean(p)
		std = np.var(p)

		if std == 0:
			alpha_mom = -1
			beta_mom = -1
		else:
			M = mu * (1 - mu)
			alpha_mom = mu * (M / std - 1)
			beta_mom = (1 - mu) * (M / std - 1)
			# tx = time()
			# if std >= M:
			# 	alpha_mom, beta_mom, _, _ = beta_t.fit(p)
			# 	self.logger_ins.info("Warning, alhpa_mom and beta_mom where negative using exact method, it took",
			# 						 time() - tx, "seconds")

		return alpha_mom, beta_mom

	def _infer_alpha_and_beta(self, frc0, prc0, nd, class_id):
		if self.var_method == 'br_penalized' and self.stat_method != "binom":
			p = np.mean(frc0 * (1 - self.prior_weight) + prc0 * self.prior_weight)
			a = p * np.power(nd.avg + self.pseudo, self.exponent) * self.coef
			b = (1 - p) * np.power(nd.avg + self.pseudo, self.exponent) * self.coef
		elif  self.stat_method == "binom":
			a = -1
			b = -1
		else:
			a, b = self.get_method_of_moments_estimates(nd.n1[class_id], nd.n2[class_id], nd.S[class_id], nd.mu[class_id], nd.probs[class_id])
		return a, b


	def _gen_beta(self, a, b, n, Pr):
		t1 = time()
		if a == 0 and b > 0:
			x = np.zeros((n, 1))
		elif b == 0 and a > 0:
			x = np.ones((n, 1))
		elif a > 0 and b > 0:
			x = random.beta(a, b, size=n)
		else:
			p = [Pr]
			x = np.asarray(p * n)
		return x.ravel()

	def _gen_binomial(self, p, nd, n_binom):
		gen_x = list()
		n = nd.f
		for i in range(len(n)):
			i_p = i // n_binom
			if n[i] > 0 and p[i_p] > 0:
				gen_x.append(np.random.binomial(n[i], p[i_p], size=1))
			else:
				gen_x.append([0]*1)
		return np.asarray(gen_x).ravel()

	def _set_children_features(self, nd, child_l, child_r, Pr, n_binom):
		if self.stat_method == "beta_binom" or self.stat_method == "binom":
			gen_x = self._gen_binomial(Pr, nd, n_binom)
			child_l.f = gen_x
			child_r.f = nd.f - gen_x
		elif self.stat_method == "beta":
			child_l.f = Pr * nd.f
			child_r.f = (1 - Pr) * nd.f
		elif self.stat_method == 'constant':
			child_l.f = np.asarray([child_l.freq] * len(nd.f)).ravel()
			child_r.f = np.asarray([child_r.freq] * len(nd.f)).ravel()

	def _gen_augmented_sample(self, tree, n_binom, n_beta, class_id):
		t1 = time()
		if n_binom == 0:
			return
		n_generate = n_binom * n_beta
		to_be_generated = np.zeros((n_generate, len(self.obs_dct)))
		for nd in tree.preorder_node_iter():
			if nd == tree.seed_node:
				nd.f = np.asarray([int(np.mean(nd.freq))] * n_binom * n_beta).ravel() if self.stat_method != "beta" else np.ones((n_binom * n_beta,))

			if nd.is_leaf():
				seq = self.fragmentaries_dict[nd.taxon.label]
				if seq in self.obs_dct:
					seq_idx = self.obs_dct[seq]
					to_be_generated[:, seq_idx] = np.asarray(nd.f).ravel()
			else:
				child_nodes = nd.child_nodes()
				frc0 = child_nodes[0].freq / (child_nodes[0].freq + child_nodes[1].freq) if child_nodes[0].freq + child_nodes[1].freq > 0 else 0
				prc0 = child_nodes[0].prior / (child_nodes[0].prior + child_nodes[1].prior) if child_nodes[0].prior + child_nodes[1].prior > 0 else 0

				pr_l = frc0 * (1 - self.prior_weight) + prc0 * self.prior_weight if (
						self.var_method == 'br_penalized') else nd.p[class_id]

				a, b = self._infer_alpha_and_beta(frc0, prc0, nd, class_id)
				# self.logger_ins.info("label", nd.label, "a:", a, "b:", b, "nd.freq:", nd.freq, "child[0].freq:", child_nodes[0].freq, "frc0", frc0, "prc0", prc0, "pr_l:", pr_l, "nd.prior", nd.prior, "child_nodes[0].prior", child_nodes[0].prior)

				x = self._gen_beta(a, b, n_beta, pr_l)
				self._set_children_features(nd, child_nodes[0], child_nodes[1], x, n_binom)

		# self.logger_ins.info(to_be_generated.sum(1))
		self.logger_ins.info("Time to add new samples", time() - t1, len(to_be_generated))
		self.augmented_samples.append(to_be_generated)

	def __load_and_generate(self, samples, label, n_augmentation, n_beta):

		t0 = time()

		tree = self._load_counts_on_tree_individual(samples, label)
		self.logger_ins.info("It took", time() - t0, "seconds to load counts on tree for this individual")
		t1 = time()
		self._gen_augmented_sample(tree, n_augmentation, n_beta, label)
		
		self.samples_generated += [samples[0]] * n_augmentation * n_beta
		self.labels_generated += [label] * n_augmentation * n_beta
		self.logger_ins.info("Time to add info of this sample", time()-t1)
		self.logger_ins.info("Generating", n_augmentation * n_beta, "samples for sample", samples, "finished in",
						 time() - t0, "seconds")
		return

	def traverse_tree(self):
		'''
		Traverse the tree and computes features for each sample
		:return:
		'''
		for class_id in self.clusters.keys():
			self._load_counts_on_tree_class(class_id)
		self.samples_generated = list()
		self.labels_generated = list()
		n_binom = self.param_pre['n_binom']
		n_beta = self.param_pre['n_beta']
		if self.generate_strategy == "individual" and self.var_method == "br_penalized":
			for i, _ in enumerate(self.sample):
				self.__load_and_generate([self.sample[i]], self.labels[i], n_binom, n_beta)

		elif self.generate_strategy == "individual" and self.var_method != "br_penalized":
			for cls in self.clusters:
				samples = self.clusters[cls]
				for _, sample in enumerate(samples):
					self.__load_and_generate([sample], cls, n_binom, n_beta)

		elif self.generate_strategy == 'balancing' and self.var_method == "br_penalized":
			for cls in self.clusters:
				samples = self.clusters[cls]
				if self.n_samples_to_select[cls] > 0:
					for sample in np.random.choice(samples, size=self.n_samples_to_select[cls], replace=True):
						self.__load_and_generate([sample], cls, n_binom, n_beta)
					sample = np.random.choice(samples,1)[0]
					if self.n_binom_extra[cls] > 0:
						self.__load_and_generate([sample], cls, self.n_binom_extra[cls], 1)
		else:
			for cls in self.clusters:
				samples = self.clusters[cls].squeeze()
				if self.n_samples_to_select[cls] > 0:
					for sample in np.random.choice(samples, size=self.n_samples_to_select[cls], replace=True):
						self.__load_and_generate([sample], cls, n_binom, n_beta)
					sample = np.random.choice(samples, 1)[0]
					if self.n_binom_extra[cls] > 0:
						self.__load_and_generate([sample], cls, self.n_binom_extra[cls], 1)



		self.augmented_samples = np.row_stack(self.augmented_samples)

		return

	def get_agum_data(self):
		feature = self.augmented_samples
		samples = np.asarray(self.samples_generated).reshape((len(self.samples_generated), -1))
		labels = np.asarray(self.labels_generated)
		otu_names = sorted(self.fragmentaries_dict.keys())
		seq_list = [self.fragmentaries_dict[x] for x in otu_names]
		new_seq_lst = list()
		idx_argsort = list()
		for seq in seq_list:
			# edge_map: key: edge_num, value: edge_idx
			if seq in self.obs_dct:
				i = self.obs_dct[seq]
				# save the order of features
				idx_argsort.append(i)
				new_seq_lst.append(seq)
		# reorganize the features
		data = feature[:, idx_argsort]
		np.asarray(data)
		data /= np.sum(data, axis=1, keepdims=True)
		return data, labels, new_seq_lst, samples.tolist()

	def __get_orig_data(self, feature, labels, samples):
		otu_names = sorted(self.fragmentaries_dict.keys())
		seq_list = [self.fragmentaries_dict[x] for x in otu_names]
		new_seq_list = list()
		idx_argsort = list()
		for seq in seq_list:
			# edge_map: key: edge_num, value: edge_idx
			if seq in self.obs_dct:
				i = self.obs_dct[seq]
				# save the order of features
				idx_argsort.append(i)
				new_seq_list.append(seq)

		# reorganize the features
		data = feature[:, idx_argsort]
		data = np.asarray(data)
		data -= self.pseudo_cnt
		data /= np.sum(data, axis=1, keepdims=True)
		return data, labels, new_seq_list, samples


	def get_orig_data_train(self):
		data_tr, labels_tr, _, sample_tr = self.__get_orig_data(self.orig_samples, self.labels, self.sample)
		return data_tr, labels_tr, sample_tr

	def get_orig_data_test(self):
		if self.test_samples is not None:
			data_ts, labels_ts, _, sample_ts = self.__get_orig_data(self.orig_samples_ts, self.labels_ts, self.samples_ts)
		else:
			data_ts = labels_ts = sample_ts = None
		return data_ts, labels_ts, sample_ts


	def _print_feature_on_file(self, file_path, meta):
		data, labels, seq_list, sample = self.get_agum_data()
		labels = self.get_labels(sample, meta)
		self.logger_ins.info("The size of features to be written on file_path is", data.shape)
		np.savez(re.sub(".csv", ".npz", file_path), data=data, index = sample, samples=sample, seq=seq_list, labels=labels)


	def _print_feature_on_file2(self, feature, labels, sample, file_path, meta):
		data, label, seq_list, samples = self.__get_orig_data(feature, labels, sample)
		self.logger_ins.info("The size of features to be written on file_path is", data.shape)
		label = self.get_labels(samples, meta)

		np.savez(re.sub(".csv", ".npz", file_path), data=data, samples=samples, index = samples, labels=label, seq = seq_list)

	def get_labels(self, samples, meta):
		main_class_label = self.main_class_label
		class_map = dict()
		for index, row in meta.iterrows():
			class_map[index] = row[main_class_label]

		labels = list()
		for sample in np.asarray(samples).squeeze():
			labels.append(class_map[sample])
		return labels
	def print_features_on_file(self, label_str=""):
		'''
		Write features on file
		'''
		# Get the name of feature, and replace it with feature name
		basename = self.basename
		dirp = self.dirpath
		if not os.path.isdir(dirp):
			os.mkdir(dirp)
		ft1 = dirp + "/" + basename + "_generated_counts_" + str(self.split_idx) + "_" + label_str + ".csv"
		ft2 = dirp + "/" + basename + "_orig_counts_train_" + str(self.split_idx)+ "_" + label_str + ".csv"
		ft3 = dirp + "/" + basename + "_orig_counts_test_" + str(self.split_idx) + "_" + label_str + ".csv"


		# delete unwanted files and dictionaries
		del self.tree
		del self.table

		# Write features on file, and delete them from memory
		self._print_feature_on_file(ft1, self.meta)
		self._print_feature_on_file2(self.orig_samples, self.labels, self.sample, ft2, self.meta)
		if self.test_samples is not None:
			self._print_feature_on_file2(self.orig_samples_ts, self.labels_ts, self.samples_ts, ft3, self.meta_ts)

		make_tarfile(self.final_output_path + '/outputs_' + str(self.split_idx) + "_" + label_str + '.tar', self.dirpath)
		shutil.rmtree(self.dirpath)






