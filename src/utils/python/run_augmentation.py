from sklearn.model_selection import StratifiedKFold
from augmentation import *
from shutil import copyfile
from clustering import *
from optparse import OptionParser
from sklearn.model_selection import ShuffleSplit
import pickle
import json
if 'TMPDIR' in os.environ:
	tmpdir = os.environ['TMPDIR']
else:
	tmpdir = tempfile.gettempdir()

from run_classification_balances import make_test_dataset

def make_freq_dict(meta, label_str):
	label_tr = meta[label_str].values
	freq_dct = dict()
	labels, counts = np.unique(label_tr, return_counts = True)
	max_found = 0
	min_found = np.inf
	for i in range(len(counts)):
		if  counts[i] > max_found:
			most_freq_class = (counts[i], labels[i])
			max_found = counts[i]
		if counts[i] < min_found:
			least_freq_class = (counts[i], labels[i])
			min_found = counts[i]
		freq_dct[labels[i]] = counts[i]
	return freq_dct, most_freq_class, least_freq_class

def clip_lower_class(meta, label_str):
	freq_dct, most_freq_class, least_freq_class =  make_freq_dict(meta, label_str)
	to_remove_len = int(least_freq_class[0])
	meta_least = meta.loc[meta[label_str] == least_freq_class[1]]
	meta_most = meta.loc[meta[label_str] == most_freq_class[1]]
	most_idxs = meta_most.index.values
	np.random.shuffle(most_idxs)
	meta_most_tr = meta_most.loc[meta_most.index.isin(most_idxs[:to_remove_len])]
	meta_most_ts = meta_most.loc[meta_most.index.isin(most_idxs[to_remove_len:])]
	meta_joined = pd.concat([meta_least, meta_most_tr])
	return most_freq_class, least_freq_class, to_remove_len, meta_joined, meta_most_ts


def get_indices_training_and_test(train_labels, test_labels, most_freq_class):
	print(most_freq_class)
	unique, counts = np.unique(train_labels, return_counts=True)
	mapping = dict()
	if int(unique[0]) is int(most_freq_class):
		mapping['train_most'] = np.isin(train_labels, unique[0])
		mapping['train_least'] = np.isin(train_labels, unique[1])
	else:
		mapping['train_most'] = np.isin(train_labels, unique[1])
		mapping['train_least'] = np.isin(train_labels, unique[0])

	unique, unique_indices, counts = np.unique(test_labels, return_counts=True, return_index=True)
	if int(unique[0]) is int(most_freq_class):
		mapping['test_most'] = np.isin(test_labels, unique[0])
		mapping['test_least'] = np.isin(test_labels, unique[1])
	else:
		mapping['test_most'] = np.isin(test_labels, unique[1])
		mapping['test_least'] = np.isin(test_labels, unique[0])
	return mapping

def swap_train_test_split(meta, meta_most_ts, label, train_index, test_index, train_samples, test_samples, cls, class_str):
	train_labels = label[train_index]
	test_labels = label[test_index]
	mapping = get_indices_training_and_test(train_labels, test_labels, cls)
	print(train_labels.shape, test_labels.shape)
	if meta_most_ts is not None:
		index_tr_most = train_samples[mapping['train_most']].values.reshape((-1,1))
		index_tr_least = test_samples[mapping['test_least']].values.reshape((-1,1))
	else:


		index_tr_most = train_samples[mapping['train_most']].values.reshape((-1,1))
		index_tr_least = train_samples[mapping['train_least']].values.reshape((-1,1))
	
	train_samples_f = np.row_stack([index_tr_most, index_tr_least])
	print(train_samples_f.shape)
	if meta_most_ts is not None:
		index_ts_most = test_samples[mapping['test_most']].values.reshape((-1,1))
		index_ts_least = train_samples[mapping['train_least']].values.reshape((-1,1))
	else:
		index_ts_most = test_samples[mapping['test_most']].values.reshape((-1,1))
		index_ts_least = test_samples[mapping['test_least']].values.reshape((-1,1))
	if meta_most_ts is not None:
		test_samples = np.row_stack([index_ts_most, index_ts_least, meta_most_ts.index.values.reshape((-1,1))])
	else:
		test_samples = np.row_stack([index_ts_most, index_ts_least])
	print(test_samples.shape)
	split_dct = {'TRAIN': train_samples_f.squeeze(), 'TEST': test_samples.squeeze()}
	meta_tr = meta.loc[meta.index.isin(train_samples_f.squeeze())]
	freq_dct, most_freq_class, least_freq_class = make_freq_dict(meta_tr, class_str)
	return split_dct, freq_dct, most_freq_class

def get_index_label_unbalance_augmentation(meta, class_str):
	most_freq_class, least_freq_class, to_remove_len, meta_joined, meta_most_ts = clip_lower_class(meta, class_str)
	index = meta_joined.index
	label = meta_joined[class_str]
	return index, label, most_freq_class, least_freq_class, to_remove_len, meta_joined, meta_most_ts

def run_experiment(experiment, meta_joined, meta_most_ts, label, train_index, test_index, train_samples,
				   test_samples, most_freq_class, class_str, meta_fp, biom_fp, logger_obj,
				   tree_fp, fasta_fp, param_fp, out_dir, otuX, num_clusters, method, rank, labels, meta, table,
				   final_column_clustering, logger_ins, idx, seed_num, training_ratio=1, split_direction=0):

	split_lst = list()
	if experiment == 'unbalance' or experiment == 'augmentation':
		if experiment == 'unbalance':

			if split_direction == 1:
				_, labels_train, train_samples_new = make_test_dataset(np.ones((len(train_samples),10)), meta_fp, class_str, train_samples,
														   labels[train_index], train_samples, labels[train_index], train_samples,
																logger_ins, training_ratio, 1)
				train_index = np.where(np.isin(train_samples, train_samples_new))[0]
			else:
				train_samples_new = train_samples

			split_dct, freq_dct, most_freq_class_t = swap_train_test_split(meta_joined, meta_most_ts, label, train_index,
																		   test_index, train_samples_new, test_samples,
																		   most_freq_class[1], class_str)
			logger_ins.info("The frequency of the classes in training before running the sample generator is", freq_dct)
			meta_tmp_fp = meta_fp
			biom_tmp_fp = biom_fp
		else:

			split_dct = {'TRAIN': train_samples, 'TEST': test_samples}
			freq_dct = None
			most_freq_class_t = None
			meta_tmp_fp = meta_fp
			biom_tmp_fp = biom_fp

		split_lst.append(split_dct)
		splgen = SampleGenerator(biom_tmp_fp, tree_fp, fasta_fp, meta_tmp_fp, param_fp, out_dir, split_lst[-1],
								 idx, class_str,
								 logger=logger_obj, freq_classes=freq_dct, most_freq_class=most_freq_class_t, seed_num=seed_num)
		splgen.traverse_tree()
		splgen.print_features_on_file(label_str=experiment)

	else:

		kmeans, meta_lst, biom_lst, label_strs, final_column = clustering(otuX, num_clusters, method, rank,
																	 labels, meta.index.values, table,
																	 out_fp=None,
																	 final_column=final_column_clustering,
																	 logger_obj=logger_obj,
																	 training_index=train_index)
		filename = out_dir + '/kmeans_model.' + str(idx) + '.pkl'
		pickle.dump(kmeans, open(filename, 'wb'))
		logger_ins.info("Wrote the kmean model on file for future use. The path is", filename)

		logger_ins.info("The final column for the clustering will be", final_column)
		for i in range(len(meta_lst)):
			class_str = final_column
			freq_dct, most_freq_class_t, _ = make_freq_dict(meta_lst[i], final_column)
			logger_ins.info("The most freq_class is", most_freq_class_t)
			logger_ins.info("The freq stat is", freq_dct)
			train_samples_index_ = np.isin(train_samples, meta_lst[i].index.values)
			test_samples_index_ = np.isin(test_samples, meta_lst[i].index.values)
			print(meta_lst[i].shape, len(train_samples), len(train_samples_index_))

			split_dct = {'TRAIN': train_samples[train_samples_index_], 'TEST': test_samples[test_samples_index_]}
			meta_tmp_fp, biom_tmp_fp = write_biom_and_meta_to_files(meta_lst[i], biom_lst[i])
			split_lst.append(split_dct)
			logger_ins.info(biom_tmp_fp, meta_tmp_fp, class_str)
			splgen = SampleGenerator(biom_tmp_fp, tree_fp, fasta_fp, meta_tmp_fp, param_fp, out_dir,
									 split_lst[-1], idx, class_str, logger=logger_obj, freq_classes=freq_dct,
									 most_freq_class=most_freq_class_t, main_class_label='class_labels', seed_num=seed_num)
			splgen.traverse_tree()
			splgen.print_features_on_file(label_str=label_strs[i])

def run_augmentationun(tree_fp, biom_fp, fasta_fp, out_dir, param_fp, meta_fp, idx, class_str, logger_obj,
						   n_repeats, n_splits, seed_num, experiment, num_clusters=4, method='braycurtis', rank=10,
						   final_column_clustering='class_labels', split_json_fp=None, split_direction=0, tr_ratio=0):
	print("Setting seed number as", seed_num)
	rskf = StratifiedKFold(n_splits=n_splits, random_state=seed_num, shuffle=True)

	table = load_table(biom_fp)
	meta = pd.read_csv(meta_fp, sep='\t', low_memory=False, dtype={'#SampleID': str},
					   index_col='#SampleID')
	meta = meta.reindex(table.ids('sample'))
	labels = meta[class_str]
	indexs = meta.index
	otuX = table.matrix_data.transpose().todense()

	for _ in range(n_repeats):
		c = 0
		meta_joined = None
		meta_most_ts = None
		most_freq_class = None
		if experiment == 'unbalance':
			if split_direction == 0:
				index_tmp, label_tmp, most_freq_class, least_freq_class, to_remove_len, meta_joined, meta_most_ts = get_index_label_unbalance_augmentation(meta, class_str)
				indexs = index_tmp
				labels = label_tmp
			else:
				_, most_freq_class, least_freq_class = make_freq_dict(meta, class_str)
				meta_joined = meta
				meta_most_ts = None
				indexs = meta.index
				labels = meta[class_str]
		elif experiment == 'augmentation':
			indexs = meta.index
			labels = meta[class_str]


		index = indexs
		label = labels
		if split_json_fp is None:
			for train_index, test_index in rskf.split(index, label):
				if c != idx:
					c += 1
					continue

				logger_ins = logger_obj.get_logger("main_test_" + str(c))
				logger_ins.info("working on split", c)
				train_samples = index[train_index]
				test_samples = index[test_index]

				run_experiment(experiment, meta_joined, meta_most_ts, label, train_index, test_index, train_samples,
							   test_samples, most_freq_class, class_str, meta_fp, biom_fp, logger_obj,
							   tree_fp, fasta_fp, param_fp, out_dir, otuX, num_clusters, method, rank, labels, meta, table,
							   final_column_clustering, logger_ins, idx, seed_num, training_ratio=tr_ratio, split_direction=split_direction)

				c += 1
		else:
			with open(split_json_fp) as f:
				split_dct = json.load(f)
			logger_ins = logger_obj.get_logger("main_test_" + str(idx))
			logger_ins.info("working on split", idx)
			train_index = split_dct['TRAIN']
			test_index = split_dct['TEST']
			train_samples = index[train_index]
			test_samples = index[test_index]
			run_experiment(experiment, meta_joined, meta_most_ts, label, train_index, test_index, train_samples,
						   test_samples, most_freq_class, class_str, meta_fp, biom_fp, logger_obj,
						   tree_fp, fasta_fp, param_fp, out_dir, otuX, num_clusters, method, rank, labels, meta, table,
						   final_column_clustering, logger_ins, idx, seed_num, training_ratio=tr_ratio)


def write_biom_and_meta_to_files(meta, table):
	meta_tmp_fp = tempfile.mktemp(dir=tmpdir)
	biom_tmp_fp = tempfile.mktemp(dir=tmpdir)
	meta.to_csv(meta_tmp_fp, sep='\t')
	with biom_open(biom_tmp_fp, 'w') as f:
		table.to_hdf5(f, "example")
	return meta_tmp_fp, biom_tmp_fp

def main(tree_fp, biom_fp, fasta_fp, out_dir, param_fp, n_splits, idx, n_repeats, seed_num, meta_fp, class_str, experiment, num_clusters=4, method='braycurtis', rank=10, final_column_clustering='class_labels', split_json_fp=None, split_direction=0, tr_ratio=1):
	np.random.seed(seed_num)
	tmp_fp = tempfile.mktemp(dir=tmpdir)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	logger_obj = logger.LOG(tmp_fp)
	print("The temporary information will be writen on the file", tmp_fp)

	run_augmentationun(tree_fp, biom_fp, fasta_fp, out_dir, param_fp, meta_fp, idx, class_str, logger_obj, n_repeats, n_splits, seed_num, experiment,
					   num_clusters=num_clusters, method=method, rank=rank, final_column_clustering=final_column_clustering, split_json_fp=split_json_fp, split_direction=split_direction, tr_ratio=tr_ratio)
	copyfile(tmp_fp, out_dir + '/main_log_' + str(idx) + ".txt")


if __name__ == "__main__":
	t0 = time()

	parser = OptionParser()
	parser.add_option("-t", "--tree", dest="tree_fp",
					  help="Phylogeny file path")

	parser.add_option("-b", "--biom", dest="biom_fp",
					  help="Biom table file path")

	parser.add_option("-s", "--sequence", dest= "fasta_fp",
					  help="Fragmentary sequences file path")

	parser.add_option("-m", "--meta", dest="meta_fp",
					  help="Meta data file path")

	parser.add_option("-o", "--output", dest="out_dir",
					  help="Output directory")

	parser.add_option("-p", "--param", dest="param_fp",
					  help="Parameter file path")

	parser.add_option("-c", "--class", dest="class_str",
					  help="Class label in meta data")

	parser.add_option("-e", "--experiment", dest="experiment",
					  help="Expriment. Options are augmentation, unbalance, clustering")

	parser.add_option("-i", "--index", dest="idx",
					  help="The cross-validation index", type="int")

	parser.add_option("-f", "--folds", dest="n_splits",
					  help="Number of folds for cross-validation", default=5, type="int")

	parser.add_option("-r", "--repeats", dest="n_repeats", default=1, type="int",
					  help="Number of repeats for cross-validation")

	parser.add_option("--seed", dest="seed_num", default=0, type="int",
					  help="Seed number")
	
	
	parser.add_option("-j",dest='split_json_fp', default=None,
					  help="The split json file path. code will look for keywords TRAIN and TEST inside the json file."+
						   "This file should have index of samples that will be used for training and testing. " +
						   "Note that the order of samples is the same as biom table.")
	parser.add_option("--num_clusters", dest="num_clusters", default=4, type="int",
					  help="Number of clusters")

	parser.add_option("--method", dest="method", help="The clustering method, either deicode or braycurtis ",
					  default="braycurtis")

	parser.add_option("--rank", dest="rank", help="rank for the deicode", default=10, type="int")

	parser.add_option("--final_column", dest="final_column_clustering", default="",
					  help="final column clustering. The options are cluster_labels (make all clusters the same size), " +
						   "class_labels (make all class labels the same size), " +
						   "clusters_and_class_labels (make all clusters and classes the same size), " +
						   "classes_within_clusters (make class labels the same in the clusters)")
	parser.add_option("--split_direction", dest="split_direction", type=int, default=0,
					 help="The direction of the split for unbalance experiments. the default value is for distressed")
	parser.add_option("--training_ratio", dest="tr_ratio", type="float", default=1,
					  help="The amount of less freq class in the data.")

	(options, args) = parser.parse_args()

	if options.tree_fp is None:
		print("Please provide a phylogeny file path")
		parser.print_help()
		sys.exit()
	else:
		tree_fp = options.tree_fp

	if options.biom_fp is None:
		print("please provide a biom table file path")
		parser.print_help()
		sys.exit()
	else:
		biom_fp = options.biom_fp

	if options.fasta_fp is None:
		parser.print_help()
		print("please provdie the fragmentary file path")
		sys.exit()
	else:
		fasta_fp = options.fasta_fp

	if options.meta_fp is None:
		parser.print_help()
		print("please provide the meta data file path")
		sys.exit()
	else:
		meta_fp = options.meta_fp

	if options.out_dir is None:
		parser.print_help()
		print("please provide the output directory")
		sys.exit()
	else:
		out_dir = options.out_dir

	if options.param_fp is None:
		parser.print_help()
		print("please provide the option file path")
		sys.exit()
	else:
		param_fp = options.param_fp

	if options.class_str is None:
		parser.print_help()
		print("please provide the class label column in the meta data")
		sys.exit()
	else:
		class_str = options.class_str

	if options.experiment is None:
		parser.print_help()
		print("please provide the experiment.")
		sys.exit()
	else:
		experiment = options.experiment


	if options.idx is None:
		parser.print_help()
		print("please provide the index of the cross-validation")
		sys.exit()
	else:
		idx = options.idx

	n_splits = options.n_splits
	n_repeats = options.n_repeats
	seed_num = options.seed_num
	split_json_fp = options.split_json_fp
	split_direction = options.split_direction
	tr_ratio = options.tr_ratio
	if options.experiment is not "unbalance" and options.experiment is not "augmentation":

		if options.num_clusters is None:
			parser.print_help()
			print("please provide the number of clusters ")
			sys.exit()
		else:
			num_clusters = options.num_clusters
		if options.method is None:
			parser.print_help()
			print("please provide the clustering method")
			sys.exit()
		else:
			method = options.method
		if options.rank is None:
			parser.print_help()
			print("please provide the rank")
			sys.exit()
		else:
			rank = options.rank
		if options.final_column_clustering is None:
			parser.print_help()
			print("please provide the final_column_clustering")
			sys.exit()
		else:
			final_column_clustering = options.final_column_clustering




	if experiment == 'unbalance' or experiment == 'augmentation':
		main(tree_fp, biom_fp, fasta_fp, out_dir, param_fp, n_splits, idx, n_repeats, seed_num, meta_fp, class_str, experiment, split_json_fp=split_json_fp, split_direction=split_direction, tr_ratio=tr_ratio)
	else:
		main(tree_fp, biom_fp, fasta_fp, out_dir, param_fp, n_splits, idx, n_repeats, seed_num, meta_fp, class_str, experiment, num_clusters=num_clusters, method=method, rank=rank, final_column_clustering= final_column_clustering, split_json_fp=split_json_fp, split_direction=split_direction, tr_ratio=tr_ratio)
	print("All experiments finished in", time() - t0, "seconds")
