from sklearn.ensemble import RandomForestClassifier
from augmentation import *
import logger
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import pandas as pd
WS_HOME = os.environ['WS_HOME']
import glob
from optparse import OptionParser
import tempfile
if 'TMPDIR' in os.environ:
	tmpdir = os.environ['TMPDIR']
else:
	tmpdir = tempfile.gettempdir()
import shutil

def eval_results(y_pred_test, y_pred_proba_test, y_test, label, logger_ins):
	y_test = np.asarray(list(map(int, y_test)))


	logger_ins.info(y_test.shape, y_pred_test.shape)

	logger_ins.info("accuracy for testing set for part", label, "is: ",
						accuracy_score(y_test, y_pred_test))
	logger_ins.info("AUC for testing set for part", label, "is:",
						roc_auc_score(y_test, y_pred_proba_test[:,1]))
	return

def untar(tar_fp):
	tar = tarfile.open(tar_fp)
	tar.extractall(os.path.dirname(tar_fp))
	tar.close()
	dir_fp = os.path.dirname(tar_fp)
	augm_fp = glob.glob(dir_fp + '/tmp*/*generated*.npz')[0]
	train_fp = glob.glob(dir_fp + '/tmp*/*train*npz')[0]
	test_fp = glob.glob(dir_fp + '/tmp*/*test*npz')[0]
	print("The augmented data is located on", augm_fp)
	print("The training data is located on", train_fp)
	print("The testing data is located on", test_fp)
	return augm_fp, train_fp, test_fp


def load_data(train_fp, test_fp, augm_fp, tar_fp):
	if tar_fp is not None:
		augm_fp, train_fp, test_fp = untar(tar_fp)
	train_dct = np.load(train_fp)
	test_dct = np.load(test_fp)
	augm_dct = np.load(augm_fp)
	return augm_dct, train_dct, test_dct



def make_freq_dict_labels(labels):
	freq_dct = dict()
	labels, counts = np.unique(np.asarray(labels).squeeze(), return_counts = True)
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


def get_meta_tables(meta_fp, class_label, samples_tr, labels_tr, samples_ts, labels_ts, samples_augm, logger_ins):
	meta = pd.read_csv(meta_fp, sep='\t', low_memory=False, dtype={'#SampleID': str}, index_col='#SampleID')
	meta_tr = meta.loc[meta.index.isin(samples_tr)]
	meta_ts = meta.loc[meta.index.isin(samples_ts)]
	meta_tr = meta_tr.reindex(samples_tr)
	meta_ts = meta_ts.reindex(samples_ts)
	labels_new_tr = meta_tr[class_label]
	labels_new_ts = meta_ts[class_label]

	if not np.all(labels_tr == labels_new_tr.values)  and np.all(labels_ts == labels_new_ts.values):
		print("Something is wrong with labels")
		sys.exit()
	else:
		logger_ins.info("The training and testing labels checked they look fine! ")
	check_augmeted_samples(meta_tr, samples_augm, logger_ins)
	return meta_tr, meta_ts


def check_augmeted_samples(meta, samples_augm, logger_ins):
	if np.isin(samples_augm, meta.index.values).sum() != len(samples_augm):
		print("something with the samples is wrong")
		sys.exit()
	else:
		logger_ins.info("The samples used for augmentation are checked! They look fine.")


def get_indices_training_and_test_unb(labels, most_freq_class):
	unique, counts = np.unique(labels, return_counts=True)
	mapping = dict()
	if int(unique[0]) is int(most_freq_class[1]):
		mapping['most'] = np.isin(labels, unique[0])
		mapping['least'] = np.isin(labels, unique[1])
	else:
		mapping['most'] = np.isin(labels, unique[1])
		mapping['least'] = np.isin(labels, unique[0])
	return mapping

def split_meta(meta, samples, mapping):
	meta_mst = meta.loc[meta.index.isin(samples[mapping['most']])]
	meta_lst = meta.loc[meta.index.isin(samples[mapping['least']])]
	return meta_mst, meta_lst
def get_new_labels(meta, samples, mapping, testing_ratio, direction):
	meta_mst, meta_lst = split_meta(meta, samples, mapping)
	len_mst = len(meta_mst)
	len_lst = len(meta_lst)

	if testing_ratio > 0.5:
		testing_ratio = 1 - testing_ratio


	if direction == 1:
		len_lst, len_mst = get_x(len_lst, len_mst, testing_ratio)
	else:
		len_lst, len_mst = get_x(len_lst, len_mst, 1-testing_ratio)

	samples_mst = meta_mst.index.values
	samples_lst = meta_lst.index.values
	np.random.shuffle(samples_mst)
	np.random.shuffle(samples_lst)
	final_samples_mst = samples_mst[:len_mst]
	final_samples_lst = samples_lst[:len_lst]
	final_samples = np.row_stack((final_samples_mst.reshape((-1,1)), final_samples_lst.reshape((-1,1))))
	print(len(final_samples))
	return final_samples

def get_x(f1, f2, ratio):
	T = f1 + f2
	x2 = int(T - f1/ratio) + 1
	x1 = int((f1 - T * ratio)/(1-ratio)) + 1
	if x1 < 0 and x2 >= 0 and f2-x2>=0:
		return f1, f2 - x2
	if x1 >= 0 and x2 < 0 and f1-x1>=0:
		return f1 - x1, f2
	if x1 > 0 and x2 >0:
		if x1 > x2 and f2-x2 >= 0:
			return f1, f2 - x2
		elif f1-x1 >= 0:
			return f1 - x1, f2
	else:
		print("It is not feasible to do this expeirment! f1:", f1, "f2:", f2, "x1:", x1, "x2:", x2)

def get_results_dct(y_train, X_test_aug_predict, X_test_orig_predict, X_test_aug_proba, X_test_orig_proba, y_test, samples_orig_tr, samples_orig_ts, samples_augm, labels_augm, labels_orig_tr):
	results = dict()
	results['y_train'] = y_train
	results['X_test_aug_predict'] = X_test_aug_predict
	results['X_test_orig_predict'] = X_test_orig_predict
	results['X_test_aug_proba'] = X_test_aug_proba
	results['X_test_orig_proba'] = X_test_orig_proba
	results['y_test'] = y_test
	results['samples_orig_tr'] = samples_orig_tr
	results['samples_orig_ts'] = samples_orig_ts
	results['samples_augm'] = samples_augm
	results['labels_augm'] = labels_augm
	results['labels_orig_tr'] = labels_orig_tr
	return results

def make_test_dataset(X_test, meta_fp, class_label, samples_orig_tr, labels_orig_tr, samples_orig_ts, y_test, samples_augm, logger_ins, testing_ratio, direction):
	meta_tr, meta_ts = get_meta_tables(meta_fp, class_label, samples_orig_tr, labels_orig_tr, samples_orig_ts, y_test, samples_augm, logger_ins)

	freq_dct_tr, most_freq_class_tr, least_freq_class_tr = make_freq_dict_labels(labels_orig_tr)

	mapping_tr = get_indices_training_and_test_unb(labels_orig_tr, most_freq_class_tr)
	mapping_ts = get_indices_training_and_test_unb(y_test, most_freq_class_tr)
	print("The most prev class in training", len(samples_orig_tr[mapping_tr['most']]), "least prev class in training", len(samples_orig_tr[mapping_tr['least']]), direction)
	print("The most prev class based on training in testing", len(samples_orig_ts[mapping_ts['most']]), "least prev class based on training in testing", len(samples_orig_ts[mapping_ts['least']]), direction)
	final_samples = get_new_labels(meta_ts, samples_orig_ts, mapping_ts, testing_ratio, direction)
	final_idx_ts = np.isin(samples_orig_ts, final_samples)
	X_test = X_test[final_idx_ts]
	y_test = y_test[final_idx_ts]
	samples_orig_ts = samples_orig_ts[final_idx_ts]
	freq_dct_ts, _, _ = make_freq_dict_labels(y_test)

	logger_ins.info("The frequency dictionary for the test classes after "
					"setting up the experiment according to training freq dict", freq_dct_tr, "direction",
					direction,"and ratio", testing_ratio, "is", freq_dct_ts)

	return X_test, y_test, samples_orig_ts

def do_clssification(clf, X_train, y_train, X_test, y_test, label, output_label, logger_ins):
	t0 = time()
	_ = clf.fit(X_train, y_train.ravel())
	predicted_probabilities = clf.predict_proba(X_test)
	predicted_labels = clf.predict(X_test)
	eval_results(predicted_labels, predicted_probabilities, y_test, "experiment: " + label + " " + output_label, logger_ins)
	logger_ins.info("Time to compute", label,time() - t0)
	return predicted_labels, predicted_probabilities

def get_data_from_dct(augm_dct, train_dct, test_dct):
	X_augm = augm_dct.f.data
	labels_augm = augm_dct.f.labels
	samples_augm = augm_dct.f.samples

	X_orig_tr = train_dct.f.data
	labels_orig_tr = train_dct.f.labels
	samples_orig_tr = train_dct.f.samples

	X_test = test_dct.f.data
	y_test = test_dct.f.labels
	samples_orig_ts = test_dct.f.samples
	return X_augm, labels_augm, samples_augm, X_orig_tr, labels_orig_tr, samples_orig_tr, X_test, y_test, samples_orig_ts

def get_classifier_obj(method, n_jobs, seed_num, logger_ins, class_weight=None):
	if method == "rf":
		if class_weight == 'original':
			class_weight = None
		logger_ins.info("creating the rf classifier")
		clf = RandomForestClassifier(n_estimators=2000, n_jobs= n_jobs, random_state=seed_num, class_weight=class_weight)
	else:
		logger_ins.info("creating the nn classifier")
		clf = MLPClassifier(hidden_layer_sizes=(2000, 1000, ), learning_rate='adaptive', random_state=seed_num,  verbose=True, early_stopping=True)
	return clf

def balance_with_subsample(X, y, augm_levels, down_sample, logger_ins):
	freq_dct, most_freq_class, least_freq_class = make_freq_dict_labels(y)
	if augm_levels >=0:
		extra_to_generate = augm_levels *  most_freq_class[0]
		diff_freq = most_freq_class[0] - least_freq_class[0]
		most_freq_idx = np.where(np.isin(y, most_freq_class[1]))[0]
		least_freq_idx = np.where(np.isin(y, least_freq_class[1]))[0]
		augm_to_increase_mst = np.random.choice(most_freq_idx, size=extra_to_generate, replace=True)
		augm_to_increase_lst = np.random.choice(least_freq_idx, size=extra_to_generate + diff_freq, replace=True)
		augm_to_increase = np.row_stack((augm_to_increase_mst.reshape((-1,1)), augm_to_increase_lst.reshape((-1,1))))
		X_subsampled_train = np.concatenate([X, X[augm_to_increase]])
		y_subsampled_train = np.concatenate([y, y[augm_to_increase]])
		freq_dct, _, _ = make_freq_dict_labels(y_subsampled_train)
		logger_ins.info("The size of new data (by subsampling) is", X_subsampled_train.shape)
		logger_ins.info("The frequency of classes is", freq_dct)
		return X_subsampled_train, y_subsampled_train

	elif down_sample == 1:
		most_freq_idx = np.where(np.isin(y, most_freq_class[1]))[0]
		least_freq_idx = np.where(np.isin(y, least_freq_class[1]))[0]
		samples_to_select = np.random.choice(most_freq_idx, size=least_freq_class[0], replace=False)
		augm_to_increase = np.row_stack((samples_to_select.reshape((-1,1)), least_freq_idx.reshape((-1,1))))
		X_subsampled_train = X[augm_to_increase]
		y_subsampled_train = y[augm_to_increase]
		freq_dct, _, _ = make_freq_dict_labels(y_subsampled_train)
		logger_ins.info("The size of new data (by downsampling) is", X_subsampled_train.shape)
		logger_ins.info("The frequency of classes is", freq_dct)
		return X_subsampled_train, y_subsampled_train
	else:
		return None, None

def upsample_data(X, y, augm_levels, logger_ins):
	freq_dct, most_freq_class, least_freq_class = make_freq_dict_labels(y)
	most_freq_idx = np.where(np.isin(y, most_freq_class[1]))[0]
	least_freq_idx = np.where(np.isin(y, least_freq_class[1]))[0]
	augm_to_increase_mst = np.random.choice(most_freq_idx, size= (augm_levels-1) * most_freq_class[0], replace=True)
	augm_to_increase_lst = np.random.choice(least_freq_idx, size= (augm_levels-1)* least_freq_class[0], replace=True)
	augm_to_increase = np.row_stack((most_freq_idx.reshape((-1,1)), least_freq_idx.reshape((-1,1)), augm_to_increase_mst.reshape((-1, 1)), augm_to_increase_lst.reshape((-1, 1))))
	X_new = X[augm_to_increase]
	freq_dct, _, _ = make_freq_dict_labels(X_new)
	logger_ins.info("The size of new data (by downsampling) is", X_new.shape)
	logger_ins.info("The frequency of classes is", freq_dct)
	return X_new, y[augm_to_increase]

if __name__ == "__main__":

	t0 = time()
	t1 = time()
	parser = OptionParser()
	parser.add_option("-a", "--augmentation", dest="augm_fp",
					  help="Augmentation data file path")

	parser.add_option("-r", "--train", dest="train_fp",
					  help="Training data file path")

	parser.add_option("-s", "--test", dest= "test_fp",
					  help="Testing data file path")

	parser.add_option("-z", "--tarfile", dest="tar_fp",
					  help="The tarfile which contains all augmentation, training and testing data")

	parser.add_option("-o", "--output", dest="out_dir",
					  help="Output directory")

	parser.add_option("-p", "--outlabel", dest="output_label",
					  help="output label")

	parser.add_option("-m", "--method", dest="method", default="rf",
					  help="classification method, options are nn or rf")

	parser.add_option("-n", "--num_jobs", dest="n_jobs",
					  help="number of cpus to be used", default=1, type="int")

	parser.add_option("-d", "--seed", dest="seed_num",
					  help="The seed number for the classification task", type="int")

	parser.add_option("-t", "--testing", dest="testing_ratio", type="float", default=0,
					  help="The split ratio between classes in test data")

	parser.add_option("-c", "--direction", dest="direction", default=1, type=int,
					  help="If the direction of most freq vs least freq class is congruent between "
						   "training and testing."  " Options are 1 (True), 0 (False)")

	parser.add_option("-e", "--experiment", dest="experiment", default="unbalance",
					  help="The experiment to be run. Options are unbalance, and augmentation")

	parser.add_option("--meta", dest="meta_fp",
					 help="Optional, the meta data file path")

	parser.add_option("--class_label", dest="class_label",
					  help="Optional. The column in the meta data to be used for the classification")

	parser.add_option("-l", dest="augm_levels", default=-1, type="int",
					  help="Optional. The amount of augmentation by subsampling for original training. If negative won't do it")

	parser.add_option("-u", dest="use_augm", default=1, type="int",
					  help="Wheather to use augmented data or not")

	parser.add_option("-w", dest="class_weight", default='original',
					  help="Wheather to do do class weighting or not (for random forest). The default is None, and other options are original, balanced_subsample, and balanced")

	parser.add_option("--trim_train", dest="trim_train_flg", default=0, type="int",
					  help="Wheather to trim training before running the experiment or not. Default is no (0)")

	parser.add_option("--trim_train_ration", dest="trim_train_ratio", default=0, type="float",
					  help="If you want to trim the training beforehand, what's the ratio. Default is None (using the original ones)")


	(options, args) = parser.parse_args()

	if options.tar_fp is None and (options.augm_fp is None or options.train_fp is None or options.test_fp is None):
		print("Please provide the training, testing and augmentation file paths or the tar file which contains them")
		parser.print_help()
		sys.exit()
	else:
		augm_fp = options.augm_fp
		train_fp = options.train_fp
		test_fp = options.test_fp
		tar_fp = options.tar_fp

	if options.out_dir is None:
		print("Please provide the output directory")
		parser.print_help()
		sys.exit()
	else:
		out_dir = options.out_dir
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)

	if options.output_label is None:
		print("Please provide an output label")
		parser.print_help()
		sys.exit()
	else:
		output_label = options.output_label

	if options.meta_fp is None or options.class_label is None:
		print("please provide meta data and the class_label in the meta data")
		parser.print_help()
		sys.exit()
	else:
		meta_fp = options.meta_fp
		class_label = options.class_label
	n_jobs = options.n_jobs
	seed_num = options.seed_num
	np.random.seed(seed_num)
	method = options.method
	experiment = options.experiment
	direction = options.direction
	testing_ratio = options.testing_ratio
	augm_levels = options.augm_levels
	use_augm = options.use_augm
	trim_train_flg = options.trim_train_flg
	trim_train_ratio = options.trim_train_ratio
	class_weight = options.class_weight

	np.random.seed(seed_num)

	tmp_dir = tempfile.mkdtemp(dir=tmpdir)

	if tar_fp is not None:
		shutil.copyfile(tar_fp, tmp_dir + '/data.tar')
		tmp_tar_fp = tmp_dir + '/data.tar'
	else:
		tmp_tar_fp = None

	log_fp = tmp_dir +  "/log_main_test_" + str(output_label) + ".txt"
	print("The information will be written on the file", log_fp)
	print("The temporary directory is", tmp_dir)

	logger_obj = logger.LOG(log_fp)
	logger_ins = logger_obj.get_logger('classfication_'+output_label)

	augm_dct, train_dct, test_dct = load_data(train_fp, test_fp, augm_fp, tmp_tar_fp)
	X_augm, labels_augm, samples_augm, X_orig_tr, labels_orig_tr, samples_orig_tr, X_test, y_test, samples_orig_ts = get_data_from_dct(augm_dct, train_dct, test_dct)



	split_lst = list()
	c = 0

	t0 = time()


	if trim_train_flg == 1:
		X_orig_tr, labels_orig_tr, samples_orig_tr = make_test_dataset(X_orig_tr, meta_fp, class_label, samples_orig_tr, labels_orig_tr,
																	   samples_orig_tr, labels_orig_tr, samples_augm, logger_ins, trim_train_ratio, 1)

		augm_idx = np.where(np.isin(samples_augm, samples_orig_tr))[0]
		X_augm = X_augm[augm_idx]
		labels_augm = labels_augm[augm_idx]
		samples_augm = samples_augm[augm_idx]

	freq_dct, most_freq_class, least_freq_class  = make_freq_dict_labels(labels_orig_tr)
	logger_ins.info("The original training data after trimming by the ratio", trim_train_ratio, "is" , freq_dct)

	if experiment == "unbalance":
		X_test, y_test, samples_orig_ts = make_test_dataset(X_test, meta_fp, class_label, samples_orig_tr, labels_orig_tr, samples_orig_ts,
															y_test, samples_augm, logger_ins, testing_ratio, direction)



	logger_ins.info("The size of the augmentation data is", X_augm.shape)
	logger_ins.info("The size of the original training data is", X_orig_tr.shape)
	logger_ins.info("The size of the original test data is", X_test.shape)

	clf = get_classifier_obj(method, n_jobs, seed_num, logger_ins, class_weight)


	X_train = np.row_stack((X_orig_tr, X_augm))
	logger_ins.info("The size of the training data (concatenated with augmentation) is", X_train.shape)
	y_train = np.concatenate((labels_orig_tr.reshape((-1, 1)), labels_augm.reshape((-1, 1))))



	if use_augm == 1:
		X_test_aug_predict, X_test_aug_proba = do_clssification(clf, X_train, y_train, X_test, y_test, 'augmented results for part', output_label, logger_ins)
		del X_train
		del X_augm
	X_test_orig_predict, X_test_orig_proba = do_clssification(clf, X_orig_tr, labels_orig_tr, X_test, y_test, 'original results for part', output_label, logger_ins)


	results = get_results_dct(y_train, X_test_aug_predict, X_test_orig_predict, X_test_aug_proba, X_test_orig_proba, y_test, samples_orig_tr, samples_orig_ts, samples_augm, labels_augm, labels_orig_tr)

	np.savez_compressed(out_dir + '/outputs_' + output_label + '.npz', **results)
	print("All classifications finished in ", time() - t1)
	shutil.copyfile(log_fp, out_dir + "/log_main_test_" + str(output_label) + ".txt")
	shutil.rmtree(tmp_dir)



