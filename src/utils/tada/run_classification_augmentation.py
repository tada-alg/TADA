from sklearn.ensemble import RandomForestClassifier
from augmentation import *
import logger
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
WS_HOME = os.environ['WS_HOME']
import glob
from optparse import OptionParser
np.random.seed(0)
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
	

	parser.add_option("-w", dest="class_weight", default='original',
					  help="Wheather to do do class weighting or not (for random forest). The default is None, and other options are original, balanced_subsample, and balanced")

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

	n_jobs = options.n_jobs
	seed_num = options.seed_num
	method = options.method
	class_weight = options.class_weight
	print("method is", method)

	if  method != 'nn_pytorch':
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
		X_augm = augm_dct.f.data
		labels_augm = augm_dct.f.labels
		samples_augm = augm_dct.f.samples


		X_orig_tr = train_dct.f.data
		labels_orig_tr = train_dct.f.labels
		samples_orig_tr = train_dct.f.samples

		X_test = test_dct.f.data
		y_test = test_dct.f.labels
		samples_orig_ts = test_dct.f.samples

		logger_ins.info("The size of the augmentation data is", X_augm.shape)
		logger_ins.info("The size of the original training data is", X_orig_tr.shape)
		logger_ins.info("The size of the original test data is", X_test.shape)


		split_lst = list()
		c = 0

		t0 = time()
		if method == "rf":
			logger_ins.info("creating the rf classifier")
			if class_weight == 'original':
				clf = RandomForestClassifier(n_estimators=2000, n_jobs= n_jobs, random_state=seed_num,class_weight=None)
			else:
				clf = RandomForestClassifier(n_estimators=2000, n_jobs= n_jobs, random_state=seed_num,class_weight=class_weight)



		else:
			logger_ins.info("creating the nn classifier")
			clf = MLPClassifier(hidden_layer_sizes=(2000, 1000, ), learning_rate='adaptive', random_state=seed_num,  verbose=True, early_stopping=True)

		if method == 'rf' or method == 'nn':
			t0 = time()

			X_train = np.row_stack((X_orig_tr, X_augm))
			logger_ins.info("The size of the training data (concatenated with augmentation) is", X_train.shape)

			y_train = np.concatenate((labels_orig_tr.reshape((-1, 1)), labels_augm.reshape((-1, 1))))
			_ = clf.fit(X_train, y_train.ravel())
			X_test_aug_proba = clf.predict_proba(X_test)
			X_test_aug_predict = clf.predict(X_test)

			eval_results(X_test_aug_predict, X_test_aug_proba, y_test, "augmented results for part " + output_label, logger_ins)
			logger_ins.info("Time when trained on the original data plus augmented data, and tested against the test data",
							time() - t0)

			t0 = time()

			if method == "rf":
				logger_ins.info("creating the rf classifier")
				clf = RandomForestClassifier(n_estimators=2000, n_jobs=n_jobs, random_state=seed_num)

			else:
				logger_ins.info("creating the nn classifier")
				clf = MLPClassifier(hidden_layer_sizes=(2000, 1000,), learning_rate='adaptive', random_state=seed_num,
									verbose=True, early_stopping=True)

			_ = clf.fit(X_orig_tr, labels_orig_tr.ravel())
			logger_ins.info("The size of the training data (not concatenated with augmentation) is", X_orig_tr.shape)

			logger_ins.info("Time when trained on the original data, and tested against the test data", time() - t0)
			X_test_orig_proba = clf.predict_proba(X_test)
			X_test_orig_predict = clf.predict(X_test)
			eval_results(X_test_orig_predict, X_test_orig_proba, y_test, "original results for part " + output_label,
						 logger_ins)


			# X_augm_proba = clf.predict_proba(X_augm)
			# X_augm_predict = clf.predict(X_augm)
			# eval_results(X_augm_predict, X_augm_proba, labels_augm,
			# 			 "predictions when training on the original training data and tested against the augmented data " + output_label,
			# 			 logger_ins)

		results = dict()

		results['y_train'] = y_train
		results['y_test'] = y_test
		results['samples_orig_tr'] = samples_orig_tr
		results['samples_orig_ts'] = samples_orig_ts
		results['samples_augm'] = samples_augm
		results['labels_augm'] = labels_augm
		results['labels_orig_tr'] = labels_orig_tr
		results['X_test_aug_predict'] = X_test_aug_predict
		results['X_test_orig_predict'] = X_test_orig_predict
		results['X_test_aug_proba'] = X_test_aug_proba
		results['X_test_orig_proba'] = X_test_orig_proba
		np.savez_compressed(out_dir + '/outputs_' + output_label + '.npz', **results)
		print("All classifications finished in ", time() - t1)
		shutil.copyfile(log_fp, out_dir + "/log_main_test_" + str(output_label) + ".txt")
		shutil.rmtree(tmp_dir)



