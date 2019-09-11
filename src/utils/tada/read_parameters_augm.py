import logger
import json
from utilities import *
import tempfile
import os

if 'TMPDIR' in os.environ:
	tmpdir = os.environ['TMPDIR']
else:
	tmpdir = tempfile.gettempdir()

def read_param_file(param_fp):
	param_dct = dict()
	with open(param_fp,'r') as f:
		param_lines = f.readlines()
		for param_line in param_lines:
			param_line = param_line.strip()
			param_line_lst = param_line.split('\t')
			param_dct[param_line_lst[0]] = param_line_lst[1:]
	return param_dct
def read_params(param_fp, out_dir, logger_fp=None):

	param_pre = read_param_file(param_fp)
	param_pre["n_binom"] = return_instance(param_pre.get("n_binom", 10))
	param_pre['seed_num'] = return_instance(param_pre.get("seed_num", 0))
	param_pre["exponent"] = return_instance(param_pre.get("exponent", 1/3))
	param_pre['coef'] = return_instance(param_pre.get("penalty", 200))
	param_pre['pseudo'] = return_instance(param_pre.get("pseudo", 1e-6))
	param_pre['n_sample'] = return_instance(param_pre.get("n_sample", 1))
	param_pre['prior_weight'] = return_instance(param_pre.get("prior_weight", 0.0))
	param_pre['bootstrap_size'] = return_instance(param_pre.get("bootstrap_size", 'same'))
	param_pre['class_label'] = return_instance(param_pre.get('class_label', None))
	param_pre['n_gen_bootstrap'] = return_instance(param_pre.get('n_gen_bootstrap', 1))
	param_pre['pseudo_cnt'] = return_instance(param_pre.get('pseudo_cnt', 5))
	param_pre['var_method'] = return_instance(param_pre.get('var_method', 'br_penalized'))
	param_pre['mini_cluster'] = return_instance(param_pre.get('mini_cluster', 10))
	param_pre['n_beta'] = return_instance(param_pre.get('n_beta', 5))
	param_pre['xgen'] = return_instance(param_pre.get('xgen', 1))
	param_pre['stat_method'] = return_instance(param_pre.get('stat_method', 'beta_binom'))
	param_pre['n_for_each_augment'] =return_instance(param_pre.get('n_for_each_augment', 100))
	param_pre['n_augment'] = return_instance(param_pre.get('n_augment', 5))
	'''
	generate_strategy defines how you want to generate new samples. Options are: 
	individual: treat each sample seperately, with a prior from the whole class
	bootstrap: do bootstrapping (of size k, default is same size as the number of samples) to generate new datasets and generate new samples based on these newly generated ones
	'''

	param_pre['generate_strategy'] = return_instance(param_pre.get("generate_strategy", "individual"))
	param_pre['tmp_dir'] = tempfile.mkdtemp(dir=tmpdir)
	param_pre['final_out_dir'] = out_dir
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	logger_fp = param_pre['tmp_dir'] + '/log.txt'
	param_pre['log_fp'] = logger_fp

	logger_obj = logger.LOG(logger_fp)
	param_pre["logger"] = logger_obj
	logger_ins = logger_obj.get_logger("reader")

	for key in param_pre:
		logger_ins.info("The parameter", key, "is set to", param_pre[key])

	return param_pre
