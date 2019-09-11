from optparse import OptionParser
from TADA_utils import *
import sys
from time import time
import re
from io import StringIO
import pandas as pd
from biom.util import biom_open

def read_table_and_labels(biom_fp, meta_fp):
	if re.search(".tsv", biom_fp):
		tsv_fh = StringIO(biom_fp)
		func = lambda x: x
		table = biom.Table.from_tsv(tsv_fh, None, None, func)

	else:
		table = biom.load_table(biom_fp)

	if meta_fp:
		meta = pd.read_csv(meta_fp, sep='\t', low_memory=False, dtype = {0: str}, header=None, usecols=[0,1])
		y = meta.iloc[:, 1]
		samples = meta.iloc[:, 0]
	else:
		y = np.ones((len(table.ids('sample')), 1))
		samples = table.ids('sample')

	table = table.sort_order(axis='sample', order=samples)

	if sum(samples != table.ids('sample')) > 0:
		print("The samples IDs in meta data and biom table are not the same! Please double check.")
		print("The difference is:", set(samples) - set(table.ids('sample')))
		exit()
	return table, y

def make_data_frame(orig_biom, augm_biom, orig_labels, augm_labels):
	orig_dct = dict()
	augm_dct = dict()

	orig_dct['samples'] = orig_biom.ids('sample')
	orig_dct['labels'] = orig_labels
	orig_pd = pd.DataFrame(data=orig_dct)

	augm_dct['samples'] = augm_biom.ids('sample')
	augm_dct['labels'] = augm_labels
	augm_pd = pd.DataFrame(data=augm_dct)
	return orig_pd, augm_pd

def write_biom_and_meta_data(orig_biom, orig_pd, augm_biom, augm_pd, out_dir, biom_fp, meta_fp):
	with biom_open(out_dir + '/' + os.path.basename(biom_fp), 'w') as f:
		orig_biom.to_hdf5(f, "original biom table")
	with biom_open(out_dir + '/augmented_data.biom', 'w') as f:
		augm_biom.to_hdf5(f, "augmented biom table")
	if meta_fp is not None:
		orig_pd.to_csv(out_dir +  '/' + os.path.basename(meta_fp), sep = '\t', header = False, index = False)
		augm_pd.to_csv(out_dir + '/augmented_meta_data.csv', sep = '\t', header = False, index = False)

	return

t0 = time()

parser = OptionParser()
parser.add_option("-t", "--tree", dest="tree_fp",
				  help="Phylogeny file in newick format.")

parser.add_option("-b", "--biom", dest="biom_fp",
				  help="The count table. This can be a biom (file with suffix .biom) or TSV (Tab Separated Values) "
					   "file (file with suffix .tsv). In TSV format rows define features, and columns define samples."
					   "The first column defines the feature IDs. "
					   "The first row defines a header where from the second column sample IDs are listed. ")

parser.add_option("-o", "--output", dest="out_dir",
				  help="The output directory.")


parser.add_option("--seed", dest="seed_num", default=0, type="int",
				  help="Seed number. The default value is 0.")

parser.add_option("-g", "--generate_strategy", dest = "generate_strategy", default="",
				  help="Specifies the generating strategy for either balancing or data augmentation without balancing."
					   " If TADA is used for augmentation, this shouldn't be passed. Otherwise, pass a meta data file "
					   "(in TSV format, a tab delimited with no header). "
					   "The first column should be samples, and second column should be class labels.")

parser.add_option("-x", "--xgen", dest = "xgen", help="Amount of generation for balancing. If TADA is used for only "
													  "balancing (no extra augmentation afterwards), 0 should be passed. "
													  "In balancing, TADA eventually will generate new samples "
													  "until all classes have [xgen+1] * [maximum class size] samples. "
													  "The default value is 0", default=0, type="int")

parser.add_option("-k","--n_beta", dest = "n_beta",
				  help="The number of draws from the beta distribution. For augmentation, TADA will generate "
					   "[n_binom]*[n_beta] samples per each sample. The default value is 1.", default=1, type="int")

parser.add_option("-u", "--n_binom", dest="n_binom",
				  help="The number of draws from binomial distribution. For augmentation, TADA will generate "
					   "[n_binom]*[n_beta] samples per each sample. The default value is 5", default=5, type=int)

parser.add_option("-v", "--var_method", dest="var_method",
				  help="Defines how to introduce the variation. Options are br_penalized and class. "
					   "The br_penalized can be used with a monotonically increasing function of "
					   "branch length to define the variation. The class options can be used to use"
					   " estimate the variation from training data. We suggest using br_penalized (default).",
				  default="br_penalized", type="str")

parser.add_option("-z", "--stat_method", dest="stat_method", default="binom", type="str",
				  help="The generative model. Options are binom or beta_binom, and the default option is binom.")

parser.add_option("-r", "--prior_weight", dest="prior_weight", default=0, type = "float",
				  help="The class conditional probability weight. The default is 0.")

parser.add_option("-c", "--coef", dest="coef", default=200, type="float",
				  help="The penalty factor in the calculation of nu. The default value is 200. This affects the amount of variation.")

parser.add_option("--exponent", dest="exponent", default=0.5, type="float",
				  help="The exponent in the calculation of nu. The default value is 0.5. This affects the amount of variation.")

parser.add_option("--br_pseudo", dest="pseudo", default=1e-6, type="float",
				  help="A pesudo small branch length will be added to all branches to avoid zero branch length "
					   "estimate problem. The default value is 1e-6.")
parser.add_option("--pseudo_cnt", dest="pseudo_cnt", default= 5, type="int",
				  help="Pseudo count to avoid zero count problem. The default value is adding 5, "
					   "meaning we add 5/#leaves to each feature value.")
parser.add_option("--normalized", dest="normalized", default=0, type="int",
				  help="If set to 1, the OTU counts will be normalized to add up to one. The default option is 0.")

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



if options.out_dir is None:
	parser.print_help()
	print("please provide the output directory")
	sys.exit()
else:
	if os.path.exists(options.out_dir):
		out_dir = options.out_dir
	else:
		out_dir = options.out_dir
		os.mkdir(out_dir)


seed_num = options.seed_num


if not os.path.exists(options.generate_strategy):
	meta_fp = None
	generate_strategy = "augmentation"
else:
	meta_fp = options.generate_strategy
	generate_strategy = "balancing"


xgen = options.xgen
n_beta = options.n_beta
n_binom = options.n_binom
var_method = options.var_method
stat_method = options.stat_method
prior_weight = options.prior_weight
coef = options.coef
exponent = options.exponent
pseudo = options.pseudo
pseudo_cnt = options.pseudo_cnt
normalized = options.normalized


sG = SampleGenerator(seed_num=seed_num, logger=None, generate_strategy=generate_strategy, tmp_dir=None,
				 xgen=xgen, n_beta=n_beta, n_binom=n_binom, var_method=var_method, stat_method=stat_method, prior_weight=prior_weight,
				 coef=coef, exponent = exponent, pseudo= pseudo, pseudo_cnt=pseudo_cnt, normalized=normalized)
table, y = read_table_and_labels(biom_fp, meta_fp)
orig_biom, orig_labels,  augm_biom, augm_labels = sG.fit_transform(table=table, y=y, tree_fp=tree_fp)

if np.sum(orig_biom.matrix_data - table.matrix_data) > 1e-20:
	print("The original biom table doesn't match the output of generator function! Please double check")
	exit()


orig_pd, augm_pd = make_data_frame(orig_biom, augm_biom, orig_labels, augm_labels)

write_biom_and_meta_data(orig_biom, orig_pd, augm_biom, augm_pd, out_dir, biom_fp, meta_fp)

os.rename(sG.log_fp, out_dir + '/' + os.path.basename(sG.log_fp))
os.rmdir(sG.tmp_dir)

print("The output biom and pandas files are writen on", out_dir)

print("All experiments finished in", time() - t0, "seconds")
