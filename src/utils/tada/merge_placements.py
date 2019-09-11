import json
import sys
import os
import subprocess
# This code is from here https://github.com/sjanssen2/ggmap/blob/1b907d3d4bd733a6caf6453078a47d70e5105330/ggmap/analyses.py#L698

files_placement = sys.argv[1].split(",")
workdir = sys.argv[2]
WS_HOME = os.environ['WS_HOME']
print(WS_HOME)
if len(files_placement) > 1:
	sys.stderr.write("step 1) merging placement files: ")
	static = None
	placements = []
	for file_placement in files_placement:
		f = open(file_placement, 'r')
		plcmnts = json.loads(f.read())
		f.close()
		placements.extend(plcmnts['placements'])
		if static is None:
			del plcmnts['placements']
			static = plcmnts
	with open('%s/%s' % (workdir,sys.argv[3]), 'w') as outfile:
		static['placements'] = placements
		json.dump(static, outfile, indent=4, sort_keys=True)
		sys.stderr.write(' done.\n')
	sys.stderr.write("step 2) placing fragments into tree: ...\n")
