import numpy as np

#__all__ = ["sample"]

def read_param_file(filename):
	
	with open(filename, "r") as file:
		names = []
		bounds = []
		num_vars = 0
		
		for row in [line.split() for line in file]:
			num_vars += 1
			names.append(row[0])
			bounds.append([float(row[1]), float(row[2])])
	
	return {'names': names, 'bounds': bounds, 'num_vars': num_vars}