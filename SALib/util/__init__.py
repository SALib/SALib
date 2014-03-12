__all__ = ["scale_samples", "scale_samples_normal", "read_param_file"]

# Rescale samples from [0, 1] to [lower, upper]
def scale_samples(params, bounds):
    for i, b in enumerate(bounds):
        params[:,i] = params[:,i] * (b[1] - b[0]) + b[0]

# Rescale standard normal samples to [mu, sigma]
def scale_samples_normal(params, bounds):
    for i, b in enumerate(bounds):
        params[:,i] = params[:,i] * b[1] + b[0]
        
def read_param_file(filename):
	
	with open(filename, "r") as file:
		names = []
		bounds = []
		num_vars = 0
		
		for row in [line.split() for line in file if not line.strip().startswith('#')]:
			num_vars += 1
			names.append(row[0])
			bounds.append([float(row[1]), float(row[2])])
	
	return {'names': names, 'bounds': bounds, 'num_vars': num_vars}
