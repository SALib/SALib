__all__ = ["scale_samples", "read_param_file"]
import csv

# Rescale samples from [0, 1] to [lower, upper]


def scale_samples(params, bounds):
    for i, b in enumerate(bounds):
        params[:, i] = params[:, i] * (b[1] - b[0]) + b[0]


def read_param_file(filename):


    names = []
    bounds = []
    num_vars = 0

    with open(filename, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        for row in reader:
            num_vars += 1
            names.append(row[0])
            bounds.append([float(row[1]), float(row[2])])

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars}
