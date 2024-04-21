from typing import Dict

import pkgutil
import csv

import numpy as np


def avail_approaches(pkg):
    """Create list of available modules.

    Parameters
    ----------
    pkg : module
        module to inspect

    Returns
    -------
    method : list
        A list of available submodules
    """
    methods = [
        modname
        for importer, modname, ispkg in pkgutil.walk_packages(path=pkg.__path__)
        if modname not in ["common_args", "directions", "sobol_sequence"]
        and "test" not in modname
    ]

    return methods


def read_param_file(filename, delimiter=None):
    """Unpacks a parameter file into a dictionary

    Reads a parameter file of format::

        Param1,0,1,Group1,dist1
        Param2,0,1,Group2,dist2
        Param3,0,1,Group3,dist3

    (Group and Dist columns are optional)

    Returns a dictionary containing:
        - names - the names of the parameters
        - bounds - a list of lists of lower and upper bounds
        - num_vars - a scalar indicating the number of variables
                     (the length of names)
        - groups - a list of group names (strings) for each variable
        - dists - a list of distributions for the problem,
                    None if not specified or all uniform

    Parameters
    ----------
    filename : str
        The path to the parameter file
    delimiter : str, default=None
        The delimiter used in the file to distinguish between columns

    """
    names = []
    bounds = []
    groups = []
    dists = []
    num_vars = 0
    fieldnames = ["name", "lower_bound", "upper_bound", "group", "dist"]

    with open(filename, "r") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=delimiter)
        csvfile.seek(0)
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, dialect=dialect)
        for row in reader:
            if row["name"].strip().startswith("#"):
                pass
            else:
                num_vars += 1
                names.append(row["name"])
                bounds.append([float(row["lower_bound"]), float(row["upper_bound"])])

                # If the fourth column does not contain a group name, use
                # the parameter name
                if row["group"] is None:
                    groups.append(row["name"])
                elif row["group"] == "NA":
                    groups.append(row["name"])
                else:
                    groups.append(row["group"])

                # If the fifth column does not contain a distribution
                # use uniform
                if row["dist"] is None:
                    dists.append("unif")
                else:
                    dists.append(row["dist"])

    if groups == names:
        groups = None
    elif len(set(groups)) == 1:
        raise ValueError(
            """Only one group defined, results will not be
            meaningful"""
        )

    # setting dists to none if all are uniform
    # because non-uniform scaling is not needed
    if all([d == "unif" for d in dists]):
        dists = None

    return {
        "names": names,
        "bounds": bounds,
        "num_vars": num_vars,
        "groups": groups,
        "dists": dists,
    }


def _check_groups(problem):
    """Check if there is more than 1 group."""
    groups = problem.get("groups")
    if not groups:
        return False
    if groups == problem["names"]:
        return False

    if len(set(groups)) == 1:
        return False

    return groups


def _check_bounds(bounds):
    """Check user supplied distribution bounds for validity.

    Parameters
    ----------
    problem : dict
        The problem definition

    Returns
    -------
    tuple : containing upper and lower bounds
    """
    b = np.array(bounds)

    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    return lower_bounds, upper_bounds


def _define_problem_with_groups(problem: Dict) -> Dict:
    """
    Checks if the user defined the 'groups' key in the problem dictionary.
    If not, makes the 'groups' key equal to the variables names. In other
    words, the number of groups will be equal to the number of variables, which
    is equivalent to no groups.

    Parameters
    ----------
    problem : dict
        The problem definition

    Returns
    -------
    problem : dict
        The problem definition with the 'groups' key, even if the user doesn't
        define it
    """
    # Checks if there isn't a key 'groups' or if it exists and is set to 'None'
    if "groups" not in problem or not problem["groups"]:
        problem["groups"] = problem["names"]
    elif len(problem["groups"]) != problem["num_vars"]:
        raise ValueError(
            "Number of entries in 'groups' should be the same " "as in 'names'"
        )

    return problem
