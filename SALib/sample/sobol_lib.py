#! /usr/bin/env python

import math
import numpy as np

__license__ = """
The MIT License (MIT)

Copyright (c) 2016 John Burkardt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def i4_bit_hi1(n):
    """Returns the position of the high 1 bit base 2 in an integer.

    Parameters
    ----------
    n : int
        the integer to be measured. Should be nonnegative.
        If `n` is nonpositive, the value will always be 0.

    Returns
    -------
    int
        the number of bits base 2

    Examples
    --------
          N    Binary     BIT
       ----    --------  ----
         0           0     0
         1           1     1
         2          10     2
         3          11     2
         4         100     3
         5         101     3
         6         110     3
         7         111     3
         8        1000     4
         9        1001     4
        10        1010     4
        11        1011     4
        12        1100     4
        13        1101     4
        14        1110     4
        15        1111     4
        16       10000     5
        17       10001     5
      1023  1111111111    10
      1024 10000000000    11
      1025 10000000001    11

    Notes
    -----
    Licensing:
        This code is distributed under the MIT license.
    Modified:
         22 February 2011
    Author:
         Original MATLAB version by John Burkardt.
         PYTHON version by Corrado Chisari

    """
    i = int(n)
    bit = 0
    while (True):
        if (i <= 0):
            break
        bit += 1
        i = (i // 2)
    return bit


def i4_bit_lo0(n):
    """Returns the position of the low 0 bit base 2 in an integer.

    Parameters
    ----------
    n : int
        the integer to be measured. `n`should be nonnegative.

    Returns
    -------
    int
        the position of the low 1 bit.

    Examples
    --------

            N    Binary     BIT
      ----    --------  ----
         0           0     1
         1           1     2
         2          10     1
         3          11     3
         4         100     1
         5         101     2
         6         110     1
         7         111     4
         8        1000     1
         9        1001     2
        10        1010     1
        11        1011     3
        12        1100     1
        13        1101     2
        14        1110     1
        15        1111     5
        16       10000     1
        17       10001     2
      1023  1111111111     1
      1024 10000000000     1
      1025 10000000001     1

    Notes
    -----
       Licensing:
         This code is distributed under the MIT license.
       Modified:
         22 February 2011
       Author:
         Original MATLAB version by John Burkardt.
      Python version by Corrado Chisari

    """
    bit = 0
    i = int(n)
    while (1):
        bit = bit + 1
        i2 = (i // 2)
        if (i == 2 * i2):
            break
        i = i2

    return bit


def i4_sobol_generate(m, n, skip):
    """Generates a Sobol dataset.

    Parameters
    ----------
    m : int
        the spatial dimension
    n : int
        the number of points to generate
    skip : int
        the number of initial points to skip

    Returns
    -------
    float
        the points

    Notes
    -----
    Licensing:
        This code is distributed under the MIT license.
    Modified:
        22 February 2011
    Author:
        Original MATLAB version by John Burkardt.
        PYTHON version by Corrado Chisari

    """
    r = np.zeros((m, n))
    for j in range(1, n + 1):
        seed = skip + j - 2
        [r[0:m, j - 1], seed] = i4_sobol(m, seed)
    return r


def i4_sobol(dim_num, seed):
    """Generates a new quasirandom Sobol vector with each call.

    Parameters
    ----------
    dim_num : int
        the number of spatial dimensions. DIM_NUM must satisfy
        1 <= DIM_NUM <= 40.
    seed : int
         the "seed" for the sequence. This is essentially the index
         in the sequence of the quasirandom value to be generated.
         On output, SEED has been set to the appropriate next value,
         usually simply SEED+1. If SEED is less than 0 on input,
         it is treated as though it were 0. An input value of 0
         requests the first (0-th) element of the sequence.

    Returns
    -------
    float
        the next quasirandom vector.

    Notes
    -----

    Discussion:
        The routine adapts the ideas of Antonov and Saleev.
    Licensing:
        This code is distributed under the MIT license.
    Modified:
        22 February 2011
    Author:
        Original FORTRAN77 version by Bennett Fox.
        MATLAB version by John Burkardt.
        PYTHON version by Corrado Chisari

    References
    ----------
    Antonov, Saleev,
    USSR Computational Mathematics and Mathematical Physics,
    olume 19, 1980, pages 252 - 256.
    Paul Bratley, Bennett Fox,

    Algorithm 659:
    Implementing Sobol's Quasirandom Sequence Generator,
    ACM Transactions on Mathematical Software,
    Volume 14, Number 1, pages 88-100, 1988.
    Bennett Fox,

    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.
    Ilya Sobol,

    USSR Computational Mathematics and Mathematical Physics,
    Volume 16, pages 236-242, 1977.
    Ilya Sobol, Levitan,
    The Production of Points Uniformly Distributed in a Multidimensional
    Cube (in Russian),
    Preprint IPM Akad. Nauk SSSR,
    Number 40, Moscow 1976.

    """
    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v

    if ('initialized' not in globals().keys()):
        initialized = 0
        dim_num_save = -1

    if (not initialized or dim_num != dim_num_save):
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1

        # Initialize (part of) V.
        v = np.zeros((dim_max, log_max))
        v[0:40, 0] = np.transpose([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        v[2:40, 1] = np.transpose([
            1, 3, 1, 3, 1, 3, 3, 1,
            3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
            1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
            3, 1, 1, 3, 1, 3, 1, 3, 1, 3])

        v[3:40, 2] = np.transpose([
            7, 5, 1, 3, 3, 7, 5,
            5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
            5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
            5, 1, 1, 5, 7, 7, 5, 1, 3, 3])

        v[5:40, 3] = np.transpose([
            1, 7, 9, 13, 11,
            1, 3, 7, 9, 5, 13, 13, 11, 3, 15,
            5, 3, 15, 7, 9, 13, 9, 1, 11, 7,
            5, 15, 1, 15, 11, 5, 3, 1, 7, 9])

        v[7:40, 4] = np.transpose([
            9, 3, 27,
            15, 29, 21, 23, 19, 11, 25, 7, 13, 17,
            1, 25, 29, 3, 31, 11, 5, 23, 27, 19,
            21, 5, 1, 17, 13, 7, 15, 9, 31, 9])

        v[13:40, 5] = np.transpose([
            37, 33, 7, 5, 11, 39, 63,
            27, 17, 15, 23, 29, 3, 21, 13, 31, 25,
            9, 49, 33, 19, 29, 11, 19, 27, 15, 25])

        v[19:40, 6] = np.transpose([
            13,
            33, 115, 41, 79, 17, 29, 119, 75, 73, 105,
            7, 59, 65, 21,	3, 113, 61, 89, 45, 107])

        v[37:40, 7] = np.transpose([
            7, 23, 39])
        # Set POLY.
        poly = [
            1,	 3,	 7,	11,	13,	19,	25,	37,	59,	47,
            61,	55,	41,	67,	97,	91, 109, 103, 115, 131,
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299]

        atmost = 2**log_max - 1
        # Find the number of bits in ATMOST.
        maxcol = i4_bit_hi1(atmost)
        # Initialize row 1 of V.
        v[0, 0:maxcol] = 1

    # Things to do only if the dimension changed.
    if (dim_num != dim_num_save):
        # Check parameters.
        if (dim_num < 1 or dim_max < dim_num):
            msg = """I4_SOBOL - Fatal error!
            The spatial dimension DIM_NUM should satisfy:
            1 <= DIM_NUM <= {}
            But this input value is DIM_NUM = {}
            """
            raise ValueError(msg.format(dim_max, dim_num))

        dim_num_save = dim_num
        # Initialize the remaining rows of V.
        for i in range(2, dim_num + 1):
            # The bits of the integer POLY(I) gives the form of polynomial I.
            # Find the degree of polynomial I from binary encoding.
            j = poly[i - 1]
            m = 0
            while (1):
                j = math.floor(j / 2.)
                if (j <= 0):
                    break
                m = m + 1

            # Expand this bit pattern to separate components of the logical
            # array INCLUD.
            j = poly[i - 1]
            includ = np.zeros(m)
            for k in range(m, 0, -1):
                j2 = math.floor(j / 2.)
                includ[k - 1] = (j != 2 * j2)
                j = j2

            # Calculate the remaining elements of row I as explained
            # in Bratley and Fox, section 2.

            for j in range(m + 1, maxcol + 1):
                newv = v[i - 1, j - m - 1]
                l = 1
                for k in range(1, m + 1):
                    l = 2 * l
                    if (includ[k - 1]):
                        newv = np.bitwise_xor(
                            int(newv), int(l * v[i - 1, j - k - 1]))
                v[i - 1, j - 1] = newv

        # Multiply columns of V by appropriate power of 2.
        l = 1
        for j in range(maxcol - 1, 0, -1):
            l = 2 * l
            v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l

    # RECIPD is 1/(common denominator of the elements in V).

        recipd = 1.0 / (2 * l)
        lastq = np.zeros(dim_num)

    seed = int(math.floor(seed))

    if (seed < 0):
        seed = 0

    if (seed == 0):
        l = 1
        lastq = np.zeros(dim_num)

    elif (seed == seed_save + 1):
        # Find the position of the right-hand zero in SEED.
        l = i4_bit_lo0(seed)

    elif (seed <= seed_save):

        seed_save = 0
        l = 1
        lastq = np.zeros(dim_num)

        for seed_temp in range(int(seed_save), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]),
                                              int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

    elif (seed_save + 1 < seed):

        for seed_temp in range(int(seed_save + 1), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]),
                                              int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

    # Check that the user is not calling too many times!
    if (maxcol < l):
        msg = """Too many calls
        MAXCOL = {}\n
        L =			{}"""
        return ValueError(msg.format(maxcol, l))

    # Calculate the new components of QUASI.
    quasi = np.zeros(dim_num)
    for i in range(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l - 1]))

    seed_save = seed
    seed = seed + 1

    return [quasi, seed]


def i4_uniform(a, b, seed):
    """Returns a scaled pseudorandom I4.

    Parameters
    ----------
    a : int
        the minimum and maximum acceptable values
    b : int
        the minimum and maximum acceptable values
    seed : int
        a seed for the random number generator


    Returns
    -------
    c : int
        the randomly chosen integer.
    seed : int
        the updated seed.

    Notes
    -----
    Discussion:
    The pseudorandom number will be scaled to be uniformly distributed
    between A and B.

    Licensing:
    This code is distributed under the MIT license.

    Modified:
    22 February 2011

    Author:
    Original MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari

    References
    ----------
    Paul Bratley, Bennett Fox, Linus Schrage,
      A Guide to Simulation,
      Springer Verlag, pages 201-202, 1983.
         Pierre L'Ecuyer,

    Random Number Generation,
      in Handbook of Simulation,
      edited by Jerry Banks,
      Wiley Interscience, page 95, 1998.
         Bennett Fox,

    Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.
         Peter Lewis, Allen Goodman, James Miller

      A Pseudo-Random Number Generator for the System/360,
      IBM Systems Journal,
      Volume 8, pages 136-143, 1969.
       """
    if (seed == 0):
        msg = "Input SEED = 0"
        raise ValueError(msg)

    seed = math.floor(seed)
    a = round(a)
    b = round(b)

    seed = np.mod(seed, 2147483647)

    if (seed < 0):
        seed = seed + 2147483647

    k = math.floor(seed / 127773)

    seed = 16807 * (seed - k * 127773) - k * 2836

    if (seed < 0):
        seed = seed + 2147483647

    r = seed * 4.656612875E-10
    # Scale R to lie between A-0.5 and B+0.5.
    r = (1.0 - r) * (min(a, b) - 0.5) + r * (max(a, b) + 0.5)
    # Use rounding to convert R to an integer between A and B.
    value = round(r)

    value = max(value, min(a, b))
    value = min(value, max(a, b))

    c = value

    return [int(c), int(seed)]


def prime_ge(n):
    """returns the smallest prime greater than or equal to `n`.

    Parameters
    ----------
    n: int
        the number to be bounded

    Returns
    -------
    int
        the smallest prime number that is greater than or equal to `n`

    Example
    -------
        >>> prime_ge(-10)
        2
        >>> prime_ge(1)
        2

    Notes
    -----
    Licensing:
    This code is distributed under the MIT license.

    Modified:
    22 February 2011

    Author:
    Original MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari
    """
    p = max(math.ceil(n), 2)
    while (not isprime(p)):
        p = p + 1

    return p


def isprime(n):
    """Returns True if N is a prime number, False otherwise

    Parameters
    ----------
    n: int
        Input, integer N, the number to be checked.

    Returns
    -------
    bool
        Output, boolean value, True or False

    Notes
    -----
    Licensing:
    This code is distributed under the MIT license.

    Modified:
    22 February 2011

    Author:
    Corrado Chisari
    """
    if n != int(n) or n < 1:
        return False
    p = 2
    while p < n:
        if n % p == 0:
            return False
        p += 1

    return True
