from __future__ import division

import math
import sys

import numpy as np

from .directions import directions


if sys.version_info[0] > 2:
    long = int

#==============================================================================
# The following code is based on the Sobol sequence generator by Frances
# Y. Kuo and Stephen Joe. The license terms are provided below.
#
# Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the names of the copyright holders nor the names of the
# University of New South Wales and the University of Waikato
# and its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#==============================================================================


def sample(N, D):
    """Generate (N x D) numpy array of Sobol sequence samples"""
    scale = 31
    result = np.zeros([N, D])
    
    if D > len(directions) + 1:
        raise ValueError("Error in Sobol sequence: not enough dimensions")

    L = int(math.ceil(math.log(N) / math.log(2)))

    if L > scale:
        raise ValueError("Error in Sobol sequence: not enough bits")

    for i in range(D):
        V = np.zeros(L + 1, dtype=long)

        if i == 0:
            for j in range(1, L + 1):
                V[j] = 1 << (scale - j)  # all m's = 1
        else:
            m = np.array(directions[i - 1], dtype=int)
            a = m[0]
            s = len(m) - 1

            # The following code discards the first row of the ``m`` array
            # Because it has floating point errors, e.g. values of 2.24e-314
            if L <= s:
                for j in range(1, L + 1):
                    V[j] = m[j] << (scale - j)
            else:
                for j in range(1, s + 1):
                    V[j] = m[j] << (scale - j)
                for j in range(s + 1, L + 1):
                    V[j] = V[j - s] ^ (V[j - s] >> s)
                    for k in range(1, s):
                        V[j] ^= ((a >> (s - 1 - k)) & 1) * V[j - k]

        X = long(0)
        for j in range(1, N):
            X ^= V[index_of_least_significant_zero_bit(j - 1)]
            result[j][i] = float(X / math.pow(2, scale))

    return result


def index_of_least_significant_zero_bit(value):
    index = 1
    while((value & 1) != 0):
        value >>= 1
        index += 1

    return index
