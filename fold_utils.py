import subprocess, os, sys, settings
import re, string, random, itertools
import numpy as np
import warnings

import string

from nupack import *

paramfile = 'rna'


def vienna_fold(sequence, constraint=None, bpp=False):
    """
    folds sequence using Vienna

    args:
    sequence is the sequence string

    returns:
    secondary structure
    """
    filename = '%s/%s' % (settings.TEMP_DIR, ''.join(random.sample(string.lowercase, 5)))
    with open('%s.fa' % filename, 'w') as f:
        f.write('>%s\n' % os.path.basename(filename))
        f.write('%s\n' % sequence)
        if constraint:
            options = ' -C'
            f.write('%s\n' % constraint)
        else:
            options = ''
    if bpp:
        options += ' -p'
    else:
        options += ' --noPS'
    if '&' in sequence:
        if sequence.count('&') > 1:
            print('Cannot handle more than 2 strands with Vienna - try the nupack option')
            sys.exit()
        command = 'RNAcofold'
    else:
        command = 'RNAfold'
    output = subprocess.check_output(
        os.path.join(settings.VIENNA_DIR, command) + options + ' -T 37.0 < %s' % filename + '.fa', shell=True)

    # parse the result
    toks = re.search('([AUGC]+)\s*([\.\)\(&]+)\s+\(\s*([-0-9\.]+)\s*\)', output)

    if bpp:
        bpp_matrix = np.zeros((len(sequence), len(sequence)))
        # get info from output file
        psfile = '%s_dp.ps' % os.path.basename(filename)
        if os.path.isfile(psfile):
            with open(psfile) as f:
                ps = f.read()

            # create bpp matrix from file
            lines = re.findall('(\d+)\s+(\d+)\s+(\d*\.*\d*)\s+ubox', ps)
            for ii in range(0, len(lines)):
                bpp_matrix[int(lines[ii][0]) - 1, int(lines[ii][1]) - 1] = float(lines[ii][2])
        else:
            warnings.warn('dotplot file %s_dp.ps not found' % filename)
        os.system('rm %s*' % filename)
        os.system('rm %s*' % os.path.basename(filename))
        return [toks.group(2), float(toks.group(3)), np.array(bpp_matrix)]

    os.system('rm %s*' % filename)
    return [toks.group(2), float(toks.group(3))]


def get_orderings(n):
    """
    get all possible orderings including last strand
    """

    """
    The function generates all possible orderings of strands for which the ordering includes the last strand. This is
    done by iteratively inserting the last strand at each possible position within each combination of strands for which
    the combination excludes the last strand.
    
    Questions: why not include the orders in which the last strand is in the first position of the ordering?
    Answer:     The ordering is circular because of the way that secondary structures are formed
                The order [1,2] == [2,1] as well as [1,2,3,4] == [4,1,2,3]
    """
    #todo: notably doent consider complexes with duplicates of the same strand ex: [A,A,B]

    all = []
    # loop over number of strands
    for i in range(1, n):
        # loop over each possible combination
        for order in list(itertools.combinations(list(range(1, n)), i)):
            # add last strand at each possible position
            for j in range(1, i + 1):
                order_list = list(order)
                order_list.insert(j, n)
                all.append(order_list)
    return all


def nupack_fold(sequence, oligo_conc=1, bpp=False):
    """
    finds most prevalent structure using nupack partition function
    """
    if '&' in sequence:
        return nupack_fold_multi(sequence, oligo_conc, bpp)
    else:
        return nupack_fold_single(sequence, bpp)


def nupack_fold_single(sequence, bpp=False):
    """
    finds most prevalent structure using nupack partition function for single strand
    """
    rand_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
    filename = '%s/%s' % (settings.TEMP_DIR, rand_string)
    options = ['-material', paramfile]
    with open('%s.in' % filename, 'w') as f:
        f.write('%s\n' % sequence)
    p = subprocess.Popen([os.path.join(settings.NUPACK_DIR, 'mfe')] + options + [filename], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout)
        print(stderr)
        raise ValueError('mfe command failed for %s' % rand_string)
    result = ['.' * len(sequence), 0, [1]]
    with open('%s.mfe' % filename) as f:
        line = f.readline()
        while line:
            if not line.startswith('%') and line.strip():
                energy = float(f.readline().strip())
                secstruct = f.readline().strip()
                result = [secstruct, energy, [1]]
                break
            line = f.readline()
    if bpp:
        p = subprocess.Popen([os.path.join(settings.NUPACK_DIR, 'pairs')] + options + [filename],
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        bpp_matrix = []
        if p.returncode != 0:
            print(stdout)
            print(stderr)
            raise ValueError('mfe command failed for %s' % rand_string)
        with open('%s.ppairs' % filename) as f:
            line = f.readline()
            while line:
                if not line.startswith('%') and line.strip():
                    bpp_matrix = nupack_read_bpp(f, len(sequence))
                line = f.readline()
        result.append(bpp_matrix)
    os.system('rm %s*' % filename)
    return result


def nupack_package_fold_single(sequence, bpp=False):
    # use nupack package fold utilites to find the minimum free energy, secondary structure, and bpp
    model = Model(material='rna')
    mfe_result = mfe(strands=sequence, model=model)
    # https://docs.nupack.org/definitions/#mfe-proxy-structure
    # uses mfe function from nupack utilites to find minimum free energy proxy structure:
    #  - defined as the secondary structure containing the MFE stacking state within its subensemble
    # save the dot-parens-plus notation string representatinon of the secondary structure and the minimum free energy
    # of the structure
    # todo: define proxy structure vs structure
    result = [str(mfe_result[0].structure), float(mfe_result[0].energy), [1]]
    if bpp:
        probability_matrix = pairs(strands=[sequence], model=model)
        # https://docs.nupack.org/utilities/#compute-equilibrium-base-pairing-probabilities
        # https://docs.nupack.org/definitions/#equilibrium-base-pairing-probabilities
        # use nupack pairs utilities to calculate the base pair probability matrix for the ensemble of sequences. this
        # is the probability that base pairs a and b will form at equilibrium
        result.append(probability_matrix.array)
    return result


def nupack_fold_multi(sequence, oligo_conc=1, bpp=False):
    """
    finds most prevalent structure using nupack partition function for multistrand
    """
    # ===== make a temp input file ===== #
    rand_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
    filename = '%s/%s' % (settings.TEMP_DIR, rand_string)
    split = sequence.split('&')

    # write the input sequences to the temp input file
    # writes the number of distinct sequences on the first line and then each sequence on the folling line then the
    # maximum complex size on the last line
    with open('%s.in' % filename, 'w') as f:
        f.write('%s\n' % len(split))
        for s in split:
            f.write('%s\n' % s)
        f.write('1\n')

    # write the orderings to a different temp input file
    # each line has a list of integers specifying th order of strands in the complex
    # this file specifies a list of complexes to be considered
    orderings = get_orderings(len(split))
    with open('%s.list' % filename, 'w') as f:
        for ordering in orderings:
            f.write('%s\n' % ' '.join([str(x) for x in ordering]))

    # write the concentrations to the temp input file
    with open('%s.con' % filename, 'w') as f:
        if isinstance(oligo_conc, list):
            assert len(split) == len(oligo_conc) + 1, \
                'length of concentrations must be one less than number of strands'
            f.write('%s\n' % '\n'.join([str(x) for x in oligo_conc]))
        else:
            f.write('%s\n' % oligo_conc * (len(split) - 1))
        f.write('1e-9\n')
        # todo: why this specific concentration
        # todo: what is the order of the sequences input

    options = ['-material', paramfile, '-ordered', '-mfe']  # , '-quiet']
    if bpp:
        options.append('-pairs')

    # ===== Use sub process to make Nupack complexes job ===== #
    # https://old.nupack.org/downloads/serve_public_file/nupack_user_guide_3.0.pdf?type=pdf
    # First calculates the identities of all distinct circular permutations π ∈ Π of strands for all
    # possible (unpseudoknotted) complexes up to a user-defined size Lmax and then calculates their respective
    # partition functions
    # Calculate all minimum free energy structures for each ordered complex as for the mfe executable
    p = subprocess.Popen([os.path.join(settings.NUPACK_DIR, 'complexes')] + options + [filename],
                         stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout)
        print(stderr)
        raise ValueError('complexes command failed for %s' % rand_string)

    # ===== use sub process to make Nupack concentrations job ===== #
    # Given user-defined concentrations for each strand species, calculates the equilibrium concentration of each
    # complex species or base pair in a large dilute solution
    # uses output of complexes executable
    p = subprocess.Popen([os.path.join(settings.NUPACK_DIR, 'concentrations'), '-ordered', filename],
                         stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout)
        print(stderr)
        raise ValueError('concentrations command failed for %s' % rand_string)
    # ===== get mfe ===== #
    complex = False
    # open output of the concentrations executable
    with open('%s.eq' % filename) as f_eq:
        for line in f_eq:
            if not line.startswith('%'):
                # get a list: [the strands in the complex] + [free energy] + [the concentration of the species in molar]
                # (default) Output is sorted by the concentration of each complex (default)
                # so the first line of the file should contain the complex with the highest concentration
                complex = line.strip().split()
                if float(complex[len(split) + 1]):  # len(split) is num sequences, this indexes the concentration
                    break
    # error checking
    if not complex:
        os.system('rm %s*' % filename)
        if bpp:
            return ['.' * len(sequence), 0, list(range(1, len(split) + 1)), []]
        return ['.' * len(sequence), 0, list(range(1, len(split) + 1))]

    # at this point we have the complex with the mfe structure, the free energy, and the concentration of that complex
    # get strand ordering
    with open('%s.ocx-key' % filename) as f_key:
        for line in f_key:
            # check to see if the entry in the ordered complex key file matches the identifiers of the complex from
            # the concentration file
            # The first and second columns are integer complex and ordered complex identifiers
            if line.startswith('%s\t%s' % (complex[0], complex[1])):
                # save the sequences that are in the complex matching the complex with the highest concentration
                strands = [int(x) for x in line.strip().split()[2:]]
                break
    # get mfe structure
    # save the mfe and secondary structure of the complex with the highest concentration
    with open('%s.ocx-mfe' % filename) as f_mfe:
        line = f_mfe.readline()
        while line:
            if line.startswith('%% complex%s-order%s' % (complex[0], complex[1])):
                f_mfe.readline()
                energy = f_mfe.readline().strip()
                secstruct = f_mfe.readline().strip()
                break
            line = f_mfe.readline()

    # get full secondary structure
    for i in range(len(split)):
        # the plus one is because nupack is one indexed not zero indexed
        if i + 1 not in strands:
            secstruct += '&' + '.' * len(split[i])
            strands.append(i + 1)
        # todo: this confuses me, are we adding the sequences that weren't in the complex with the highest concentration to the secondary structure?
        # todo: look at the function which is used to evaluate structures further along in the program, maybe it heavily penalizes unpaired bases (like '.' )

    if bpp:
        bpp_matrix = []
        with open('%s.cx-epairs' % filename) as f_pairs:
            line = f_pairs.readline()
            while line:
                if line.startswith('%% complex%s' % complex[0]):
                    bpp_matrix = nupack_read_bpp(f_pairs, len(sequence))
                    break
                line = f_pairs.readline()
        os.system('rm %s*' % filename)
        return [secstruct.replace('+', '&'), float(energy), strands, bpp_matrix]

    os.system('rm %s*' % filename)
    return [secstruct.replace('+', '&'), float(energy), strands]


def nupack_package_fold_multi(concatenated_sequences, oligo_conc=1, bpp=False):
    # make a list of nupack sequence objects
    # todo: consider complexes with duplicate elements (complexes without duplicates were not considered in original function)
    # todo: check understanding: make sure that complex analysis jobs will make all possible complexes using circular ordering
    # todo: understand the use of oligo_conc
    strands =[]
    for name, seq in list(zip(string.ascii_uppercase,concatenated_sequences.split('&'))):
        strands.append(Strand(sequence=seq, name=name))

    # use complex ensemble if the concentration is not provided and use test tube ensemble if it is provided


    if oligo_conc == 1:





def nupack_read_bpp(f, n):
    """
    read a base pair probability matrix from a nupack output file
    """
    bpp_matrix = np.zeros((n, n))
    f.readline()
    line = f.readline()
    while line and not line.startswith('%'):
        bp = line.strip().split()
        if int(bp[1]) - 1 != n:
            bpp_matrix[int(bp[0]) - 1, int(bp[1]) - 1] = float(bp[2])
        line = f.readline()
    return bpp_matrix


def nupack_energy(sequence, secstruct):
    rand_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
    filename = '%s/%s' % (settings.TEMP_DIR, rand_string)
    multi = '&' in sequence
    split = sequence.split('&')
    with open('%s.in' % filename, 'w') as f:
        if multi:
            f.write('%s\n' % len(split))
        for sequence in split:
            f.write('%s\n' % sequence)
        if multi:
            f.write('%s\n' % ' '.join([str(i) for i in secstruct[1]]))
            f.write(secstruct[0].replace('&', '+'))
        else:
            f.write(secstruct.replace('&', '+'))
    if '&' in sequence:
        options = ['multi']
    else:
        options = []
    result = subprocess.check_output([os.path.join(settings.NUPACK_DIR, 'energy')] + options + [filename])
    os.system('rm %s*' % filename)
    return float(result.strip().split('\n')[-1])
