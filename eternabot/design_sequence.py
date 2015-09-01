import ensemble_design
import ensemble_utils
import eterna_utils, inv_utils
import design_utils
import switch_designer
import sys
import os
import json
import argparse
import requests
import settings
import varna
import copy

def get_objective_dict(o):
    n = len(o['secstruct'])
    constrained = ensemble_design.get_sequence_array('o'*n)
    struct = ensemble_design.get_sequence_array(o['secstruct'])
    if 'structure_constrained_bases' in o.keys() and len(o['structure_constrained_bases']) > 0:
        for i in range(0, len(o['structure_constrained_bases']), 2):
            [lo, hi] = o['structure_constrained_bases'][i:i+2]
            for j in range(lo, hi+1):
                constrained[j] = 'x'
        del o['structure_constrained_bases']
    if 'anti_structure_constrained_bases' in o.keys() and len(o['anti_structure_constrained_bases']) > 0:
        for i in range(0, len(o['anti_structure_constrained_bases']), 2):
            [lo, hi] = o['anti_structure_constrained_bases'][i:i+2]
            for j in range(lo, hi+1):
                constrained[j] = 'x'
                struct[j] = '.'
        del o['anti_secstruct'], o['anti_structure_constrained_bases']
    if 'structure_unpaired_constrained_bases' in o.keys() and len(o['structure_unpaired_constrained_bases']) > 0:
        for i in range(0, len(o['structure_unpaired_constrained_bases']), 2):
            [lo, hi] = o['structure_unpaired_constrained_bases'][i:i+2]
            for j in range(lo, hi+1):
                constrained[j] = 'u'
        del o['structure_unpaired_constrained_bases']
    o['secstruct'] = ensemble_design.get_sequence_string(struct)
    o['constrained'] = ensemble_design.get_sequence_string(constrained)
    return o
    
def read_puzzle_json(text, mode = "hairpin", scoring = "bpp"):
    """
    read in puzzle as a json file
    """
    data = json.loads(text)['data']
    p = data['puzzle']
    id = data['nid']

    #if p['rna_type'] == 'single':
    #    return SequenceDesigner(id, p['secstruct'], p['locks'])

    # get basic parameters
    beginseq = p['beginseq']
    constraints = p['locks']

    outputs = False
    if "outputs" in p:
        outputs = {}
        for key in p['outputs']:
            outputs[key] = get_objective_dict(p['outputs'][key])

    # load in objective secondary structures
    objective = json.loads(p['objective'])
    secstruct = [] 
    for o in objective:
        if outputs:
            objective = copy.deepcopy(outputs[o['output']])
            objective['type'] = o['type']
            objective['inputs'] = o['inputs']
            secstruct.append(objective)
        else:
            secstruct.append(get_objective_dict(o))

    if 'linker' not in p:
        p['linker'] = "AACAA"

    if p['rna_type'] == "multi_input" or p['rna_type'] == "multi_input_oligo":
        return switch_designer.SwitchDesigner(id, p['rna_type'], beginseq, constraints, secstruct, p['linker'], scoring, p['inputs'], mode)
    return switch_designer.SwitchDesigner(id, p['rna_type'], beginseq, constraints, secstruct, p['linker'], scoring, mode=mode)

def optimize_n(puzzle, niter, ncool, n, submit, draw, fout, cotrans, prints, greedy):
    # run puzzle n times
    solutions = []
    scores = []
    i = 0 
    attempts = 0
    while i < n:
        puzzle.reset_sequence()
        puzzle.optimize_sequence(niter, ncool, greedy=greedy, cotrans=cotrans, prints=prints)
        if puzzle.check_current_secstructs():
            sol = puzzle.get_solution()
            if sol[0] not in solutions:
                solutions.append(sol[0])
                scores.append(sol[2])
                print sol
                if submit:
                    post_solution(puzzle, 'solution %s' % i)
                if draw:
                    puzzle.draw_solution(i)
                if fout:
                    params = ""
                    if greedy:
                        params += "greedy "
                    if cotrans:
                        params += "cotranscriptional"
                    with open(fout, 'a') as f:
                        f.write("# %s iterations, %s coolings, %s\n" % (niter, ncool, params))
                        f.write("%s\t%1.6f\n" % (sol[0], sol[2]))
                i += 1
                attempts = 0
        else:
            #niter += 500
            print "best distance: %s" % puzzle.best_bp_distance
            print "final conc: %s" % puzzle.oligo_conc
            attempts += 1
            if attempts == 10:
                break
        print "%s sequence(s) calculated" % i
    return [solutions, scores]

def get_puzzle(id, mode, scoring):
    puzzlefile = os.path.join(settings.PUZZLE_DIR, "%s.json" % id)
    if os.path.isfile(puzzlefile): 
        with open(puzzlefile, 'r') as f:
            puzzle = read_puzzle_json(f.read(), mode, scoring)
    else:
        puzzle = get_puzzle_from_server(id, mode, scoring)
    return puzzle

def get_puzzle_from_server(id, mode, scoring):
    """
    get puzzle with id number id from eterna server
    """
    #r = requests.get('http://nando.eternadev.org/get/?type=puzzle&nid=%s' % id)
    r = requests.get('http://eternagame.org/get/?type=puzzle&nid=%s' % id)
    return read_puzzle_json(r.text, mode, scoring)

def post_solutions(puzzleid, filename, mode):
    filename = os.path.join(settings.PUZZLE_DIR, filename)
    for i, sol in enumerate(open(filename, 'r')):
        if sol.startswith('#'):
            continue
        spl = sol.split()
        post_solution(puzzleid, "eternabot solution %s" % i, "UUUGCUCGUCUUAUCUUCUUCGC" +spl[0], spl[1])
        print i
    return

def post_solution(puzzleid, title, sequence, score):
    fold = inv_utils.fold(sequence)
    design = eterna_utils.get_design_from_sequence(sequence, fold[0])
    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    login = {'type': 'login',
             'name': 'theeternabot',
             'pass': 'iamarobot',
             'workbranch': 'main'}
    solution = {'type': 'post_solution',
                'puznid': puzzleid,
                'title': title,
                'body': 'eternabot solution, score %s' % score,
                'sequence': sequence,
                'energy': fold[1],
                'gc': design['gc'],
                'gu': design['gu'],
                'ua': design['ua'],
                'melt': design['meltpoint'],
                'pointsrank': 'false',
                'recommend-puzzle': 'true'}

    url = "http://jnicol.eternadev.org"
    loginurl = "%s/login/" % url
    posturl = "%s/post/" % url
    with requests.Session() as s:
        r = s.post(loginurl, data=login, headers=header)
        r = s.post(posturl, data=solution, headers=header)
    return

def view_sequence(puzzle, seq):
    puzzle.update_sequence(seq)
    puzzle.update_best()
    print puzzle.targets
    print puzzle.get_solution()

def main():
    # parse arguments
    p = argparse.ArgumentParser()
    p.add_argument('puzzleid', help="name of puzzle filename or eterna id number", type=str)
    p.add_argument('-n', '--nsol', help="number of solutions", type=int, default=1)
    p.add_argument('-i', '--niter', help="number of iterations", type=int, default=2000)
    p.add_argument('-c', '--ncool', help="number of times to cool", type=int, default=50)
    p.add_argument('-m', '--mode', help="mode for multi inputs", type=str, default="hairpin")
    p.add_argument('-s', '--score', help="scoring function", type=str, default="bpp")
    p.add_argument('--submit', help="submit the solution(s)", default=False, action='store_true')
    p.add_argument('--draw', help="draw the solution(s)", default=False, action='store_true')
    p.add_argument('--nowrite', help="suppress write to file", default=False, action='store_true')
    p.add_argument('--cotrans', help="enable cotranscriptional folding", default=False, action='store_true')
    p.add_argument('--prints', help="print sequences throughout optimization", default=False, action='store_true')
    p.add_argument('--greedy', help="greedy search", default=False, action='store_true')
    args = p.parse_args()

    # read puzzle
    puzzle = get_puzzle(args.puzzleid, args.mode, args.score)
    if not args.nowrite:
        fout = os.path.join(settings.PUZZLE_DIR, args.puzzleid + "_" + puzzle.mode + ".out")
    else:
        fout = False
    
    #seq = "UCUACUAAAGCGCACGACUAAUGCAUAUCCGUAAACAUGAGGAUCACCCAUGUGCACCUGGUCGAGCACCUAAAGUCUGCUGCUACUGGUUUAGUAUCAACAUUCACAAACAGUCAUGAUAU"
    #puzzle.update_sequence(seq)
    #puzzle.update_best()
    #puzzle.draw_solution("solution0")
    #view_sequence(puzzle, seq)

    # find solutions
    [solutions, scores] = optimize_n(puzzle, args.niter, args.ncool, args.nsol, args.submit, args.draw, fout, args.cotrans, args.prints, args.greedy)

if __name__ == "__main__":
    #unittest.main()
    main()


