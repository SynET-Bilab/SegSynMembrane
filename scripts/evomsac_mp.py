#!/usr/bin/env python

import sys
import multiprocessing
from synseg.evomsac import EAPop

if __name__ == "__main__":
    # setup
    # read args
    args = sys.argv[1:]
    state_pkl = args[0]
    n_gen = int(args[1])
    dump_step = int(args[2])

    # setup
    eap = EAPop(state=state_pkl)

    # run parallel
    pool = multiprocessing.Pool()
    eap.register_map(pool.map)
    eap.init_pop()
    eap.evolve(n_gen, dump_step=dump_step, state_pkl=state_pkl)
    pool.close()
