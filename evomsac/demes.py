#!/usr/bin/env python
""" EvoMSAC
"""

import pickle
import itertools
import multiprocessing.dummy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import deap, deap.base, deap.tools

from synseg.utils import mask_to_coord, coord_to_mask
from synseg.evomsac import IndivMeta

class EADemes:
    """ evolving populations
    Usage:
        # evolve
        imeta = IndivMeta(B, n_vxy, n_uz, nz_eachu, r_thresh)
        eap = EADemes(imeta, n_pop, n_demes)
        eap.init_demes()
        eap.evolve(n_gen, dump_step, state_pkl)
        # dump
        eap.dump_state(state_pkl)
        # load
        eap = EADemes(state_pkl=state_pkl)
        # stats and plot
        df_stats = eap.get_log_stats()
        eap.plot_log_stats(xlim, save)
        # surface and plot
        B_arr = get_surfaces([indiv1, indiv2])
        imshow3d(imeta.B, B_arr)
    """
    def __init__(self, imeta=None, state=None, n_pop=2, n_demes=2):
        """ init
        :param imeta: IndivMeta(); if given, init; if None, init later
        :param state: state (from dump_state); or a pickle file
        :param n_pop: population size, multiples of 2
        :param n_demes: demes size
        """
        # attributes
        self.imeta = None
        self.toolbox = None
        self.stats = None
        self.demes = None
        self.log_stats = None
        self.log_best = None

        # read config
        # from imeta
        if imeta is not None:
            self.init_from_imeta(imeta)
            self.n_pop = n_pop + n_pop%2
            self.n_demes = n_demes
        
        # from state or state pickle file
        elif state is not None:
            # if a pickle file, load
            if isinstance(state, str):
                with open(state, "rb") as pkl:
                    state = pickle.load(pkl)
            # load state
            imeta = IndivMeta(config=state["imeta_config"])
            self.init_from_imeta(imeta)
            self.n_pop = state["n_pop"]
            self.n_demes = state["n_demes"]
            if state["demes_list"] is not None:
                self.demes = [
                    [self.imeta.from_list_fitness(*p) for p in pop_list]
                    for pop_list in state["demes_list"]
                ]
            if state["log_stats"] is not None:
                self.log_stats = state["log_stats"]
            if state["log_best_list"] is not None:
                self.log_best = [
                    [self.imeta.from_list_fitness(*p) for p in pop_list]
                    for pop_list in state["log_best_list"]
                ]
        else:
            raise ValueError("Should provide either imeta or state")


    def init_from_imeta(self, imeta):
        """ initialize tools for evolution algorithm
        :param imeta: IndivMeta()
        :return: None
        :action: assign variables imeta, toolbox, stats
        """
        # setup meta
        self.imeta = imeta
        self.toolbox = deap.base.Toolbox()
        
        # operations
        self.toolbox.register("evaluate", self.imeta.evaluate)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=2)
        self.toolbox.register("mate", deap.tools.cxTwoPoint)
        self.toolbox.register("mutate", self.imeta.mutate)
        self.toolbox.register("migrate", deap.tools.migRing, k=1,
            selection=deap.tools.selRandom,
            replacement=deap.tools.selRandom
        )

        # stats
        self.stats = deap.tools.Statistics(
            key=lambda indiv: indiv.fitness.values)
        self.stats.register('mean', np.mean)
        self.stats.register('best', np.min)
        self.stats.register('std', np.std)
    
    def dump_state(self, state_pkl=None):
        """ dump population state ={imeta,pop,log_stats}
        :param state_pkl: name of pickle file to dump; or None
        :return: state
        """
        # convert EAIndiv to list
        # demes
        if self.demes is None:
            demes_list = None
        else:
            demes_list = [
                [self.imeta.to_list_fitness(i) for i in pop]
                for pop in self.demes
            ]
        # log_best
        if self.log_best is None:
            log_best_list = None
        else:
            log_best_list = [
                [self.imeta.to_list_fitness(i) for i in log_best_pop]
                for log_best_pop in self.log_best
            ]
        
        # collect state
        state = dict(
            imeta_config=self.imeta.get_config(),
            n_pop=self.n_pop,
            n_demes=self.n_demes,
            demes_list=demes_list,
            log_stats=self.log_stats,
            log_best_list=log_best_list
        )
        
        # dump
        if state_pkl is not None:
            with open(state_pkl, "wb") as pkl:
                pickle.dump(state, pkl)
        return state

    def register_map(self, func_map=map):
        """ for applying multiprocessing.dummy.Pool().map from __main__
        """
        self.toolbox.register("map", func_map)

    def init_demes(self):
        """ initialize demes, logbook, evaluate
        """
        # generation demes
        self.demes = [
            [self.imeta.random() for _ in range(self.n_pop)]
            for _ in range(self.n_demes)
        ]

        # evaluate, sort, log stats
        self.evaluate_pop(itertools.chain(*self.demes))
        self.demes = [
            deap.tools.selBest(pop, self.n_pop, fit_attr='fitness')
            for pop in self.demes
        ]

        # log
        self.log_stats = deap.tools.Logbook()
        self.log_best = []
        self.logging_demes(gen=0)

    def evaluate_pop(self, pop):
        """ evaluate population
        :param pop: list of individuals
        :return: None
        :action: assigned fitness to individuals
        """
        # find individuals that are not evaluated
        pop_invalid = [indiv for indiv in pop if not indiv.fitness.valid]
        # evaluate
        fit_invalid = self.toolbox.map(self.toolbox.evaluate, pop_invalid)
        for indiv, fit in zip(pop_invalid, fit_invalid):
            indiv.fitness.values = fit
    
    def logging_demes(self, gen):
        """ log demes stats and best
        :param gen: index of generation
        """
        for i, pop in enumerate(self.demes):
            self.log_stats.record(gen=gen, deme=i, **self.stats.compile(pop))
        self.log_best.append([self.toolbox.clone(pop[0]) for pop in self.demes])

    def evolve_one(self, action):
        """ evolve one generation, perform crossover or mutation
        :param action: select an action, 0=crossover, 1=mutation
        :return: None
        :action: update self.demes, self.log_stats, self.log_best
        """
        # generate offsprings
        offsprings = []
        for pop in self.demes:
            # select for variation, copy, shuffle
            offspring_i = self.toolbox.select(pop, self.n_pop)
            offspring_i = [self.toolbox.clone(i) for i in offspring_i]
            np.random.shuffle(offspring_i)
            
            # crossover
            if action == 0:
                for child1, child2 in zip(offspring_i[::2], offspring_i[1::2]):
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # mutation
            elif action == 1:
                for mutant in offspring_i:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            else:
                raise ValueError("action should be 0(crossover) or 1(mutation)")
            
            offsprings.append(offspring_i)
        
        # update fitness for all offsprings
        self.evaluate_pop(itertools.chain(*offsprings))
        
        # select next generation
        # best of offspring + best of parent
        for i in range(self.n_demes):
            n_half = int(self.n_pop/2)
            parent_best = deap.tools.selBest(self.demes[i], n_half, fit_attr='fitness')
            offspring_best = deap.tools.selBest(offsprings[i], n_half, fit_attr='fitness')
            self.demes[i] = parent_best + offspring_best
        
        # migrate
        self.toolbox.migrate(self.demes)

        # log
        self.logging_demes(gen=(self.log_stats[-1]["gen"] + 1))

    def evolve(self, n_gen, dump_step=None, state_pkl=None, n_proc=None):
        """ evolve n generations, using multithreading
        :param n_gen: number of generations
        :param dump_step, state_pkl: dump into pkl file (state_pkl) at intervals (dump_step)
        :param n_proc: number of processors for multithreading
        :return: None
        :action: update self.demes, self.log_stats, self.log_best
        """
        # setup: pool, map
        pool = multiprocessing.dummy.Pool(n_proc)
        self.register_map(pool.map)

        # run
        for i in range(1, n_gen+1):
            # evolve
            self.evolve_one(action=i % 2)
            # dump
            if (dump_step is not None) and (state_pkl is not None):
                # at intervals or at the last step
                if (n_gen % dump_step == 0) or (i == n_gen-1):
                    self.dump_state(state_pkl)
        
        # clean-up: map, pool
        self.register_map()
        pool.close()

    def get_log_stats(self):
        """ get log_stats in DataFrame
        :return: DataFrame(log_stats)
        """
        df = pd.DataFrame(self.log_stats)
        df = pd.pivot(df,
            index="gen",
            columns="deme",
            values=["mean", "best", "std"]
        )
        return df

    def plot_log_stats(self, xlim=(None, None), save=None):
        """ plot log_stats
        :param xlim: xlim for plotting
        :param save: name of figure file to save
        :return: fig, ax
        """
        # convert to dataframe
        df = self.get_log_stats()
        # plot
        fig, ax = plt.subplots()
        for i in range(self.n_demes):
            ax.plot(df[("best", i)],
                c=f"C{i}", ls="-", label=f"deme {i}: best")
            ax.plot(df[("mean", i)],
                c=f"C{i}", ls="--", label=f"deme {i}: mean")
        ax.legend()
        ax.set(xlabel="generation", ylabel="fitness")
        ax.set(xlim=xlim)
        # save
        if save is not None:
            fig.savefig(save)
        return fig, ax
    
    def fit_surfaces_eval(self, indiv_arr, u_eval=None, v_eval=None):
        """ fit surfaces from individuals, evaluate at net
        :param indiv_arr: array of individuals
        :param u_eval, v_eval: arrays of evaluation points along u, v
        :return: B_arr
            B_arr: [B_sample_0, B_surf_0, B_sample_1, B_surf_1, ...]
        """
        B_arr = []
        for indiv in indiv_arr:
            B_surf, _ = self.imeta.fit_surface_eval(
                indiv, u_eval=u_eval, v_eval=v_eval
            )
            B_sample = coord_to_mask(
                self.imeta.flatten_net(self.imeta.get_coord_net(indiv)),
                self.imeta.shape)
            B_arr.extend([B_sample, B_surf])
        return B_arr
