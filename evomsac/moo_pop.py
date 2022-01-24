#!/usr/bin/env python
""" single objective optimization: population
"""

import pickle
import multiprocessing.dummy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import deap, deap.base, deap.tools

from synseg.utils import coord_to_mask
from synseg.evomsac import MOOTools

class MOOPop:
    """ evolving populations
    Usage:
        # evolve
        mootools = MOOTools(B, n_vxy, n_uz, nz_eachu, r_thresh)
        moopop = MOOPop(mootools, n_pop)
        moopop.init_pop()
        moopop.evolve(n_gen, dump_step, state_pkl, n_proc)
        # dump
        moopop.dump_state(state_pkl)
        # load
        moopop = MOOPop(state_pkl=state_pkl)
        # stats and plot
        moopop.plot_log_stats(save)
        moopop.plot_log_best(save)
        # surface and plot
        B_arr = get_surfaces([indiv1, indiv2])
        imshow3d(mootools.B, B_arr)
    """
    def __init__(self, mootools=None, state=None, n_pop=2):
        """ init
        :param mootools: MOOTools(); if given, init; if None, init later
        :param state: state (from dump_state); or a pickle file
        :param n_pop: population size, multiples of 4 (required by selTournamentDCD)
        """
        # attributes
        self.mootools = None
        self.toolbox = None
        self.stats = None
        self.pop = None
        self.log_stats = None
        self.log_best = None

        # read config
        # from mootools
        if mootools is not None:
            self.init_from_mootools(mootools)
            self.n_pop = n_pop + n_pop%4
        
        # from state or state pickle file
        elif state is not None:
            # if a pickle file, load
            if isinstance(state, str):
                with open(state, "rb") as pkl:
                    state = pickle.load(pkl)
            # load state
            mootools = MOOTools(config=state["mootools_config"])
            self.init_from_mootools(mootools)
            self.n_pop = state["n_pop"]
            if state["pop_list"] is not None:
                self.pop = [self.mootools.from_list_fitness(*p) for p in state["pop_list"]]
                self.pop = self.toolbox.select_best(self.pop, self.n_pop)
            if state["log_stats"] is not None:
                self.log_stats = state["log_stats"]
            if state["log_best_list"] is not None:
                self.log_best = [
                    [self.mootools.from_list_fitness(*p) for p in front_list]
                    for front_list in state["log_best_list"]
                ]
        else:
            raise ValueError("Should provide either mootools or state")

    def init_from_mootools(self, mootools):
        """ initialize tools for evolution algorithm
        :param mootools: MOOTools()
        :return: None
        :action: assign variables mootools, toolbox, stats
        """
        # setup meta
        self.mootools = mootools
        self.toolbox = deap.base.Toolbox()

        # operations
        self.toolbox.register("evaluate", self.mootools.evaluate)
        self.toolbox.register("mate", deap.tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mootools.mutate)

        # multi-objective
        # select for variation
        self.toolbox.register("select_var", deap.tools.selTournamentDCD)
        # select best
        self.toolbox.register("select_best", deap.tools.selNSGA2, nd='standard')
        # sort
        self.toolbox.register("sort_fronts", deap.tools.sortNondominated)

        # stats
        self.stats = deap.tools.Statistics(
            key=lambda indiv: indiv.fitness.values)
        self.stats.register('mean', np.mean, axis=0)
        self.stats.register('best', np.min, axis=0)
        self.stats.register('std', np.std, axis=0)
    
    def dump_state(self, state_pkl=None):
        """ dump population state ={mootools,pop,log_stats}
        :param state_pkl: name of pickle file to dump; or None
        :return: state
        """
        # convert MOOIndiv to list
        if self.pop is not None:
            pop_list = [self.mootools.to_list_fitness(i) for i in self.pop]
        else:
            pop_list = None

        # log_best
        if self.log_best is not None:
            log_best_list = [
                [self.mootools.to_list_fitness(i) for i in log_best_pop]
                for log_best_pop in self.log_best
            ]
        else:
            log_best_list = None
        
        # collect state
        state = dict(
            mootools_config=self.mootools.get_config(),
            n_pop=self.n_pop,
            pop_list=pop_list,
            log_stats=self.log_stats,
            log_best_list=log_best_list
        )
        if state_pkl is not None:
            with open(state_pkl, "wb") as pkl:
                pickle.dump(state, pkl)
        return state

    def register_map(self, func_map=map):
        """ for applying multiprocessing.dummy.Pool().map from __main__
        """
        self.toolbox.register("map", func_map)

    def init_pop(self, n_proc=None):
        """ initialize population, logbook, evaluate
        """
        # generation population
        self.pop = [self.mootools.random() for _ in range(self.n_pop)]

        # evaluate
        pool = multiprocessing.dummy.Pool(n_proc)
        self.register_map(pool.map)
        self.evaluate_pop(self.pop)
        self.register_map()
        pool.close()

        # sort
        self.pop = self.toolbox.select_best(self.pop, self.n_pop)

        # log
        self.log_stats = deap.tools.Logbook()
        self.log_best = []
        self.logging_pop(gen=0)
    
    def logging_pop(self, gen):
        """ log pop stats and best
        :param gen: index of generation
        """
        # stats
        stats_dict = self.stats.compile(self.pop)
        for i in range(2):
            self.log_stats.record(gen=gen, fitness=i,
                **{k: v[i] for k, v in stats_dict.items()}
            )

        # pareto front
        front = self.toolbox.sort_fronts(self.pop, self.n_pop, first_front_only=True)[0]
        front = sorted(front, key=lambda indiv: indiv.fitness.values[0])  # sort by overlap
        self.log_best.append([self.toolbox.clone(indiv) for indiv in front])

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

    def evolve_one(self, action):
        """ evolve one generation, perform crossover or mutation
        :param action: select an action, 0=crossover, 1=mutation
        :return: None
        :action: update self.pop, self.log_stats, self.log_best
        """
        # select for variation, copy, shuffle
        offspring = self.toolbox.select_var(self.pop, self.n_pop)
        offspring = [self.toolbox.clone(i) for i in offspring]

        # crossover
        if action == 0:
            np.random.shuffle(offspring)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutation
        elif action == 1:
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        else:
            raise ValueError("action should be 0(crossover) or 1(mutation)")
        
        # update fitness
        self.evaluate_pop(offspring)
        
        # select next generation
        self.pop = self.toolbox.select_best(self.pop+offspring, self.n_pop)

        # log
        self.logging_pop(gen=self.log_stats[-1]["gen"]+1)

    def evolve(self, n_gen, dump_step=None, state_pkl=None, n_proc=None):
        """ evolve n generations, using multithreading
        :param n_gen: number of generations
        :param dump_step, state_pkl: dump into pkl file (state_pkl) at intervals (dump_step)
        :param n_proc: number of processors for multithreading
        :return: None
        :action: update self.pop, self.log_stats, self.log_best
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
            columns="fitness",
            values=["mean", "best", "std"]
        )
        return df

    def plot_log_stats(self, moo_labels=("overlap", "fit extra"), save=None):
        """ plot log_stats
        :param moo_labels: names of moo fitness
        :param xlim: xlim for plotting
        :param save: name of figure file to save
        :return: fig, ax
        """
        # convert to dataframe
        df = self.get_log_stats()
        # plot
        fig, axes = plt.subplots(ncols=2, constrained_layout=True)
        for i in range(2):
            axes[i].plot(df[("best", i)], label="best")
            axes[i].plot(df[("mean", i)], label="mean")
            axes[i].set(xlabel="generation", ylabel="fitness")
            axes[i].set(title=moo_labels[i])
            axes[i].set_xticks(np.linspace(0, len(df)-1, 6, dtype=int))
        # save
        if save is not None:
            fig.savefig(save)
        return fig, axes
    
    def plot_log_best(self, log_best=None, moo_labels=("overlap", "fit extra"), save=None):
        """ plot pareto fronts
        :param log_best: log of fronts, use self.log_best if None
        :return: fig, ax
        """
        # get log_best
        if log_best is None:
            log_best = self.log_best

        # configure colormap
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(log_best)-1)
        cmapper = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=matplotlib.cm.viridis)

        # plot fronts
        fig, ax = plt.subplots()
        for i, best in enumerate(log_best):
            front = np.array([indiv.fitness.values for indiv in best])
            ax.plot(*np.array(front).T, marker="o", alpha=0.5, c=cmapper.to_rgba(i))
        ax.set(xlabel=f"fitness: {moo_labels[0]}", ylabel=f"fitness: {moo_labels[1]}")
        # save
        if save is not None:
            fig.savefig(save)
        return fig, ax
    
    def fit_surfaces_eval(self, pop, u_eval=None, v_eval=None):
        """ fit surfaces from individuals, evaluate at net
        :param pop: array of individuals
        :param u_eval, v_eval: arrays of evaluation points along u, v
        :return: B_arr
            B_arr: [B_sample_0, B_surf_0, B_sample_1, B_surf_1, ...]
        """
        B_arr = []
        for indiv in pop:
            B_surf, _ = self.mootools.fit_surface_eval(
                indiv, u_eval=u_eval, v_eval=v_eval
            )
            B_sample = coord_to_mask(
                self.mootools.flatten_net(self.mootools.get_coord_net(indiv)),
                self.mootools.shape)
            B_arr.extend([B_sample, B_surf])
        return B_arr
