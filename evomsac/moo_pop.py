""" single objective optimization: population
"""

import pickle
import multiprocessing.dummy
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import deap, deap.base, deap.tools

from etsynseg.evomsac import MOOTools

class MOOPop:
    """ evolving populations
    Usage:
        # evolve
        mootools = MOOTools(B, n_vxy, n_uz, nz_eachu, r_thresh)
        moopop = MOOPop(mootools, pop_size)
        moopop.init_pop()
        moopop.evolve(maxiter, (tol, n_back), dump_step, state_pkl, n_proc)
        # dump, load
        state = moopop.dump_state(file_pkl)
        moopop = MOOPop(state=file_pkl)
        # stats and plot
        moopop.plot_logs(save)
        # fit surface and plot
        B_arr = moopop.fit_surfaces_eval([indiv1, indiv2])
        imshow3d(mootools.B, B_arr)
    """
    def __init__(self, mootools=None, state=None, pop_size=4):
        """ init
        :param mootools: MOOTools(); if given, init; if None, init later
        :param state: state (from dump_state); or a pickle file
        :param pop_size: population size, multiples of 4 (required by selTournamentDCD)
        """
        # attributes
        self.mootools = None
        self.toolbox = None
        self.pop = None
        self.log_front = None
        self.log_indicator = None

        # read config
        # from mootools
        if mootools is not None:
            self.init_from_mootools(mootools)
            self.pop_size = pop_size + pop_size%4
        
        # from state or state pickle file
        elif state is not None:
            # if a pickle file, load
            if isinstance(state, str):
                with open(state, "rb") as pkl:
                    state = pickle.load(pkl)
            # load state
            mootools = MOOTools(config=state["mootools_config"])
            self.init_from_mootools(mootools)
            self.pop_size = state["pop_size"]
            if state["pop_list"] is not None:
                self.pop = [self.mootools.from_list_fitness(*p) for p in state["pop_list"]]
                self.pop = self.toolbox.select_best(self.pop, self.pop_size)
            if state["log_front_list"] is not None:
                self.log_front = [
                    [self.mootools.from_list_fitness(*p) for p in front_list]
                    for front_list in state["log_front_list"]
                ]
            if state["log_indicator"] is not None:
                self.log_indicator = state["log_indicator"]
        else:
            raise ValueError("Should provide either mootools or state")

    def init_from_mootools(self, mootools):
        """ initialize tools for evolution algorithm
        :param mootools: MOOTools()
        :return: None
        :action: assign variables mootools, toolbox
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
    
    def register_map(self, func_map=map):
        """ for applying multiprocessing.dummy.Pool().map
        """
        self.toolbox.register("map", func_map)
    
    def dump_state(self, file_pkl=None):
        """ dump population state
        :param file_pkl: name of pickle file to dump; or None
        :return: state
            state: {mootools,pop_size,pop_list,log_front_list,log_indicator}
        """
        # convert MOOIndiv to list
        if self.pop is not None:
            pop_list = [self.mootools.to_list_fitness(i) for i in self.pop]
        else:
            pop_list = None

        # log_front
        if self.log_front is not None:
            log_front_list = [
                [self.mootools.to_list_fitness(i) for i in log_front_pop]
                for log_front_pop in self.log_front
            ]
        else:
            log_front_list = None
        
        # collect state
        state = dict(
            mootools_config=self.mootools.get_config(),
            pop_size=self.pop_size,
            pop_list=pop_list,
            log_front_list=log_front_list,
            log_indicator=self.log_indicator
        )
        if file_pkl is not None:
            with open(file_pkl, "wb") as pkl:
                pickle.dump(state, pkl)
        return state

    def init_pop(self, pop=None, n_proc=None):
        """ initialize population, logbook, evaluate
        :param pop: provide pop, then init does the rest
        """
        # generation population
        if pop is None:
            self.pop = [self.mootools.indiv_random() for _ in range(self.pop_size)]
        else:
            self.pop = pop

        # evaluate
        pool = multiprocessing.dummy.Pool(n_proc)
        self.register_map(pool.map)
        self.evaluate_pop(self.pop)
        self.register_map()
        pool.close()

        # sort
        self.pop = self.toolbox.select_best(self.pop, self.pop_size)

        # log
        self.log_front = []
        self.log_indicator = []

    def calc_hypervolume(self, front):
        """ calculate hypervolume of the front
        :param front: list of indiv, composing the same front
        :return: hypervolume
        """
        # weighted objectives, to maximize
        wobj = np.array([ind.fitness.wvalues for ind in front])
        # hypervolume w.r.t ref=(0, 0)
        hypervolume = deap.tools._hypervolume.hv.hypervolume(wobj, (0, 0))
        return hypervolume
    
    def select_by_hypervolume(self, front):
        """select individual with least contribution to hypervolume
        :param front: list of indiv, composing the same front
        :return: indiv
            indiv: selected indiv
        """
        # calculate hypervolume for front w/o each individual
        hv_arr = []
        for i in range(len(front)):
            hv_i = self.calc_hypervolume(front[:i]+front[i+1:])
            hv_arr.append(hv_i)
        # select one with min value, corresponding to min contribution
        i_min = np.argmin(hv_arr)
        indiv = front[i_min]
        return indiv

    def logging_pop(self, n_back=None):
        """ log front and indicator
        :param n_back: number of previous steps for calculation indicator changes
        :action: update self.log_front, self.log_indicator
        """
        # pareto front
        front = self.toolbox.sort_fronts(self.pop, self.pop_size, first_front_only=True)[0]
        front = sorted(front, key=lambda indiv: indiv.fitness.values[0])  # sort by coverage
        
        # log front
        self.log_front.append([self.toolbox.clone(indiv) for indiv in front])
        
        # log indicators
        coverage = front[0].fitness.values[0]
        # fit_extra = np.min([indiv.fitness.values[1] for indiv in front])
        # hypervolume = self.calc_hypervolume(front)
        if (n_back is None) or (len(self.log_indicator) < n_back):
            change_ratio = np.nan
        else:
            ind_nback = [ind["coverage"] for ind in self.log_indicator[-n_back:]]
            if np.max(ind_nback) <= 0:
                change_ratio = 0
            else:
                change_ratio = 1 - np.mean(ind_nback) / np.max(ind_nback)
        
        indicator = {"coverage": coverage, "change_ratio": change_ratio}
        self.log_indicator.append(indicator)

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

    def evolve_one_gen(self, variation):
        """ evolve one generation, perform crossover or mutation
        :param action: select an action, 0=crossover, 1=mutation
        :return: None
        :action: update self.pop, self.log_stats, self.log_front
        """
        # select for variation, copy, shuffle
        offspring = self.toolbox.select_var(self.pop, self.pop_size)
        offspring = [self.toolbox.clone(i) for i in offspring]

        # crossover
        if variation == 0:
            np.random.shuffle(offspring)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutation
        elif variation == 1:
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        else:
            raise ValueError("action should be 0(crossover) or 1(mutation)")
        
        # update fitness
        self.evaluate_pop(offspring)
        
        # select next generation
        self.pop = self.toolbox.select_best(self.pop+offspring, self.pop_size)

    def evolve(self, var_cycle=(0, 1), tol=(0.01, 10), max_iter=200,
            step_dump=None, file_pkl=None, n_proc=None
        ):
        """ evolve using multithreading
        :param var_cycle: variations to cycle, 0=crossover, 1=mutate
        :param tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
        :param max_iter: max number of generations
        :param step_dump, file_pkl: dump into file_pkl at step_dump intervals
        :param n_proc: number of processors for multithreading
        :return: None
        :action: update self.pop, self.log_front, self.log_indicator
        """
        # setup: pool, map
        pool = multiprocessing.dummy.Pool(n_proc)
        self.register_map(pool.map)

        # run
        for i, var in zip(range(max_iter), itertools.cycle(var_cycle)):
            # evolve
            self.evolve_one_gen(variation=var)
            self.logging_pop(n_back=tol[1])
            
            # judge termination
            if self.log_indicator[-1]["change_ratio"] < tol[0]:
                break

            # dump
            if (step_dump is not None) and (file_pkl is not None):
                # at intervals or at the last step
                if i % step_dump == 0:
                    self.dump_state(file_pkl)
        
        # final dump
        if file_pkl is not None:
            self.dump_state(file_pkl)

        # clean-up: map, pool
        self.register_map()
        pool.close()
    
    def plot_logs(self, log_front=None, log_indicator=None,
            moo_labels=("coverage", "fit extra"), save=None):
        """ plot pareto fronts
        :param log_front: log of fronts, use self.log_front if None
        :return: fig, ax
        """
        # set default
        if log_front is None:
            log_front = self.log_front
        if log_indicator is None:
            log_indicator = self.log_indicator

        fig, axes = plt.subplots(
            ncols=2, constrained_layout=True
        )

        # plot fronts
        # configure colormap
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(log_front)-1)
        cmapper = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=matplotlib.cm.viridis)
        # plot each front
        for i, best in enumerate(log_front):
            front = np.array([indiv.fitness.values for indiv in best])
            axes[0].plot(*np.array(front).T, marker="o", alpha=0.5, c=cmapper.to_rgba(i))
        axes[0].set(xlabel=f"fitness: {moo_labels[0]}", ylabel=f"fitness: {moo_labels[1]}")

        # plot indicators
        # indicator
        axes[1].plot([ind["coverage"] for ind in log_indicator], c="C0")
        axes[1].set(xlabel="generation", ylabel="coverage")
        axes[1].tick_params(axis='y', labelcolor="C0")
        # change ratio
        axes1_twin = axes[1].twinx()
        axes1_twin.plot([ind["change_ratio"] for ind in log_indicator], c="C1")
        axes1_twin.set(xlabel="generation", ylabel="change_ratio")
        axes1_twin.tick_params(axis='y', labelcolor="C1")

        # save
        if save is not None:
            fig.savefig(save)
        return fig, axes
    
    def fit_surfaces_eval(self, pop, u_eval=None, v_eval=None):
        """ fit surfaces from individuals, evaluate at net
        :param pop: array of individuals
        :param u_eval, v_eval: arrays of evaluation points along u, v
        :return: zyx_arr
            zyx_arr: [zyx_sample_0, zyx_surf_0, zyx_sample_1, zyx_surf_1, ...]
        """
        zyx_arr = []
        for indiv in pop:
            zyx_surf, _ = self.mootools.fit_surface_eval(
                indiv, u_eval=u_eval, v_eval=v_eval
            )
            zyx_sample = self.mootools.flatten_net(self.mootools.get_coord_net(indiv))
            zyx_arr.extend([zyx_sample, zyx_surf])
        return zyx_arr
