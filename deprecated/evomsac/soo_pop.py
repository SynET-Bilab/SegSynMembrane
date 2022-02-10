# """ single objective optimization: population
# """

# import pickle
# import multiprocessing.dummy
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import deap, deap.base, deap.tools

# from etsynseg.utils import coord_to_mask
# from etsynseg.evomsac import SOOTools

# class SOOPop:
#     """ evolving populations
#     Usage:
#         # evolve
#         sootools = SOOTools(B, n_vxy, n_uz, nz_eachu, r_thresh)
#         soopop = SOOPop(sootools, n_pop)
#         soopop.init_pop()
#         soopop.evolve(n_gen, dump_step, state_pkl, n_proc)
#         # dump
#         soopop.dump_state(state_pkl)
#         # load
#         soopop = SOOPop(state_pkl=state_pkl)
#         # stats and plot
#         df_stats = soopop.get_log_stats()
#         soopop.plot_log_stats(xlim, save)
#         # surface and plot
#         B_arr = get_surfaces([indiv1, indiv2])
#         imshow3d(sootools.B, B_arr)
#     """
#     def __init__(self, sootools=None, state=None, n_pop=2):
#         """ init
#         :param sootools: SOOTools(); if given, init; if None, init later
#         :param state: state (from dump_state); or a pickle file
#         :param n_pop: population size, multiples of 2
#         """
#         # attributes
#         self.sootools = None
#         self.toolbox = None
#         self.stats = None
#         self.pop = None
#         self.log_stats = None
#         self.log_best = None

#         # read config
#         # from sootools
#         if sootools is not None:
#             self.init_from_sootools(sootools)
#             self.n_pop = n_pop + n_pop%2
        
#         # from state or state pickle file
#         elif state is not None:
#             # if a pickle file, load
#             if isinstance(state, str):
#                 with open(state, "rb") as pkl:
#                     state = pickle.load(pkl)
#             # load state
#             sootools = SOOTools(config=state["sootools_config"])
#             self.init_from_sootools(sootools)
#             self.n_pop = state["n_pop"]
#             if state["pop_list"] is not None:
#                 self.pop = [self.sootools.from_list_fitness(*p) for p in state["pop_list"]]
#             if state["log_stats"] is not None:
#                 self.log_stats = state["log_stats"]
#             if state["log_best_list"] is not None:
#                 self.log_best = [self.sootools.from_list_fitness(*p) for p in state["log_best_list"]]
#         else:
#             raise ValueError("Should provide either sootools or state")

#     def init_from_sootools(self, sootools):
#         """ initialize tools for evolution algorithm
#         :param sootools: SOOTools()
#         :return: None
#         :action: assign variables sootools, toolbox, stats
#         """
#         # setup meta
#         self.sootools = sootools
#         self.toolbox = deap.base.Toolbox()

#         # operations
#         self.toolbox.register("evaluate", self.sootools.evaluate)
#         self.toolbox.register("select", deap.tools.selTournament, tournsize=2)
#         self.toolbox.register("mate", deap.tools.cxTwoPoint)
#         self.toolbox.register("mutate", self.sootools.mutate)

#         # stats
#         self.stats = deap.tools.Statistics(
#             key=lambda indiv: indiv.fitness.values)
#         self.stats.register('mean', np.mean)
#         self.stats.register('best', np.min)
#         self.stats.register('std', np.std)
    
#     def dump_state(self, state_pkl=None):
#         """ dump population state ={sootools,pop,log_stats}
#         :param state_pkl: name of pickle file to dump; or None
#         :return: state
#         """
#         # convert SOOIndiv to list
#         if self.pop is not None:
#             pop_list = [self.mootools.to_list_fitness(i) for i in self.pop]
#         else:
#             pop_list = None

#         if self.log_best is not None:
#             log_best_list = [self.mootools.to_list_fitness(i) for i in self.log_best]
#         else:
#             log_best_list = None

#         # collect state
#         state = dict(
#             sootools_config=self.sootools.get_config(),
#             n_pop=self.n_pop,
#             pop_list=pop_list,
#             log_stats=self.log_stats,
#             log_best_list=log_best_list
#         )
#         if state_pkl is not None:
#             with open(state_pkl, "wb") as pkl:
#                 pickle.dump(state, pkl)
#         return state

#     def register_map(self, func_map=map):
#         """ for applying multiprocessing.dummy.Pool().map from __main__
#         """
#         self.toolbox.register("map", func_map)

#     def init_pop(self):
#         """ initialize population, logbook, evaluate
#         """
#         # generation population
#         self.pop = [self.sootools.random() for _ in range(self.n_pop)]

#         # evaluate, sort, log stats
#         self.log_stats = deap.tools.Logbook()
#         self.evaluate_pop(self.pop)
#         self.pop = deap.tools.selBest(self.pop, self.n_pop, fit_attr='fitness')
#         self.log_stats.record(gen=0, **self.stats.compile(self.pop))
        
#         # log best individual
#         self.log_best = [self.toolbox.clone(self.pop[0])]

#     def evaluate_pop(self, pop):
#         """ evaluate population
#         :param pop: list of individuals
#         :return: None
#         :action: assigned fitness to individuals
#         """
#         # find individuals that are not evaluated
#         pop_invalid = [indiv for indiv in pop if not indiv.fitness.valid]
#         # evaluate
#         fit_invalid = self.toolbox.map(self.toolbox.evaluate, pop_invalid)
#         for indiv, fit in zip(pop_invalid, fit_invalid):
#             indiv.fitness.values = fit

#     def evolve_one(self, action):
#         """ evolve one generation, perform crossover or mutation
#         :param action: select an action, 0=crossover, 1=mutation
#         :return: None
#         :action: update self.pop, self.log_stats, self.log_best
#         """
#         # select for variation, copy, shuffle
#         offspring = self.toolbox.select(self.pop, self.n_pop)
#         offspring = [self.toolbox.clone(i) for i in offspring]
#         np.random.shuffle(offspring)
        
#         # crossover
#         if action == 0:
#             for child1, child2 in zip(offspring[::2], offspring[1::2]):
#                 self.toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
#         # mutation
#         elif action == 1:
#             for mutant in offspring:
#                 self.toolbox.mutate(mutant)
#                 del mutant.fitness.values
#         else:
#             raise ValueError("action should be 0(crossover) or 1(mutation)")
        
#         # update fitness
#         self.evaluate_pop(offspring)
        
#         # select next generation: best of offspring + best of parent
#         n_half = int(self.n_pop/2)
#         parent_best = deap.tools.selBest(self.pop, n_half, fit_attr='fitness')
#         offspring_best = deap.tools.selBest(offspring, n_half, fit_attr='fitness')
#         self.pop = parent_best + offspring_best

#         # log
#         gen = self.log_stats[-1]["gen"] + 1
#         self.log_stats.record(gen=gen, **self.stats.compile(self.pop))
#         self.log_best.append(self.toolbox.clone(self.pop[0]))

#     def evolve(self, n_gen, dump_step=None, state_pkl=None, n_proc=None):
#         """ evolve n generations, using multithreading
#         :param n_gen: number of generations
#         :param dump_step, state_pkl: dump into pkl file (state_pkl) at intervals (dump_step)
#         :param n_proc: number of processors for multithreading
#         :return: None
#         :action: update self.pop, self.log_stats, self.log_best
#         """
#         # setup: pool, map
#         pool = multiprocessing.dummy.Pool(n_proc)
#         self.register_map(pool.map)

#         # run
#         for i in range(1, n_gen+1):
#             # evolve
#             self.evolve_one(action=i % 2)
#             # dump
#             if (dump_step is not None) and (state_pkl is not None):
#                 # at intervals or at the last step
#                 if (n_gen % dump_step == 0) or (i == n_gen-1):
#                     self.dump_state(state_pkl)
        
#         # clean-up: map, pool
#         self.register_map()
#         pool.close()

#     def get_log_stats(self):
#         """ get log_stats in DataFrame
#         :return: DataFrame(log_stats)
#         """
#         return pd.DataFrame(self.log_stats)

#     def plot_log_stats(self, xlim=(None, None), save=None):
#         """ plot log_stats
#         :param xlim: xlim for plotting
#         :param save: name of figure file to save
#         :return: fig, ax
#         """
#         # convert to dataframe
#         df = self.get_log_stats()
#         # plot
#         fig, ax = plt.subplots()
#         ax.plot(df["best"], label="best")
#         ax.plot(df["mean"], label="mean")
#         ax.legend()
#         ax.set(xlabel="generation", ylabel="fitness")
#         ax.set(xlim=xlim)
#         # save
#         if save is not None:
#             fig.savefig(save)
#         return fig, ax
    
#     def fit_surfaces_eval(self, indiv_arr, u_eval=None, v_eval=None):
#         """ fit surfaces from individuals, evaluate at net
#         :param indiv_arr: array of individuals
#         :param u_eval, v_eval: arrays of evaluation points along u, v
#         :return: B_arr
#             B_arr: [B_sample_0, B_surf_0, B_sample_1, B_surf_1, ...]
#         """
#         B_arr = []
#         for indiv in indiv_arr:
#             B_surf, _ = self.sootools.fit_surface_eval(
#                 indiv, u_eval=u_eval, v_eval=v_eval
#             )
#             B_sample = coord_to_mask(
#                 self.sootools.flatten_net(self.sootools.get_coord_net(indiv)),
#                 self.sootools.shape)
#             B_arr.extend([B_sample, B_surf])
#         return B_arr
