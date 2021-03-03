#' ------------------------------------------------------------------------
#' ------------------------------------------------------------------------
#' 
#' Bayesian hierarchical modelling of cetacean responses to sound exposure
#' 
#' ------------------------------------------------------------------------
#' ------------------------------------------------------------------------ 
#' 
#' Author: Phil J Bouchet
#' Last update: 2021-01-15
#' R version: 4.0.3
#' --------------------------------------------------------------------

#'--------------------------------------------------------------------
# Load required libraries
#'--------------------------------------------------------------------

# pacman::p_install_gh("dvats/mcmcse", "knudson1/stableGR/stableGR") # Need RTools on Windows

pacman::p_load(rjags,          # Bayesian graphical models using MCMC
               coda,           # Output analysis and diagnostics for MCMC
               MASS,           # Support functions for applied statistics
               truncnorm,      # Truncated Normal
               tidyverse,      # R packages for data science
               lubridate,      # Dates and times made easy
               janitor,        # Simple tools for examining and cleaning dirty data
               parallel,       # Support for parallel computation
               doParallel,     # Parallel backend for the foreach/%dopar% function
               utils,          # R utility functions
               pals,           # Comprehensive palettes and palette evaluation tools
               viridisLite,    # Viridis colour palette
               ArgumentCheck,  # Improved communication re problems in function arguments         
               crayon,         # Colored terminal output
               ggnewscale,     # Multiple colour scales in ggplot2
               MCMCvis,        # Visualise, manipulate, and summarise MCMC output
               HDInterval,     # Highest (Posterior) Density Intervals 
               reshape2,       # Flexibly reshape data
               bayesplot,      # Plotting for Bayesian Models
               gridExtra,      # Miscellaneous functions for "grid" graphics
               ggpubr,         # Tools for publication ready plots
               plyr,           # The split-apply-combine paradigm for R
               rlang,          # Tools for core language features of R and the tidyverse
               ggridges,       # Ridgeline plots with ggplot2
               hms,            # Pretty time of day
               cowplot,        # Add-on for complex ggplot
               grDevices,      # R Graphics Devices and Support for Colours and Fonts
               here,           # Find files
               tictoc)         # Functions for timing
      

#'--------------------------------------------------------------------
# Set the random seed
#'--------------------------------------------------------------------

set.seed(45)

#'---------------------------------------------
# Set tibble options
#'---------------------------------------------

options(tibble.width = Inf) # All tibble columns shown
options(pillar.neg = FALSE) # No colouring negative numbers
options(pillar.subtle = TRUE)
options(pillar.sigfig = 4)

options(dplyr.summarise.inform = FALSE) # Suppress dplyr::summarise warnings

#'--------------------------------------------------------------------
# Load required functions
#'--------------------------------------------------------------------

source(here::here("R/BayesianBR_simulation_functions.R"))

#'-------------------------------------------------
# Run the simulations (example)
#'-------------------------------------------------

mcmc.results <- run_scenario(scenario = 2,
                             n.sim = 2, 
                             n.whales = c(5, 10, 20, 30),
                             uncertainty.dose = c(2.5, 5, 10, 20, 35),
                             prop.sat = c(20, 40, 60, 80, 100),
                             mcmc.auto = FALSE,
                             mcmc.n = 1000,
                             mcmc.save = TRUE,
                             burn.in = 1000,
                             mcmc.thin = 1,
                             mcmc.chains = 3,
                             verbose = TRUE,
                             no.tracePlots = 2,
                             parallel.cores = 10, 
                             check.convergence = TRUE,
                             save.results = TRUE)

#'-------------------------------------------------
# Extract and plot results
#'-------------------------------------------------

plot_doseresponse(mcmc.object = mcmc.results, save.to.disk = TRUE, n.row = 3, n.col = 2, select.n = c(5, 10, 30), select.obs = c(2.5, 35), concatenate = TRUE, plot.height = 4, plot.width = 3, plot.labels = TRUE)

plot_results(mcmc.object = mcmc.results, layout.ncol = 1, save.to.disk = TRUE, save.individual.plots = TRUE, 
             common.scale = TRUE, include.ERR = TRUE, plot.labels = FALSE, pt.size = 5, max_credwidth = 80, max_prb = 50)

#'-------------------------------------------------
# Compile different runs
#'-------------------------------------------------

mcmc.compiled <- compile_sim(scenario = 2)
mcmc.full <- extra_sim(mcmc.object = mcmc.compiled, replace.sims = TRUE, update.dr = TRUE)
