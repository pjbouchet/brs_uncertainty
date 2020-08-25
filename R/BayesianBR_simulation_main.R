#' ------------------------------------------------------------------------
#' ------------------------------------------------------------------------
#' 
#' Bayesian hierarchical modelling of cetacean responses to sound exposure
#' 
#' ------------------------------------------------------------------------
#' ------------------------------------------------------------------------ 
#' 
#' Author: Phil Bouchet
#' Last update: 2020-04-07
#' Project: LMR Dose-Response
#' R version: 3.6.0 "Planting of a Tree"
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
               tictoc)         # Functions for timing
      

#'--------------------------------------------------------------------
# Set the random seed
#'--------------------------------------------------------------------

set.seed(45)

#'--------------------------------------------------------------------
# Set the working directory
#'--------------------------------------------------------------------

setwd("/Users/philippebouchet/Google Drive/Documents/postdoc/creem/dMOCHA/dose_response/simulation/dose_uncertainty")

#'---------------------------------------------
# Set tibble options
#'---------------------------------------------

options(tibble.width = Inf) # All tibble columns shown
options(pillar.neg = FALSE) # No colouring negative numbers
options(pillar.subtle = TRUE)
options(pillar.sigfig = 4)

#'--------------------------------------------------------------------
# Load required functions
#'--------------------------------------------------------------------

source('/Users/philippebouchet/Google Drive/Documents/git/doublemocha_sim/R/BayesianBR_simulation_functions.R')

#'-------------------------------------------------
# Run the simulations (parallel)
#'-------------------------------------------------

mcmc.iter12 <- read.table('iter/BayesianBRModel_iter_12.txt', header = TRUE)
mcmc.iter34 <- read.table('iter/BayesianBRModel_iter_34.txt', header = TRUE)

mcmc.results <- run_scenario(scenario = 1,
                             n.sim = 2, 
                             n.whales = c(5, 10, 15, 20, 40),
                             uncertainty.dose = c(2.5, 5, 10, 15, 20, 25, 30, 35),
                             mcmc.auto = TRUE,
                             mcmc.n = 10000,
                             mcmc.save = TRUE,
                             burn.in = 5000,
                             mcmc.thin = 1,
                             mcmc.chains = 3,
                             verbose = TRUE,
                             no.tracePlots = 2,
                             parallel.cores = 2, 
                             check.convergence = TRUE,
                             save.results = TRUE)

#'-------------------------------------------------
# Extract, update, and plot results
#'-------------------------------------------------

mcmc.s1 <- compile_sim(scenario = 1)
mcmc.rs1 <- extra_sim(mcmc.object = mcmc.s1, replace.sims = TRUE, update.dr = TRUE)

plot_results(mcmc.object = mcmc.rs1, layout.ncol = 1, save.to.disk = TRUE)
plot_results(mcmc.object = mcmc.s1, layout.ncol = 1, save.to.disk = TRUE, select.n = c(5, 10, 20, 40), select.obs = c(2.5,5,10,20,35), save.individual.plots = TRUE, common.scale = TRUE)
plot_results(mcmc.object = mcmc.s3, layout.ncol = 1, save.to.disk = TRUE, select.n = c(5, 10, 20, 40), select.obs = c(20,40,60,80,100), save.individual.plots = TRUE, common.scale = TRUE)
plot_doseresponse(mcmc.object = mcmc.rs1, save.to.disk = TRUE, n.row = 3, n.col = 3)
