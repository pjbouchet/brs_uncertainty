#'--------------------------------------------------------------------
# Function to conduct Bayesian simulations based on scenarios 1-4
#'--------------------------------------------------------------------
# Simulation plan https://docs.google.com/document/d/1p93JOeD1EL9qvp9swjtdXgLtP76MSNw6aKDf4F-fOsM/edit?usp=sharing

run_scenario <- function(
  
  # Scenario
  
  scenario,
  n.sim, 
  n.whales, 
  n.trials.per.whale = 3, 
  uncertainty.dose = NULL,
  prop.sat = NULL,
  dtag.sd = 2.5,
  source.level = 210,
  species.argos = "Zc",
  
  # Parameters
  
  true.mu = 150, # Moretti et al. 2014 
  lower.bound = 60, # Schick et al. 2019
  upper.bound = 210,
  true.omega = 30,
  omega.upper.bound = 40,
  true.phi = 20,
  phi.upper.bound = 30,
  true.sigma = 25,
  sigma.upper.bound = 30,
  true.beta = 20, # Magnitude of difference between MFAS/LFAS
  beta.sd = 10,
  true.alpha = 8,
  alpha.sd = 10,
  censor.right = c(190, 200),
  
  # MCMC
  
  burn.in = 5000, 
  mcmc.adapt = 1000,
  mcmc.chains = 3, 
  mcmc.thin = 1,
  mcmc.n = 10000,
  mcmc.auto = TRUE,
  mcmc.gelmanrubin = 1.1,
  mcmc.save = FALSE,
  
  # ERR
  
  animal.density = 1,
  n.bins = 500,
  
  # Model diagnostics
  
  check.convergence = TRUE,
  gr.multivariate = FALSE,
  posterior.checks = FALSE,
  min.ESS = 1000,
  correlation.threshold = 0.2,
  no.tracePlots = 1,
  save.trace = TRUE,
  n.yrep = 50,
  hdi.value = 0.95,

  # General

  verbose = TRUE,
  parallel.cores = 1,
  record.time = TRUE,
  save.textsummary = TRUE,
  save.results = TRUE){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param scenario Scenario ID. Must be an integer between 1 and 4.
  #' @param n.sim Integer. Number of simulations to run. 
  #' @param n.whales Integer. Number of whales. Can be supplied as a single value or vector of integers, which will be assessed iteratively.
  #' @param n.trials.per.whale Integer. Number of exposure sessions per animal.
  #' @param uncertainty.dose Uncertainty in acoustic dose, expressed as a standard deviation in received sound levels (in dB). Can be supplied as a single value, or vector of integers, which will be assessed iteratively. Relevant only to scenarios 1 and 2.
  #' @param prop.sat Proportion of animals fitted with satellite tags. Can be supplied as a single value, or vector of integers, which will be assessed iteratively. Relevant only to scenarios 3 and 4.
  #' @param dtag.sd Uncertainty in dose measurements made on DTAGs within scenarios 3 and 4, expressed as a standard deviation in received sound levels (in dB). Defaults to 2.5 dB.
  #' @param source.level Source level of sonar. Defaults to 210 dB re 1μPa m at a nominal frequency of 3 kHz, as per Tyack & Thomas (2019).
  #' @param species.argos Cetacean species to simulate in scenarios 3 and 4. One of either 'Zc' for Cuvier's beaked whale (Ziphius cavirostris) or 'Gm' for short-finned pilot whale (Globicephala macrorhynchus). ARGOS ellipses are simulated for each species based on real data from tagged animals (courtesy of Rob Schick at Duke University).  
  #' @param true.mu Mean response threshold for all whales. Follows a truncated Normal with lower and upper truncation points given by \code{lower.bound} and \code{upper.bound}.
  #' @param lower.bound Lower bound for \code{true.mu}, i.e. minimum response threshold. Defaults to 60 dB, under the conservative assumption that any sound below this threshold will not be heard for a 'typical' odontocete above ambient noise (Schick et al. 2019).
  #' @param upper.bound Upper bound for \code{true.mu}, i.e. threshold at which all whales are expected to respond. Defaults to 190 dB for a typical odontocete.
  #' @param true.omega Combined (between and within-whale) variation in response threshold (SD). Relevant to scenarios 1 and 3.
  #' @param omega.upper.bound Upper bound for \code{true.omega}.
  #' @param true.phi Between-whale variation (SD). Defaults to 20 dB. Relevant to scenarios 2 and 4.
  #' @param phi.upper.bound Upper bound for \code{true.phi}.
  #' @param true.sigma Within-whale variation (SD). Relevant to scenarios 2 and 4.
  #' @param sigma.upper.bound Upper bound for \code{true.sigma}.
  #' @param true.beta Parameter governing the effect of MFAS (Mid-frequency active sonar) relative to LFAS (Low-frequency active sonar) on the animals' response threshold. Follows a Normal distribution with standard deviation \code{beta.sd}.
  #' @param beta.sd Standard deviation of the MFAS effect (\code{true.beta}). 
  #' @param true.alpha Parameter representing the effect of previous exposure to sonar on the animals' response threshold. Follows a Normal distribution with standard deviation given by alpha.sd.
  #' @param alpha.sd Standard deviation of the exposure effect (true.alpha).
  #' @param censor.right Right-censoring range. Must be a vector of two values representing the lower and upper bounds, respectively. In each simulation, a value is drawn from a Uniform distribution bounded by this range, and any observation above this threshold is set to NA.
  #' @param burn.in Number of Markov Chain Monte Carlo (MCMC) iterations required to achieve chain convergence. These will be discarded (burn-in). When \code{mcmc.auto} is set to \code{TRUE}, an initial burn-in of 5,000 is applied, and the \code{autorun.jags} function triggered, ensuring that all chains converge before they are returned.
  #' @param mcmc.adapt Number of iterations used for adaptation. When a JAGS model is compiled, it may require an initial sampling phase during which the samplers adapt their behaviour to maximize their efficiency (e.g. a Metropolis-Hastings random walk algorithm may change its step size). The sequence of samples generated during this adaptive phase is not a Markov chain, and therefore may not be used for posterior inference on the model.
  #' @param mcmc.chains Number of MCMC chains. Defaults to 3.
  #' @param mcmc.thin Thinning rate. Thinning is used to reduce autocorrelation in the MCMC posterior sample, by only retaining every mcmc.thin^{th} value.
  #' @param mcmc.n Integer. Number of MCMC samples extracted after convergence.    
  #' @param mcmc.auto Logical. Run the MCMC model in JAGS with automatically run length and convergence diagnostics, using the \code{runjags} package.
  #' @param mcmc.gelmanrubin Threshold for assessing chain convergence according to the Gelman-Rubin statistic. Defaults to 1.1. 
  #' @param mcmc.save Logical. Whether to save a copy of the MCMC samples on disk. Note that this may results in large files.
  #' @param animal.density Number of whales per km2. Used in the calculation of the effective response range (ERR). Defaults to 1.
  #' @param n.bins Number of bins used in the calculation of the effective response range (ERR). Defaults to 500.
  #' @param check.convergence Logical. Whether to perform convergence checks on MCMC chains, including calculating values of the Gelman-Rubin statistic and producing trace plots. Defaults to TRUE.
  #' @param gr.multivariate Logical. If TRUE, the multivariate Gelman-Rubin statistic is returned.
  #' @param posterior.checks Logical. Whether to perform posterior checks, including PPO: Prior posterior overlap; PPC: Posterior predictive checks. Defaults to TRUE.
  #' @param min.ESS Minimum effective sample size (ESS) considered to be acceptable. A warning will be issued if this value is not attained for each parameter. Defaults to 1000.
  #' @param correlation.threshold Threshold for assessingMCMC chain correlation. If this value is exceeded for any lag, a warning will be triggered.
  #' @param no.tracePlots Integer between 1 and 5. Number of simulations (chosen at random) for which to produce trace plots when \code{check.convergence} is \code{TRUE}.
  #' @param n.yrep Number of replicate datasets to generate for posterior predictive checking.
  #' @param hdi.value Probability mass to include in credible intervals. Defaults to 0.95 for 95% HDI.
  #' @param verbose Logical. Whether to prints an overview of simulation parameters/steps to the R console during execution. Defaults to TRUE.
  #' @param parallel.cores Integer. Number of cores to use when computations are run in parallel. Default is 1 for single core computing.
  #' @param record.time Logical. Whether to time the simulations. Defaults to TRUE.
  #' @param save.results Logical. Whether to save the simulations outputs as an .rds file on disk. Defaults to TRUE.
  #'---------------------------------------------

# Start up ----------------------------------------------------------------

  if(record.time) tictoc::tic() # Begin stopwatch
  
  #'-------------------------------------------------
  # Text-based simulation summary
  #'-------------------------------------------------
  
  if(verbose) {
    
    cat(paste0("\n========================================================\n"))
    cat(paste0("BAYESIAN DOSE-RESPONSE MODELS: SIMULATIONS\n"))
    cat(paste0("========================================================\n"))
    
    cat(paste0("\nScenario: ", scenario, "\n"))
    
    cat(crayon::bold(paste0("\nParameters:\n")))
    cat(crayon::bold(paste0("-------------------------------------\n")))
    
    cat(paste0("- Number of whales: ", paste0(paste0(n.whales), collapse = " / "), "\n"))
    if(scenario%in%c(1,2))
      cat(paste0("- Uncertainty around dose (SD): ", paste0(paste0(uncertainty.dose, " dB "), collapse = "/ "), "\n"))
    if(scenario%in%c(3,4)){
      cat(paste0("- Proportion of SAT tags: ", paste0(paste0(prop.sat, "% "), collapse = "/ "), "\n"))
      cat(paste0("- Species tagged: ", paste0(ifelse(species.argos=="Zc", "Ziphius cavirostris", "Globicephala macrorhynchus")), "\n"))}
    
    cat(paste0("- Mean response threshold (all whales): ", true.mu, " dB (+/- ", true.omega, " SD)\n"))
    cat(paste0("- Minimum response threshold (lower limit of hearing): ", lower.bound, " dB\n"))
    cat(paste0("- Upper response threshold (response from all animals): ", upper.bound, " dB\n"))
    
    if(scenario%in%c(1,3)) cat(paste0("- Overall variation (between + within-whale): ", true.omega, " dB "), "\n")
    
    if(scenario%in%c(2, 4)){
      cat(paste0("- Between-whale variation: ", true.phi, " dB [0,", phi.upper.bound, "]\n"))
      cat(paste0("- Within-whale variation: ", true.sigma, " dB [0,", sigma.upper.bound, "]\n"))} 
    
    if(scenario%in%c(2, 4)){cat(paste0("- MFAS effect: ", true.beta/2, " dB (+/- ", beta.sd, " SD)\n"))} 
    if(scenario%in%c(2, 4)){cat(paste0("- Exposure effect: ", true.alpha, " dB (+/- ", alpha.sd, " SD)\n"))} 
    
    cat(crayon::bold(paste0("\nMCMC sampling:\n")))
    cat(crayon::bold(paste0("-------------------------------------\n")))
    
    if(!mcmc.auto){
      if(length(burn.in)==1) cat(paste0("- Nsim: ", format(n.sim, nsmall = 0, big.mark = ","), " simulations\n",
               "- burn-in (iterations to convergence): ", format(burn.in, nsmall = 0, big.mark = ","),
               "\n", "- ", mcmc.chains, ", thinned (", mcmc.thin, "x)")) else 
                 cat(paste0("- Nsim: ", format(n.sim, nsmall = 0, big.mark = ","), " simulations\n", "- Variable burn-in (iterations to convergence) \n", "- ", mcmc.chains, ", thinned (", mcmc.thin, "x)"))

    }else{
      if(length(burn.in)==1) cat(paste0("- N: ", format(n.sim, nsmall = 0, big.mark = ","), " simulations\n",
                 "- burn-in: ", format(burn.in, nsmall = 0, big.mark = ","), " iterations\n", "- ", mcmc.chains, " chain(s), run until convergence (Gelman-Rubin = ", mcmc.gelmanrubin, ")")) else cat(paste0("- N: ", format(n.sim, nsmall = 0, big.mark = ","), " simulations\n", "- Variable burn-in (iterations to convergence) \n", "- ", mcmc.chains, " chain(s), run until convergence (Gelman-Rubin = ", mcmc.gelmanrubin, ")"))
    }
    
    cat(paste0("\n- Total posterior samples (per simulation): ", format(mcmc.n, nsmall = 0, big.mark = ",")))
    cat(paste0("\n"))
    
    cat(crayon::bold(paste0("\nSimulations:\n")))
    cat(crayon::bold(paste0("-------------------------------------\n")))
  }
  
# Perform initial checks --------------------------------------------------

  if(verbose) cat("Performing start up checks ...")
  
  #'-------------------------------------------------
  # Create object to store error/warning messages
  #'-------------------------------------------------
  
  f.checks <- ArgumentCheck::newArgCheck() 
  
  #'-------------------------------------------------
  # Errors
  #'-------------------------------------------------
  
  # One of prop.sat or uncertainty.dose must be supplied
  
  if (is.null(uncertainty.dose) & is.null(prop.sat)) 
    ArgumentCheck::addError(
      msg = "Unspecified parameters: uncertainty.dose and prop.sat",
      argcheck = f.checks)
  
  # Missing arguments
  
  if (scenario %in% c(1,2) & is.null(uncertainty.dose)) 
    ArgumentCheck::addError(
      msg = "Dose measurement errors must be specified in scenarios 1 and 2",
      argcheck = f.checks)
  
  if (scenario %in% c(3,4) & is.null(prop.sat)) 
    ArgumentCheck::addError(
      msg = "Tag ratios must be specified in scenarios 3 and 4",
      argcheck = f.checks)
  
  if (mcmc.n==0) 
    ArgumentCheck::addError(
      msg = "mcmc.n must be strictly positive.",
      argcheck = f.checks)
  
  # ARGOS species correctly specified
  
  if (!species.argos%in%c("Gm", "Zc"))
    ArgumentCheck::addError(
      msg = "Species not found - species.argos can only be one of 'Zc' or 'Gm'",
      argcheck = f.checks)
  
  # Right-censoring correctly specified
  
  if (!length(censor.right)==2 | !is.numeric(censor.right) | censor.right[2]<=censor.right[1])
    ArgumentCheck::addError(
      msg = "Cannot apply right-censoring, Please check function inputs.",
      argcheck = f.checks)
  
  # Lower bound cannot exceed upper bound
  
  if (lower.bound > upper.bound) 
    ArgumentCheck::addError(
      msg = "Lower bound cannot exceed upper bound.",
      argcheck = f.checks)
  
  # Mean cannot exceed upper bound
  
  if (true.omega > omega.upper.bound) 
    ArgumentCheck::addError(
      msg = "True parameter value cannot exceed upper bound.",
      argcheck = f.checks)
  
  if (true.mu > upper.bound) 
    ArgumentCheck::addError(
      msg = "True parameter value cannot exceed upper bound.",
      argcheck = f.checks)
  
  if(upper.bound > source.level)
    ArgumentCheck::addError(
      msg = "Upper.bound cannot exceed source level.",
      argcheck = f.checks)
  
  # Parameters are in the correct format 
  
  if (sum(n.sim%%1) > 0 | sum(n.whales%%1) > 0 | sum(burn.in%%1) > 0 | sum(mcmc.n%%1) > 0 | sum(mcmc.chains%%1) > 0 | sum(mcmc.thin%%1) > 0 | sum(parallel.cores%%1) > 0 | sum(mcmc.n%%1) > 0 | sum(n.trials.per.whale%%1) > 0) 
    ArgumentCheck::addError(
      msg = "Incorrect input parameter(s). Some parameters can only be strictly positive integers.",
      argcheck = f.checks)
  
  if (mcmc.thin == 0 | mcmc.thin%%1 > 0) 
    ArgumentCheck::addError(
      msg = "Incorrect thinning parameter: mcmc.thin must be a strictly positive integer.",
      argcheck = f.checks)
  
  if (length(burn.in) > 1){
    if(scenario %in% c(1,2)){
    if(!length(burn.in)==length(uncertainty.dose))
    ArgumentCheck::addError(
      msg = "burn.in has incorrect dimensions.",
      argcheck = f.checks)
    }else{
      if(!length(burn.in)==length(prop.sat))
        ArgumentCheck::addError(
          msg = "burn.in has incorrect dimensions.",
          argcheck = f.checks)
    }}

  # Number of cores required cannot exceed number of cores available
  
  if (parallel.cores > parallel::detectCores()) 
    ArgumentCheck::addError(
      msg = "Not enough cores available.",
      argcheck = f.checks)
  
  # Convergence checks require multiple chains
  
  if (mcmc.chains==1 & check.convergence) 
    ArgumentCheck::addError(
      msg = "Convergence checks require more than one chain.",
      argcheck = f.checks)
  
  # Scenarios 1-4
  
  if (!is.numeric(scenario) | !scenario%in%1:4) 
    ArgumentCheck::addError(
      msg = "Unknown scenario",
      argcheck = f.checks)
  
  #'-------------------------------------------------
  # Warnings
  #'-------------------------------------------------
  
  issued.warning <- FALSE
  
  #'-------------------------------------------------
  # Output directory
  #'-------------------------------------------------

  # This will be folder for the appropriate simulation scenario, within the 'out' directory 
  # First, check whether the directory exists.
  
  out.dir <- paste0(getwd(),"/out/")
  scenario.dir <- paste0(out.dir, "scenario_", scenario, "/")
  
  if (!dir.exists(out.dir)) dir.create(path = out.dir, showWarnings = FALSE)
  if(!dir.exists(scenario.dir))  dir.create(path = scenario.dir, showWarnings = FALSE)
    
  # If so, check whether an INDEX.txt file exists. 
  # This file stores a sequential integer used to keep track of all 
  # simulations run for a particular scenario. 
  # If the file doesn't exist, create it.
  # If it does, retrieve the index and add 1.
  
  index.file <- file.path(paste0(getwd(), paste0("/out/scenario_", scenario, "/INDEX.txt")))
  
  if(!file.exists(index.file)){
  file.create(index.file, showWarnings = FALSE)
  fileConn <- file(index.file)
  index.sim <- addlzero(1)
  writeLines(index.sim, fileConn)
  close(fileConn)
  }else{
  index.sim <- addlzero(read.table(index.file)+1)
  fileConn <- file(index.file)
  writeLines(index.sim, fileConn)
  close(fileConn)
  }
  
  # Usually not more than 5 MCMC chains
  
  if (mcmc.chains > 5){
    cat(crayon::yellow("\n\u2717 Warning: High number of chains. Model fitting my be slow. \n"))
    issued.warning <- TRUE}
  
  # All available cores used
  
  if (parallel.cores == parallel::detectCores()){
    cat(crayon::yellow(" \u2717 Warning: Maximum number of cores used. \n"))
    issued.warning <- TRUE}
  
  # prop.sat
  
  if(scenario %in% c(3,4)){
  if (all(prop.sat<=1) & length(prop.sat)>1) 
    cat(crayon::yellow(" \u2717 Warning: prop.sat must be expressed as a percentage between 0 and 100%. Results may not be reliable. \n"))
    issued.warning <- TRUE}
  
  # true.beta
  
  if(scenario %in% c(2,4)){
    if (true.beta<0) 
      cat(crayon::yellow(" \u2717 Warning: true.beta must be positive. Absolute value taken instead. \n"))
    issued.warning <- TRUE}
  
  # Number of trace plots cannot exceed number of simulations
  
  if (no.tracePlots > n.sim){
    cat(crayon::yellow(paste0(" \u2717 Warning: Number of trace plots cannot exceed number of simulations.",
                              " Only ", n.sim, "plot(s) will be generated.\n")))
  no.tracePlots <- n.sim
  issued.warning <- TRUE}
  
  # Sanity checks
  
  if (check.convergence){
    if(no.tracePlots==0){
      cat(crayon::yellow(" \u2717 Warning: max.tracePlots cannot be nil and was changed to 1. \n"))
      max.tracePlots <- 1
      issued.warning <- TRUE
    }
    if(no.tracePlots>5){
      cat(crayon::yellow(" \u2717 Warning: max.tracePlots changed to 5 (max allowable). \n"))
      issued.warning <- TRUE
      max.tracePlots <- 5
    }
  } 
  
  # Return errors and warnings (if any)
  
  ArgumentCheck::finishArgCheck(f.checks) 
  
  if(!issued.warning){if(verbose) cat(crayon::green(" \u2713\n"))}

# ARGOS --------------------------------------------------------------
  
  #'-------------------------------------------------
  # Import ARGOS data as appropriate
  #'-------------------------------------------------
  
  # if(scenario %in% c(3,4)){
    
  argos <- purrr::map_dfr(.x = list.files(path = "./data", pattern = ".csv", full.names = TRUE), 
                          .f = ~readr::read_csv(file = .x, 
                                                col_types = c("cccTccddddddc"),
                                                na = "")) %>%
    janitor::clean_names(.) %>% 
    dplyr::mutate(species = ifelse(deploy_id == "ZcTag056", "Zc", "Gm")) %>% 
    dplyr::filter(species == species.argos) %>% 
    dplyr::mutate_if(sapply(., is.character), as.factor) %>%  # Convert characters to factors
    dplyr::filter(is.na(comment)) %>% 
    dplyr::select(-comment) %>% 
    na.omit(.)
  
# JAGS Model --------------------------------------------------------------

  if(verbose) cat("\nInitialising ...")
  
  true.beta <- abs(true.beta)/2
  
  #'--------------------------------------------------------------------
  # Define JAGS model file
  #'--------------------------------------------------------------------
  
  model_file <- paste0("jags/BayesianBRModel_JAGSsim_scenario", scenario, ".txt")
  
  #'-------------------------------------------------
  # Monitored parameters
  #'-------------------------------------------------
  
  # Need to be in alphabetical order, as this is how JAGS outputs are returned
  
  if(scenario %in% c(1,3)) params.monitored <- sort(c("mu", "omega"))
  if(scenario %in% c(2,4)) params.monitored <- sort(c("mu", "phi", "sigma", "beta", "alpha"))

  #'-------------------------------------------------
  # Define relevant parameter for observation model
  #'-------------------------------------------------
  
  # This takes the values of uncertainty.dose in scenarios 1 and 2
  # and the values of prop.sat in scenarios 3 and 4
  
  if(scenario %in% c(1,2)) obs.param <- uncertainty.dose
  if(scenario %in% c(3,4)) obs.param <- prop.sat
  
  #'--------------------------------------------------------------------
  # Start the parallel cluster
  #'--------------------------------------------------------------------
  
  start_cluster(n.cores = parallel.cores)
  
  #'--------------------------------------------------------------------
  # Build and run the model using parallelisation
  #'--------------------------------------------------------------------

  if(verbose) cat(crayon::green(" \u2713\n"))
  if(verbose) cat("Running simulations ... ")
  
  mcmc.sims <- foreach::foreach(nowh = 1:length(n.whales)) %:%
    
    foreach::foreach(obsval = 1:length(obs.param)) %:%
    
    foreach::foreach(sim = 1:n.sim,
                     .packages = c("tidyverse", "magrittr", "MASS", "truncnorm", "coda", "runjags"),
                     .export = c("TL", "range_finder", "xy_error", "argos")) %dopar% {
                       
      #'-------------------------------------------------
      # Generate response thresholds from truncated normals
      #'-------------------------------------------------
      
      if(scenario %in% c(1,3)){
        
      mu.i <- truncnorm::rtruncnorm(n = n.whales[nowh], 
                                    a = lower.bound, 
                                    b = upper.bound, 
                                    mean = true.mu, 
                                    sd = true.omega)
      
      }else if(scenario %in% c(2,4)){
        
      mu.i <- truncnorm::rtruncnorm(n = n.whales[nowh], 
                                      a = lower.bound, 
                                      b = upper.bound, 
                                      mean = true.mu, 
                                      sd = true.phi)}
      
      #'-------------------------------------------------
      # Add covariate effects where appropriate
      #'-------------------------------------------------
      
      if(scenario %in% c(2,4)){
      
        #'-----------------------------------------------------
        # Generate trials
        #'-----------------------------------------------------
        
        n.trials <- n.whales[nowh]*n.trials.per.whale
        whale.id <- rep(1:n.whales[nowh], each = n.trials.per.whale)
        
        #'-----------------------------------------------------
        # Previous exposure + signal type
        #'-----------------------------------------------------
        
        is.exposed <- rep(c(0, rep(1, n.trials.per.whale-1)), n.whales[nowh])
        is.mfas <- sample.int(n = 2, size = n.trials, replace = TRUE)-1
        is.lfas <- as.numeric(!is.mfas)

        mu.ij <- rep(mu.i, each = n.trials.per.whale) + 
          is.exposed*true.alpha - 
          is.mfas*(true.beta) + # MFAS
          is.lfas*(true.beta) # LFAS
      
        t.ij <- truncnorm::rtruncnorm(n = n.trials, 
                                      a = lower.bound, 
                                      b = upper.bound, 
                                      mean = mu.ij, 
                                      sd = true.sigma)}
      
      #'-------------------------------------------------
      # Assign tag types to each whale according to tag ratio
      #'-------------------------------------------------
      # 0 = DTAG, 1 = SAT
      
      if(scenario %in% c(3, 4)){
        
      attached.tags <- rep(0, n.whales[nowh])
      attached.tags[sample(x = seq_along(attached.tags), 
                           size = round(length(attached.tags)*obs.param[obsval]/100), 
                           replace = FALSE)] <- 1}
                       
      if(scenario == 4) attached.tags <- rep(attached.tags, each = n.trials.per.whale) 
      
      #'-------------------------------------------------
      # Generate observations
      #'-------------------------------------------------

      # For scenarios 3 and 4, the uncertainty in the received dose needs to appropriately account
      # for the positional uncertainty inherent to the various types of tags used (DTAG vs SAT).
      # This is done by converting received levels into estimates of range, adding a bivariate xy error
      # to these estimates, and then converting back to RL.
      
      if(scenario == 3){
        argos.correction <- purrr::map(.x = mu.i[attached.tags==1], 
                                       .f = ~xy_error(argos.data = argos,
                                                      received.lvl = .x, 
                                                      source.lvl = source.level,
                                                      multi = FALSE, 
                                                      plot.ellipse = FALSE))
        
        # The below line is wrong as it may lead to thresholds < lower.bound, which will cause
        # the JAGS model to fail.
        # mu.i[attached.tags==1] <- purrr::map_dbl(.x = argos.correction, "mean")
        sd.RL <- purrr::map_dbl(.x = argos.correction, "sd")
        
      }
                       
        if(scenario == 4){
        argos.correction <- purrr::map(.x = t.ij[attached.tags==1], 
                                       .f = ~xy_error(argos.data = argos,
                                                      received.lvl = .x, 
                                                      source.lvl = source.level, 
                                                      multi = FALSE, plot.ellipse = FALSE))
        
        # t.ij[attached.tags==1] <- purrr::map_dbl(.x = argos.correction, "mean")
        sd.RL <- purrr::map_dbl(.x = argos.correction, "sd")

        }
      
      if(scenario == 3){
        uncertainty.RL <- rep(dtag.sd, n.whales[nowh])
        uncertainty.RL[which(attached.tags==1)] <- sd.RL}
      
      if(scenario == 4){
        uncertainty.RL <- rep(dtag.sd, n.trials)
        uncertainty.RL[which(attached.tags==1)] <- sd.RL}
      
      # Final observations

      if(scenario == 1) y <- stats::rnorm(n = n.whales[nowh], mean = mu.i, sd = obs.param[obsval])
      if(scenario == 2) y <- stats::rnorm(n = n.trials, mean = t.ij, sd = obs.param[obsval])
      if(scenario == 3) y <- stats::rnorm(n = n.whales[nowh], mean = mu.i, sd = uncertainty.RL)
      if(scenario == 4) y <- stats::rnorm(n = n.trials, mean = t.ij, sd = uncertainty.RL)

      #'-------------------------------------------------
      # Censoring
      #'-------------------------------------------------
      
      # Upper bound reached in the case of censoring - sampled from U(left, right)
      
      if(scenario%in%c(1,3)) Rc <- runif(n = n.whales[nowh], min = censor.right[1], max = censor.right[2]) else Rc <- rep(runif(n = n.whales[nowh], min = censor.right[1], max = censor.right[2]), each = n.trials.per.whale)
      
      U <- rep(upper.bound, ifelse(scenario%in%c(1,3), n.whales[nowh], n.trials))
      
      # http://doingbayesiandataanalysis.blogspot.com/2012/01/complete-example-of-right-censoring-in.html
      # https://stats.stackexchange.com/questions/70858/right-censored-survival-fit-with-jags?rq=1
      # When is.censored is 1 and y is NA (missing), then all JAGS knows is that
      # y is somewhere above U (the censoring limit), so JAGS effectively 
      # imputes a random value for y. To understand this, let's unpack the
      # dinterval() distribution (see JAGS model code). Suppose that U has
      # 3 values in it, rather than just one, such that U = c(170,180,190).
      # Then, randomly generated values from dinterval(y, c(170,180,190)) 
      # will be either 0, 1, 2, or 3 depending on whether y<170, 170<y<800,
      # 180<y<190, or 190<y. So, if y = 175, dinterval(y,c(170,180,190)) 
      # has output of 1 with 100% probability. The trick is this: We instead
      # specify the output of dinterval, and impute a random value of y 
      # that could produce it. Thus, if we say 1 ~ dinterval(y,c(170,180,190))
      # then y is imputed as a random value between 170 and 180.
      
      is.censored <- ifelse(y>Rc, 1, 0)
      y.censored <- y; y.censored[is.censored==1] <- NA
      
      #'-------------------------------------------------
      # Left-censoring
      #'-------------------------------------------------
      
      # We do not account for left-censoring here, but see below.
      # http://jamescurran.co.nz/2014/06/bayesian-modelling-of-left-censored-data-using-jags/
      
      #'-------------------------------------------------
      # Precision, i.e. inverse of variance
      #'-------------------------------------------------
      
      if(scenario %in% c(1,2)) measurement.precision <- 1/(obs.param[obsval]^2)
      if(scenario %in% c(3,4)) measurement.precision <- 1/(uncertainty.RL^2)
    
      #'-------------------------------------------------
      # Write the data and initial values to lists
      #'-------------------------------------------------
      
      # Data
      
      sim_data <- list(n.whales = n.whales[nowh],
                       lower.bound = lower.bound,
                       upper.bound = upper.bound,
                       measurement.precision = measurement.precision,
                       y = y.censored,
                       I.censored = is.censored,
                       U = U)
      
      if(scenario %in% c(1,3)) sim_data <- append(sim_data, list(omega.upper.bound = omega.upper.bound))
      if(scenario %in% c(2,4)) sim_data <- append(sim_data, list(whale.id = whale.id,
                                                                 n.trials = n.trials,
                                                                 I.mfas = is.mfas, 
                                                                 I.lfas = is.lfas,
                                                                 I.exposed = is.exposed,
                                                                 beta_priorsd = beta.sd,
                                                                 alpha_priorsd = alpha.sd,
                                                                 phi.upper.bound = phi.upper.bound,
                                                                 sigma.upper.bound = sigma.upper.bound))

      # Initial values
      
      sim_inits <- list(mu = true.mu, mu_i = rep(true.mu, length(mu.i)))
      
      if(scenario %in% c(1,3)) sim_inits <- append(sim_inits, list(omega = true.omega))
      if(scenario %in% c(2,4)) sim_inits <- append(sim_inits, list(t_ij = t.ij, 
                                                                   beta = true.beta, 
                                                                   alpha = true.alpha,
                                                                   phi = true.phi, 
                                                                   sigma = true.sigma))
      
      if(!mcmc.auto){
      
      #'-------------------------------------------------
      # Run the model in JAGS
      #'-------------------------------------------------
      
      m <- rjags::jags.model(file = model_file,
                             data = sim_data,
                             inits = sim_inits,
                             n.adapt = mcmc.adapt,
                             quiet = TRUE,
                             n.chains = mcmc.chains)
      
      #'-------------------------------------------------
      # Burn-in
      #'-------------------------------------------------
      
      if(length(burn.in)==1) rjags:::update.jags(object = m, 
                                                 n.iter = burn.in, 
                                                 progress.bar = "none")
      
      if(length(burn.in)>1) rjags:::update.jags(object = m, 
                                                 n.iter = burn.in[obsval], 
                                                 progress.bar = "none")
      
      #'-------------------------------------------------
      # Draw samples from the posterior
      #'-------------------------------------------------
      
      rjags::coda.samples(model = m,
                          variable.names = params.monitored,
                          n.iter = (mcmc.n*mcmc.thin),
                          thin = mcmc.thin,
                          progress.bar = "none")
      
      }else{
        
        #'-------------------------------------------------
        # Assess convergence automatically
        #'-------------------------------------------------
        
        # It can be challenging to ensure that chain convergence is reached in all simulations.
        # The autorun.jags function from the runjags package is designed to runs iterations 
        # until the Gelman and Rubin statistic becomes less than a chosen threshold
        # (1.05 by default) for all model parameters.
        # Thin.sample is used to thin the chains after convergence, in order to reduce the size
        # of the output object.
        
        if(length(burn.in)==1) br <- rep(burn.in, length(obs.param)) else br <- burn.in
        
        runjags::autorun.jags(model = paste0(getwd(), "/", model_file),
                                                     monitor = params.monitored,
                                                     data = sim_data,
                                                     inits = sim_inits,
                                                     adapt = mcmc.adapt,
                                                     thin = mcmc.thin,
                                                     n.chains = mcmc.chains,
                                                     startburnin = br[obsval],
                                                     psrf.target = mcmc.gelmanrubin)
        
      }
 
    } # End simulation loop

  if(verbose) cat(crayon::green("\u2713"))
  
# Post-processing ---------------------------------------------------------

  if(verbose){
    
    cat(crayon::bold(paste0("\n\nPost-processing:\n")))
    cat(crayon::bold(paste0("-------------------------------------\n")))
  }
  
  #'--------------------------------------------------------------------
  # Assign names to output lists
  #'--------------------------------------------------------------------
  
  mcmc.sims <- name.list(scenario.id = scenario,
                         input.list = mcmc.sims,
                         Nwhales = n.whales,
                         Nsim = n.sim,
                         dose.or.ratio = obs.param)

  #'--------------------------------------------------------------------
# Extract posterior samples ====
  
  if(verbose) cat("Extracting posterior samples ... ")
  
  if(scenario %in% c(1,2)) col_names <- c("n", "RL", "sim", "param")
  if(scenario %in% c(3,4)) col_names <- c("n", "ratio", "sim", "param")
  
  #'-------------------------------------------------
  # Chain lengths
  #'-------------------------------------------------
  
  if(mcmc.auto) chain.lengths <- purrr::map_depth(.x = mcmc.sims, .depth = 3, .f = "mcmc") %>% purrr::map_depth(., 3, ~as.matrix(.x[[1]]) %>% nrow(.)) %>% unlist() # This should be equal to burn.in when mcmc.auto = F

  if(!mcmc.auto){
    
    if(length(burn.in)==1) chain.lengths <- purrr::map_depth(.x = mcmc.sims, .depth = 4, .f = ~burn.in) %>% unlist()
    
    if(length(burn.in)>1) {chain.names <- purrr::map_depth(.x = mcmc.sims, .depth = 4, .f = ~NA) %>% unlist()
    chain.names <- names(chain.names)
    chain.lengths <- rep(burn.in, each = mcmc.chains*n.sim*length(n.whales))
    names(chain.lengths) <- chain.names}
  }
  
  chain.lg <- chain.lengths %>% 
    reshape2::melt(.) %>% 
    tibble::rownames_to_column(.) %>% 
    tibble::as_tibble(.) %>% 
    tidyr::separate(col = rowname, into = c("n", col_names[2], "sim"), sep = "\\.") %>% 
    removelabels(scenario.id = scenario, tbl = .) %>% 
    dplyr::group_by(.dots = col_names[1:2]) %>% 
    dplyr::summarise(maxlg = max(value)) %>% 
    dplyr::mutate(maxlg = plyr::round_any(maxlg, 1000))
  
  
  #'-------------------------------------------------
  # Update the sampler
  #'-------------------------------------------------
  
  # mcmc.samples is used for posterior inference
  # mcmc.sims is used for model/convergence checks
  
  if(!mcmc.auto){
    
    mcmc.samples <- purrr::map_depth(.x = mcmc.sims,.depth = 3, 
                                     .f = ~ .x[[sample(1:mcmc.chains, 1)]] %>% 
                                     as.matrix(.))}
  
  if(mcmc.auto){

    # The below can be run using purrr, which is more elegant, but may
    # take longer.
    
    # mcmc.samples <- purrr::map_depth(.x = mcmc.sims,
    #                   .depth = 3,
    #                  .f = ~quiet(runjags::extend.jags(runjags.object = .x,
    #                                                  sample = mcmc.n,
    #                                                  combine = FALSE,
    #                                                  silent.jags = TRUE,
    #                                                  drop.chain = sample(x = 1:mcmc.chains,
    #                                                                size = mcmc.chains-1),
    #                                                                  summarise = FALSE)) %>%
    #                                  coda::as.mcmc.list(., vars = params.monitored) %>%
    #                                    .[[1]] %>% as.matrix(.))
    # 
    # mcmc.sims <- purrr::map_depth(.x = mcmc.sims,
    #                               .depth = 3,
    #                               .f = ~quiet(runjags::extend.jags(runjags.object = .x,
    #                                                                sample = mcmc.n,
    #                                                                combine = FALSE,
    #                                                                silent.jags = TRUE,
    #                                                                summarise = FALSE)) %>%
    #                                 coda::as.mcmc.list(., vars = params.monitored))
    
      mcmc.samples <- foreach::foreach(nowh = 1:length(n.whales)) %:%

      foreach::foreach(obsval = 1:length(obs.param)) %:%

      foreach::foreach(sim = 1:n.sim,
                       .packages = c("tidyverse", "magrittr", "runjags", "coda"),
                       .export = c("quiet")) %dopar% {

                         ext <- quiet(runjags::extend.jags(runjags.object = mcmc.sims[[nowh]][[obsval]][[sim]],
                                                    sample = mcmc.n,
                                                    silent.jags = TRUE,
                                                    combine = FALSE,
                                                    drop.chain = sample(x = 1:mcmc.chains,
                                                                        size = mcmc.chains-1),
                                                    summarise = FALSE))
                   coda::as.mcmc.list(ext, vars = params.monitored) %>%
                           .[[1]] %>% as.matrix(.)}

      mcmc.sims <- foreach::foreach(nowh = 1:length(n.whales)) %:%

        foreach::foreach(obsval = 1:length(obs.param)) %:%

        foreach::foreach(sim = 1:n.sim,
                         .packages = c("tidyverse", "magrittr", "runjags", "coda"),
                         .export = c("quiet")) %dopar% {

                           ext <- quiet(runjags::extend.jags(runjags.object = mcmc.sims[[nowh]][[obsval]][[sim]],
                                                             sample = mcmc.n,
                                                             combine = FALSE,
                                                             silent.jags = TRUE,
                                                             summarise = FALSE))
                           coda::as.mcmc.list(ext, vars = params.monitored)}

  mcmc.samples <- name.list(scenario.id = scenario,
                         input.list = mcmc.samples,
                         Nwhales = n.whales,
                         Nsim = n.sim,
                         dose.or.ratio = obs.param)

  mcmc.sims <- name.list(scenario.id = scenario,
                            input.list = mcmc.sims,
                            Nwhales = n.whales,
                            Nsim = n.sim,
                            dose.or.ratio = obs.param)
  
  }
  
  #'-------------------------------------------------
  # Extract results
  #'-------------------------------------------------
  
  mcmc.results <- mcmc.samples %>%
    reshape2::melt(.) %>%
    tibble::as_tibble(.) %>%
    dplyr::rename(n = L1, !!col_names[2] := L2, sim = L3, param = Var2) %>%
    dplyr::select_at(., tidyselect::all_of(c(col_names, "value"))) %>% 
    dplyr::group_by(.dots = col_names) %>%
    dplyr::summarise(post.mean = mean(value),
                     post.median = median(value),
                     post.sd = sd(value)) %>%
    dplyr::ungroup() %>%
    removelabels(scenario.id = scenario, tbl = .) %>%
    dplyr::arrange_at(., vars(col_names[c(1:2, 4)])) %>% 
    dplyr::mutate(param = as.character(param))
  
  #'-------------------------------------------------
  # Coverage for each parameter
  #'-------------------------------------------------
  
  # Calculate credible intervals

  coverage.tbl <- purrr::map_depth(.x = mcmc.samples, 
                                   .depth = 3, 
                                   .f = ~sapply(1:ncol(.x),
                                    function(y) HDInterval::hdi(.x[,y], hdi.value)) %>% 
                                     t(.) %>% data.frame(., row.names = params.monitored) %>% t(.)) %>% 
    reshape2::melt(.) %>%
    tibble::as_tibble(.) %>%
    dplyr::rename(n = L1, !!col_names[2] := L2, sim = L3, param = Var2, post = Var1) %>%
    tidyr::pivot_wider(., id_cols = tidyselect::all_of(col_names), names_from = post, values_from = value) %>% 
    dplyr::rename(post.lower = lower, post.upper = upper) %>% 
    removelabels(scenario.id = scenario, tbl = .) %>% 
    dplyr::mutate(param = as.character(param))
  

  # Retrieve true values and determine if they fall within CI 
  
  coverage.tbl$true <- map_dbl(.x = 1:nrow(coverage.tbl), ~get(paste0("true.", coverage.tbl[.x,"param"])))
  coverage.tbl <- coverage.tbl %>% 
    dplyr::mutate(param.coverage = ifelse(true>=post.lower & true <= post.upper, 1, 0))
  
  # Add to results

  mcmc.results <- mcmc.results %>% 
    dplyr::left_join(x = ., y = coverage.tbl, by = col_names)
  
  if(verbose) cat(crayon::green("\u2713\n"))
  
  #'--------------------------------------------------------------------
  # Create tibble of posterior values
  #'--------------------------------------------------------------------

  mcmc.tbl <- mcmc.samples %>%
    reshape2::melt(.) %>%
    tibble::as_tibble(.) %>%
    dplyr::rename(n = L1, !!col_names[2] := L2, sim = L3, param = Var2) %>% 
    removelabels(scenario.id = scenario, tbl = .) %>%
    dplyr::select(., value, tidyselect::all_of(col_names)) %>% 
    dplyr::mutate(param = gsub(pattern = "posterior.", replacement = "", x = param)) %>% 
    dplyr::arrange_at(., tidyselect::all_of(vars(col_names))) %>%
    dplyr::mutate(comb = paste0(n, "-", tidyselect::all_of(!!as.name(col_names[2]))))

  if(exists("internal.call")) mcmc.tbl.saved <- mcmc.tbl
  
  if(mcmc.save){
    for(i in 1:n.sim){
      mcmc.tbl %>% 
        dplyr::filter(sim == addlzero(i)) %>% 
        saveRDS(., file = paste0(getwd(),"/out/scenario_", scenario,"/mcmc.samples_", index.sim, "_sim", addlzero(i), ".rds"))
    }
  }
  
  # mcmc_tbl_backup <- mcmc.tbl %>% 
  #   split(., f = .$comb) %>%  
  #   purrr::map(.x = ., .f = ~split(.x, f = .x$param))
  
  # Split by n x (dose/ratio) combinations, then simulation ID, then parameter
  
  mcmc_tbl <- mcmc.tbl %>% 
    split(., f = .$comb) %>%  
    purrr::map(.x = ., .f = ~split(.x, f = .x$sim)) %>% 
    purrr::map_depth(.x = ., .depth = 2, .f = ~split(.x, f = .x$param))
  
# Effective response range (ERR) ------------------------------------------

  if(verbose) cat("Computing ERR values ... ")
  
  #'--------------------------------------------------------------------
  # Find range corresponding to lower threshold of hearing
  #'--------------------------------------------------------------------
  
  # The function optimize searches the interval from lower to upper 
  # for a minimum or maximum of the function f with respect to its first argument (of f).
  
  ret <- stats::optimize(f = range_finder, interval = c(0, 30000), 
                         SL = source.level, target.L = lower.bound) 
  
  #'--------------------------------------------------------------------
  # Maximum range (to nearest higher 10km)
  #'--------------------------------------------------------------------
  
  # The maximum range is the distance at which the RL drops below response.lowerbound
  # and thus where the probability of response is zero.
  
  maximum.range <- ceiling(ret$minimum/10)*10
  
  #'--------------------------------------------------------------------
  # Calculate received levels as function of range
  #'--------------------------------------------------------------------

  range.interval <- maximum.range/n.bins
  cutpoints <- seq(0, maximum.range, length = (n.bins+1))
  range.pts <- cutpoints[-1]-range.interval/2  # midpoints
  
  TLs <- TL(rge = range.pts) # How much is lost
  RLs <- source.level-TLs # Received levels given loss

  #'---------------------------------------------
  # Probabilities of response
  #'---------------------------------------------
  
  # Sanity check

  # testm <- mcmc.tbl %>% filter(param =="mu")
  # testo <- mcmc.tbl %>% filter(param =="omega")
  # 
  # testr <- c()
  # 
  # for(i in 1:nrow(testm)){
  # 
  #   testr[i] <- effective_range(response.threshold = testm$value[i],
  #                               response.sd = testo$value[i],
  #                               response.lowerbound = lower.bound,
  #                               response.upperbound = upper.bound,
  #                               received.level = RLs,
  #                               D = animal.density,
  #                               n.bins = n.bins,
  #                               maximum.rge = maximum.range)
  # 
  # }
  
  # ERR.N <- mcmc.n*n.sim/length(params.monitored)
  
  # CDF from a truncated Normal
  
  prob.response <- foreach::foreach(co = 1:length(mcmc_tbl)) %:%
                     foreach::foreach(pp = 1:(n.sim), 
                       .packages = c("truncnorm")) %dopar%{
                                      
                       if(scenario %in% c(1,3)){
                         
                         truncnorm::ptruncnorm(q = rep(RLs, each = mcmc.n),
                                               a = lower.bound,
                                               b = upper.bound,
                                               mean = mcmc_tbl[[co]][[pp]]$mu$value,
                                               sd = mcmc_tbl[[co]][[pp]]$omega$value)
                                        
                       }else{
                         
                         truncnorm::ptruncnorm(q = rep(RLs, each = mcmc.n),
                                               a = lower.bound,
                                               b = upper.bound,
                                               mean = mcmc_tbl[[co]][[pp]]$mu$value,
                                               sd = sqrt(mcmc_tbl[[co]][[pp]]$phi$value^2+mcmc_tbl[[co]][[pp]]$sigma$value^2))}}

  # Reorder elements
  
  prob.response.l <- purrr::map_depth(.x = prob.response, 
                               .depth = 2,
                               .f = ~lapply(1:mcmc.n, FUN = function(x) nth_element(.x, x, mcmc.n))) %>% 
    purrr::set_names(x = ., nm = names(mcmc_tbl)) %>% 
    purrr::map_depth(.x = ., .depth = 1, .f = ~purrr::set_names(x = .x, nm = names(mcmc_tbl[[1]])))
    
  # Calculate ERR
  
  err.values <- purrr::map_depth(.x = prob.response.l, .depth = 2, .f = ~lapply(1:mcmc.n, FUN = function(x) sqrt(animal.density*sum(animal.density*pi*(cutpoints[-1]^2-cutpoints[1:n.bins]^2)*.x[[x]])/pi)) %>% do.call(c, .)) %>% unlist()
  names(err.values) <- NULL

  #'--------------------------------------------------------------------
  # Compile results
  #'--------------------------------------------------------------------
  
  mcmc.tbl <- suppressWarnings(mcmc.tbl %>% 
    tidyr::pivot_wider(., names_from = param,
                       values_from = value, 
                       id_cols = tidyselect::all_of(c(!!col_names[1:3], "comb"))) %>% 
    tidyr::unnest(cols = tidyselect::all_of(params.monitored))) %>% 
    dplyr::mutate(ERR = err.values)
  
  mcmc.tbl <- mcmc.tbl %>% dplyr::mutate(comb = paste0(n, "-", !!as.name(col_names[2]), "-", sim))
  
  #'--------------------------------------------------------------------
  # Compute posterior mean, median, and HDI
  #'--------------------------------------------------------------------

  err.summary <- mcmc.tbl %>%
    dplyr::group_by_at(tidyselect::all_of(vars(col_names[1:3]))) %>% 
    dplyr::summarise(err.mean = mean(ERR),
                     err.median = median(ERR),
                     err.sd = sd(ERR),
                     err.low = HDInterval::hdi(ERR, credMass = hdi.value)[1],
                     err.up = HDInterval::hdi(ERR, credMass = hdi.value)[2]) %>% 
    dplyr::ungroup()
  
  #'--------------------------------------------------------------------
  # Compute coverage
  #'--------------------------------------------------------------------
  
  # 'True' ERR based on chosen values of model parameters
  
  true.err <- effective_range(response.threshold = true.mu,
                              response.sd = dplyr::case_when(scenario %in% c(1,3) ~ true.omega,
                                                             scenario %in% c(2,4) ~ sqrt(true.phi^2+true.sigma^2)),
                              response.lowerbound = lower.bound,
                              response.upperbound = upper.bound, 
                              received.level = RLs, 
                              D = animal.density, 
                              n.bins = n.bins, 
                              maximum.rge = maximum.range)
  
  err.summary <- err.summary %>%
    dplyr::rowwise() %>%
    dplyr::mutate(err.coverage = ifelse(err.low<=true.err & err.up>=true.err, 1, 0)) %>%
    dplyr::ungroup()
  
  mcmc.results <- mcmc.results %>%
    dplyr::left_join(., err.summary, by = c(col_names[1:3]))
  
  if(verbose) cat(crayon::green("\u2713\n"))

# Dose-response curves ----------------------------------------------------

  if(verbose) cat("Computing dose-response curves ... ")
  
  #'--------------------------------------------------------------------
  # Define range over which to plot the dose-response function
  #'--------------------------------------------------------------------
  
  dose.range <- seq(lower.bound, upper.bound, length = 100)
  
  #'--------------------------------------------------------------------
  # Extract posterior means for each simulation
  #'--------------------------------------------------------------------
  
  dose.summary <- mcmc.tbl %>%
    dplyr::group_by_at(vars(col_names[1:3])) %>% 
    dplyr::summarise_at(params.monitored, mean, na.rm = TRUE) %>% 
    dplyr::mutate(comb = paste0(n, "-", !!as.name(col_names[2]), "-", sim),
                  nR = paste0("n_", n, ifelse(scenario %in% c(1,2), "-RLsd_", "-sat_"), !!as.name(col_names[2])))
  
  #'--------------------------------------------------------------------
  # Calculate response probabilities for each pair of posterior values
  #'--------------------------------------------------------------------
  
  # Sanity check
  #                      
  # ERR.Nbackup <- mcmc.n * n.sim        
  # doseresp.values <- foreach::foreach(k = 1:length(mcmc_tbl_backup),
  #                                   .packages = c("truncnorm")) %dopar%{
  #                                     
  #                                     if(scenario %in% c(1,3)){truncnorm::ptruncnorm(q = rep(dose.range, each = ERR.Nbackup),
  #                                                                                    a = lower.bound,
  #                                                                                    b = upper.bound,
  #                                                                                    mean = mcmc_tbl_backup[[k]]$mu$value,
  #                                                                                    sd = mcmc_tbl_backup[[k]]$omega$value)
  #                                       
  #                                     }else{truncnorm::ptruncnorm(q = rep(dose.range, each = ERR.Nbackup),
  #                                                                 a = lower.bound,
  #                                                                 b = upper.bound,
  #                                                                 mean = mcmc_tbl_backup[[k]]$mu$value,
  #                                                                 sd = sqrt(mcmc_tbl_backup[[k]]$phi$value^2+mcmc_tbl_backup[[k]]$sigma$value^2))
  #                                     }}
  # 
  # doseresp.values <- purrr::map(.x = doseresp.values, 
  #                               .f = ~ {m <- matrix(data = .x, nrow = ERR.N, ncol = length(dose.range))
  #                               split(m, 1:nrow(m))}) %>% 
  #   purrr::flatten(.)
  
  doseresp.values <- foreach::foreach(co = 1:length(mcmc_tbl)) %:%
    foreach::foreach(pp = 1:n.sim, 
                     .packages = c("truncnorm")) %dopar%{
                       
                       if(scenario %in% c(1,3)){truncnorm::ptruncnorm(q = rep(dose.range, each = mcmc.n),
                                                                      a = lower.bound,
                                                                      b = upper.bound,
                                                                      mean = mcmc_tbl[[co]][[pp]]$mu$value,
                                                                      sd = mcmc_tbl[[co]][[pp]]$omega$value)
                         
                       }else{truncnorm::ptruncnorm(q = rep(dose.range, each = mcmc.n),
                                                   a = lower.bound,
                                                   b = upper.bound,
                                                   mean = mcmc_tbl[[co]][[pp]]$mu$value,
                                                   sd = sqrt(mcmc_tbl[[co]][[pp]]$phi$value^2+mcmc_tbl[[co]][[pp]]$sigma$value^2))
                         
                       }}
  
  doseresp.values <- purrr::map_depth(.x = doseresp.values, 
                                      .depth = 2,
                                      .f = ~lapply(1:mcmc.n, FUN = function(x) nth_element(.x, x, mcmc.n))) %>% 
    purrr::set_names(x = ., nm = names(mcmc_tbl)) %>% 
    purrr::map_depth(.x = ., .depth = 1, .f = ~purrr::set_names(x = .x, nm = names(mcmc_tbl[[1]])))
  
  #'--------------------------------------------------------------------
  # Quantiles of p(response)
  #'--------------------------------------------------------------------

  # Define quantiles
  
  quants <- seq(95, 1, by = -5)

  doseresp <- foreach::foreach(g = 1:nrow(dose.summary),
                               .packages = c("magrittr", "tidyverse", "truncnorm"),
                               .export = c("removelabels")) %dopar% {
                                
                                #'----------------------------------------------------
                                # Extract row indices corresponding to each simulation    
                                #'----------------------------------------------------
                                        
                                # row.indices <- which(mcmc.tbl$comb==dose.summary$comb[g])
                                
                                gn <- paste0(dose.summary[g,]$n, "-", dose.summary[g,col_names[2]])
                                gs <- dose.summary[g,]$sim
                                
                                #'----------------------------------------------------
                                # Extract relevant dose-response curves  
                                #'---------------------------------------------------- 

                                # dose_resp <- doseresp.values[row.indices] %>% 
                                #   do.call(rbind,.)
                                
                                dose_resp <- doseresp.values[[gn]][[gs]] %>% do.call(rbind, .)
          
                                #'----------------------------------------------------
                                # Calculate median dose-response curve   
                                #'---------------------------------------------------- 
                                 
                                p.median <- apply(X = dose_resp, MARGIN = 2, FUN = median)
                                 
                                #'----------------------------------------------------
                                # Calculate quantiles
                                #'---------------------------------------------------- 

                                 q.low <- purrr::map(.x = quants, 
                                                     .f = ~apply(X = dose_resp, MARGIN = 2, 
                                                                 FUN = quantile, (50-.x/2)/100)) %>% 
                                   do.call(rbind,.)
                                 
                                 q.up <- purrr::map(.x = quants, 
                                                    .f = ~apply(X = dose_resp, MARGIN = 2, 
                                                                FUN = quantile, (50+.x/2)/100)) %>% 
                                   do.call(rbind,.)
                                 
                                 #'----------------------------------------------------
                                 # Return results
                                 #'---------------------------------------------------- 
                                 
                                 list(doseresp.median = p.median, 
                                      doseresp.lower = q.low, 
                                      doseresp.upper = q.up)}

  doseresp <- doseresp %>% purrr::set_names(x = ., nm = dose.summary$nR)
  
  #'--------------------------------------------------------------------
  # Take average across simulations
  #'--------------------------------------------------------------------

  doseresp.avg <- foreach::foreach(g = unique(dose.summary$nR),
                               .packages = c("dplyr", "magrittr")) %dopar% {

                                #'----------------------------------------------------
                                # Extract row indices corresponding to n x RL / ratios   
                                #'----------------------------------------------------  
                                 
                                # row.indices <- which(dose.summary$nR==g)
                                sub.l <- doseresp[names(doseresp)==g]
   
                                nR.mean <- purrr::map(.x = sub.l, .f = "doseresp.median") %>% 
                                  do.call(rbind, .) %>% 
                                  apply(X = ., MARGIN = 2, mean)
                                 
                                # nR.mean <- purrr::map(.x = doseresp, 
                                #                         .f = "doseresp.median")[row.indices] %>% 
                                #    do.call(rbind, .) %>% 
                                #     apply(X = ., MARGIN = 2, mean)
                                
                                nR.qlow <- purrr::map(.x = sub.l, .f = "doseresp.lower")
                                
                                #'----------------------------------------------------
                                # Takes the mean over all matrices (by row)
                                #'----------------------------------------------------  
                                
                                nR.q.low <- do.call(cbind, nR.qlow) %>% 
                                  array(., dim = c(dim(nR.qlow[[1]]), length(nR.qlow))) %>% 
                                  apply(., c(1, 2), mean, na.rm = TRUE)
                                 
                                nR.qup <- purrr::map(.x = doseresp, .f = "doseresp.upper")
                                
                                nR.q.up <- do.call(cbind, nR.qup) %>% 
                                  array(., dim = c(dim(nR.qup[[1]]), length(nR.qup))) %>% 
                                  apply(., c(1, 2), mean, na.rm = TRUE)
                                 
                                 list(doseresp.mean = nR.mean,
                                      doseresp.lower = nR.q.low,
                                      doseresp.upper = nR.q.up)}

  # Assign names to list elements
  
  doseresp.avg <- purrr::set_names(doseresp.avg, unique(dose.summary$nR))
    
  if(verbose) cat(crayon::green("\u2713\n"))

# Model checks ------------------------------------------------------------
  
  if(verbose){
  cat(crayon::bold(paste0("\nModel checks:\n")))
  cat(crayon::bold(paste0("-------------------------------------\n")))}
  
  #'--------------------------------------------------------------------
  # Effective sample sizes (ESS) and autocorrelation
  #'--------------------------------------------------------------------
  
  # THINNING: One way to decrease autocorrelation is to thin the sample, using only every nth step.
  # If we keep 50,000 thinned steps with small autocorrelation, then we very probably have a 
  # more precise estimate of the posterior than 50,000 unthinned steps with high autocorrelation. 
  # But to get 50,000 kept steps in a thinned chain, we need to generate n*50,000 steps. 
  # With such a long chain, the clumpy autocorrelation has probably all been averaged out 
  # anyway! In fact, Link and Eaton show that the longer (unthinned) chain usually yields 
  # better estimates of the true posterior than the shorter thinned chain, even for 
  # percentiles in the tail of the distribution, at least for the particular cases they consider.
  # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210X.2011.00131.x
  
  # Sample size of a completely non-autocorrelated chain. This corresponds to 
  # the number of effectively independent draws from the posterior distribution 
  # that the Markov chain is equivalent to. It is calculated as the ratio between
  # the actual sample size and the amount of autocorrelation.
  # What ESS is adequate? The larger the better.
  # A general guideline for reasonably accurate and stable estimates of the limits of the 95% HDI
  # an ESS of 10,000 is recommended.
  # http://doingbayesiandataanalysis.blogspot.com/2018/02/run-mcmc-to-achieve-effective-sample.html
  
    ess.tbl <- purrr::map_depth(.x = mcmc.sims, 
                                .depth = 4, 
                                .f = ~sapply(1:ncol(.x), function(p) coda::effectiveSize(.x[,p])) %>% 
                                    data.frame(., row.names = params.monitored) %>% t(.))  %>% 
    reshape2::melt(.) %>%
    tibble::as_tibble(.) %>%
    dplyr::rename(n = L1, !!col_names[2] := L2, sim = L3, param = Var2) %>%
    dplyr::select(-Var1, -L4) %>% 
    removelabels(scenario.id = scenario, tbl = .) %>% 
    dplyr::mutate(param = as.character(param)) %>% 
    dplyr::rename(ESS = value) %>% 
    dplyr::group_by_at(vars(col_names)) %>% 
    dplyr::summarise(ESS = mean(ESS)) %>% dplyr::ungroup()


    ess.msg <- sapply(X = params.monitored, FUN = function(pp) round(mean(ess.tbl[ess.tbl$param==pp,]$ESS)))
    
    if(verbose){
      cat("Retrieving effective sample sizes ... \n")
      cat(paste0(params.monitored, " (", ess.msg, ");"))
      
      if(any(ess.msg < min.ESS)) {
        cat(crayon::yellow(paste0("\n\u2717 ESS < ", min.ESS, 
                                  ". Posterior estimates may not be accurate/stable.\n")))
      }else{
          cat(crayon::green(" \u2713 \n"))}}
  
  #'--------------------------------------------------------------------
  # Autocorrelation
  #'--------------------------------------------------------------------
  
  # High levels of autocorrelation in a MCMC algorithm are not fatal
  # in and of themselves but they will indicate that a very long run
  # of the sampler may be required. Thinning is not a strategy for
  # avoiding these long runs, but it is a strategy for dealing with
  # the otherwise overwhelming amount of MCMC output. 
  # Unless there is severe autocorrelation, e.g., high correlation
  # with, say [lag]=30, we don't believe that thinning is worthwhile.
  # 
  # Another way to check for convergence is to look at the 
  # autocorrelations between the samples returned by our MCMC. 
  # The lag-k autocorrelation is the correlation between every 
  # sample and the sample k steps before. This autocorrelation 
  # should become smaller as k increases, i.e. samples can be 
  # considered as independent. If, on the other hand, autocorrelation
  # remains high for higher values of k, this indicates a high degree
  # of correlation between our samples and slow mixing.
  # If autocorrelation persists, we can thin the MCMC chain, 
  # that is we discard n samples for every sample that we keep.
  # http://sbfnk.github.io/mfiidd/mcmc_diagnostics.html
  
  if(verbose) cat(paste0("Checking autocorrelation ..."))
  
  #'-------------------------------------------------
  #' Calculate in-chain correlation at large lags
  #'-------------------------------------------------

  mcmc.cor <- purrr::map_depth(.x = mcmc.sims, .depth = 3, 
                               .f = ~coda::autocorr.diag(.x, lags = c(20, 30, 40, 50))) %>% 
    reshape2::melt(.) %>% 
    tibble::as_tibble(.) %>% 
    dplyr::rename(correl = value, n = L1, !!col_names[2] :=L2, sim = L3, param = Var2, lag = Var1) %>% 
    dplyr::select(., n, !!col_names[2], sim, lag, correl, param) %>% 
    removelabels(scenario.id = scenario, tbl = .)
  
  #'-------------------------------------------------
  # Average correlation per lag (across sims) for 
  # each n, RL/ratio, and parameter
  #'-------------------------------------------------
  
  cor.means <- mcmc.cor %>% 
    group_by_at(vars(col_names[1:2], lag, param)) %>% 
    dplyr::summarise(correl = mean(correl))
  
  if(any(cor.means$correl > correlation.threshold)){
    autocorr.msg <- "There is evidence of in-chain autocorrelation at large lags in some simulations."
    if(verbose) cat(crayon::yellow(paste0(" \u2717 ", autocorr.msg, "\n")))
  }else{
    autocorr.msg <- "Chain values are uncorrelated."
    if(verbose) cat(crayon::green(paste0(" \u2713 ", autocorr.msg, "\n")))
  }
  
# PPO: Prior posterior overlap ------------------------------------------------------------
  
  # https://lynchlab.com/2018/01/18/check-your-prior-posterior-overlap-ppo-mcmc-wrangling-in-r-made-easy-with-mcmcvis/
  #   
  # Checking the PPO has particular utility when trying to determine if the parameters 
  # in a model are identifiable. If substantial PPO exists, the prior may simply be 
  # dictating the posterior distribution – the data may have little influence on the results. 
  # By contrast, if a small degree of PPO exists, the data were informative enough to overcome 
  # the influence of the prior. In the field of ecology, nonidentifiability is a particular 
  # concern in some types of mark-recapture models. 
  #  In a nutshell: Unidentifiable parameters will have high PPO values; Gimenez et al (2009) 
  #  suggest that overlap greater than 35% indicates weak identifiability.
  
  suppressWarnings(ppo.tbl <- lapply(X = params.monitored,
                                     FUN = function(pp){
                                       suppressMessages(purrr::map_depth(.x = mcmc.sims, 
                                                                         .depth = 3, 
                                                                         .f = ~{
                                                                           
                                                                           p.priors <- list(mu = runif(n = 15000, min = lower.bound, max = upper.bound),
                                                                                            omega = runif(n = 15000, min = 0, max = omega.upper.bound),
                                                                                            phi = runif(n = 15000, min = 0, max = phi.upper.bound),
                                                                                            sigma = runif(n = 15000, min = 0, max = sigma.upper.bound),
                                                                                            alpha = rnorm(n = 15000, mean = 0, sd = alpha.sd),
                                                                                            beta = rnorm(n = 15000, mean = 0, sd = beta.sd))           
                                                                           
                                                                           MCMCvis::MCMCtrace(object = .x, params = pp, 
                                                                                              priors = p.priors[[pp]], 
                                                                                              pdf = FALSE, 
                                                                                              plot = FALSE,
                                                                                              PPO_out = TRUE)
                                                                         }) %>% 
                                                          reshape2::melt(.)) %>% 
                                         dplyr::rename(., ppo = value, sim = L3, !!col_names[2] := L2, n = L1) %>% 
                                         dplyr::mutate(param = pp)
                                       
                                     }) %>% do.call(rbind,.) %>% 
                     tibble::as_tibble() %>% 
                     removelabels(scenario.id = scenario, tbl = .) %>% 
                     dplyr::mutate(ppo.check = ppo<=35) %>% 
                     dplyr::select(., n, !!col_names[2], sim, ppo, ppo.check, param))
  

    ppo.sum <- sapply(X = params.monitored, 
                      FUN = function(pp) sum(ppo.tbl[ppo.tbl$param==pp,]$ppo.check)==0)
    
    if(sum(ppo.sum)==sum(rep(1, length(ppo.sum)))){
      ppo.msg <- "Parameters are identifiable."
      if(verbose) cat(crayon::green(paste0(" \u2713 ", ppo.msg, "\n")))
    }else{
      ppo.msg <- "Some parameters show limited identifiability."
      if(verbose) cat(crayon::yellow(paste0(" \u2717 ", ppo.msg, "\n")))}
  
  
# Convergence diagnostics ------------------------------------------------------------

  if(check.convergence){
    
    if(verbose){
      cat(crayon::bold(paste0("\nConvergence diagnostics:\n")))
      cat(crayon::bold(paste0("-------------------------------------\n")))
    }
    
    #'-------------------------------------------------
    # R-hat / Gelman-Rubin statistic ====
    #'-------------------------------------------------
    
    # This is used to assess the convergence of the chains in the model. 
    # Compares between chain variance to within chain variance, 
    # with Rhat < 1.1 generally indicating successful convergence.
    # However, stopping at Rhat = 1.1 can be too early.
    # The use of autorun.jags can ensure that all chains have converged as per the rhat threshold
    # defined by mcmc.gelmanrubin - however this may mean the chains are run for far longer than needed.
    # 
    # Can also use a stable version of the GR statistic, which 
    # incorporates more sophisticated method of variance estimation.
    # This has the advantage of leading to less variable variance estimates, such that 
    # Rhat stabilizes. See https://arxiv.org/abs/1812.09384 for details.
    # e.g. stable.gelman <- purrr::map_depth(.x = mcmc.sims, .depth = 3, 
    # .f = ~stableGR::n.eff(x = .x$mcmc, multivariate = TRUE)$converged)
    
    if(gr.multivariate){
      
      # Can use the multivariate potential scale reduction factor mpsrf for multivariate chains
      
      gelman.rubin <- purrr::map_depth(.x = mcmc.sims, .depth = 3, 
                                       .f = ~coda::gelman.diag(x = .x, multivariate = TRUE)$mpsrf) %>% 
        reshape2::melt(.) %>% 
        tibble::as_tibble(.)
      
    }else{
      
      suppressMessages(gelman.rubin <- purrr::map_depth(.x = mcmc.sims, .depth = 3, 
                                       .f = ~coda::gelman.diag(x = .x)$psrf[,2] %>% 
                                         tibble::enframe(.))%>% 
        reshape2::melt(.) %>% 
        tibble::as_tibble(.))
    } 
    
    if(gr.multivariate){
      
      suppressMessages(gelman.rubin <- gelman.rubin %>% 
        dplyr::rename(rhat = value, n = L1, !!col_names[2] := L2, sim = L3) %>% 
        removelabels(scenario.id = scenario, tbl = .) %>% 
        dplyr::select(., n, !!col_names[2], sim, rhat))
      
    }else{
      
      suppressMessages(gelman.rubin <- gelman.rubin %>% 
        dplyr::rename(param = name, rhat = value, n = L1, !!col_names[2] := L2, sim = L3) %>% 
        removelabels(scenario.id = scenario, tbl = .) %>% 
        dplyr::select(., n, !!col_names[2], sim, param, rhat))
    }
    
    # Summarise convergence
    
    gelman.rubin <- gelman.rubin %>% dplyr::mutate(index = index.sim)
    
    gelman.check <- gelman.rubin %>% 
      dplyr::mutate(rcheck = rhat > mcmc.gelmanrubin) %>% 
      dplyr::group_by_at(.,vars(col_names[1:3])) %>% 
      dplyr::summarise(not.converged = max(rcheck)) %>% 
      dplyr::group_by_at(.,vars(col_names[1:2])) %>% 
      dplyr::summarise(not.converged = sum(not.converged)) %>% 
      dplyr::ungroup()

      if(verbose) cat("Gelman-Rubin statistic: ")
      
      if(sum(gelman.check$not.converged)==0){
        convergence.msg <- "MCMC chains converged in all simulations."
        if(verbose) cat(crayon::green(paste0("\u2713 ", convergence.msg, "\n")))
      }else{ 
        convergence.msg <- "MCMC chains did not converge in all simulations."
        if(verbose) cat(crayon::yellow(paste0("\u2717 ", convergence.msg, "\n")))}


    
    
    # Trace plots of MCMC chains ------------------------------------------------------------

    if(verbose) cat("Trace plots: ")
    
    # Set active bayesplot theme and colour scheme
    
    bayesplot::bayesplot_theme_set(new = theme_gray())
    bayesplot::bayesplot_theme_update(title = element_text(size = 12, family = "sans",  face = "bold"),
                                      text = element_text(size = 14, family = "sans"))
    bayesplot::color_scheme_set("darkgray")
    
    # Select random sample of simulations for which to get trace plots

    which.trace <- paste0("sim_", addlzero(sample(x = 1:n.sim, 
                                                  size = no.tracePlots, 
                                                  replace = FALSE)))
    
    trace.df <- expand.grid(n.whales, obs.param, which.trace) %>% 
      tibble::as_tibble(.) %>% 
      dplyr::rename(n = Var1, !!col_names[2] := Var2, sim = Var3) %>% 
      dplyr::arrange_at(., vars(col_names[1:2]))
    
    # Create plots and store them in a list
    
    nrow.per.page <- ifelse(nrow(trace.df) < 4, nrow(trace.df), 4)
    
    trace.plots <- try(purrr::map(.x = seq_len(nrow(trace.df)), 
     .f = ~{
       
       bayesplot::mcmc_trace(mcmc.sims[[paste0("n_", addlzero(trace.df$n[.x]))]][[paste0(ifelse(scenario %in% c(1,2), "RLsd_", "sat_"), addlzero(trace.df[.x,2]))]][[trace.df$sim[.x]]], pars = params.monitored) + 
           ggplot2::ggtitle(paste0("n = ", trace.df$n[.x], " whales     ", 
                                   ifelse(scenario %in% c(1,2), "SD(dose) = ", "SAT = "),
                                   ifelse(scenario %in% c(3,4), trace.df[.x, 2], trace.df[.x, 2]),
                                   ifelse(scenario %in% c(1,2), " dB", " %")))

       }) %>%  gridExtra::marrangeGrob(grobs = ., nrow = nrow.per.page, ncol = 1), silent = TRUE)
    
    # Save plots to file
    
    if(class(trace.plots)[1]== "try-error"){
      
      if(verbose) cat(crayon::yellow("\u2717 An error occurred. Plots could not be generated.\n"))
      
    }else{
      
      if(save.trace){
      suppressMessages(ggpubr::ggexport(trace.plots, filename = paste0(getwd(), "/out/scenario_", scenario, "/S", scenario, "_tracePlots_", index.sim, ".pdf")))}
      
      if(verbose) cat(crayon::green("\u2713 Plots saved to disk.\n"))
    }
  } # End if check.convergence
  

# Posterior predictive checks ------------------------------------------------------------
  
if(posterior.checks){
    
    if(verbose){
      cat(crayon::bold(paste0("\nPosterior checks:\n")))
      cat(crayon::bold(paste0("-------------------------------------\n")))
      cat("Prior posterior overlap (PPO):")}

    y.dat <- foreach::foreach(nowh = 1:length(n.whales)) %:%
      
      foreach::foreach(obsval = 1:length(obs.param)) %:%
      
      foreach::foreach(sim = 1:n.sim, 
                       .packages = c("tidyverse", "magrittr", "MASS"),
                       .export = c("TL", "range_finder", "xy_error")) %dopar% {
        
        #'-------------------------------------------------
        # Response threshold
        #'-------------------------------------------------
        
        if(scenario %in% c(1,3)){
          
          mu.i <- truncnorm::rtruncnorm(n = n.whales[nowh], 
                                        a = lower.bound, 
                                        b = upper.bound, 
                                        mean = true.mu, 
                                        sd = true.omega)
        }else if(scenario %in% c(2,4)){
          
          mu.i <- truncnorm::rtruncnorm(n = n.whales[nowh], 
                                        a = lower.bound, 
                                        b = upper.bound, 
                                        mean = true.mu, 
                                        sd = true.phi)
        }
        
        #'-------------------------------------------------
        # Add covariate effects
        #'-------------------------------------------------
        
        if(scenario %in% c(2,4)){
                           
           #'-----------------------------------------------------
           # Generate trials
           #'-----------------------------------------------------
                           
           n.trials <- n.whales[nowh]*n.trials.per.whale
           whale.id <- rep(1:n.whales[nowh], each = n.trials.per.whale)
                           
           #'-----------------------------------------------------
           # Previous exposure + signal type
           #'-----------------------------------------------------
            
           is.exposed <- rep(c(0, rep(1, n.trials.per.whale-1)), n.whales[nowh])               
           is.mfas <- sample.int(n = 2, size = n.trials, replace = TRUE)-1
           is.lfas <- as.numeric(!is.mfas)
                           
           mu.ij <- rep(mu.i, each = n.trials.per.whale) + 
             is.exposed*true.alpha - 
             is.mfas*(true.beta) + # MFAS
             is.lfas*(true.beta) # LFAS
                           
           t.ij <- truncnorm::rtruncnorm(n = n.trials, 
                                         a = lower.bound, 
                                         b = upper.bound, 
                                         mean = mu.ij, 
                                         sd = true.sigma)}
                         
        
        #'-------------------------------------------------
        # Assign tag types to each whale according to tag ratio
        #'-------------------------------------------------
        # 0 = DTAG, 1 = SAT
        
        if(scenario %in% c(3,4)){
          
          attached.tags <- rep(0, n.whales[nowh])
          attached.tags[sample(x = seq_along(attached.tags), 
                               size = round(length(attached.tags)*obs.param[obsval]/100), 
                               replace = FALSE)] <- 1}
        
        #'-------------------------------------------------
        # Observations
        #'-------------------------------------------------
        
        # For scenarios 3 and 4, the uncertainty in the received dose needs to appropriately account
        # for the positional uncertainty inherent to the various types of tags used (DTAG vs SAT).
        # This is done by converting received levels into estimates of range, adding a bivariate xy error
        # to these estimates, and then converting back to RL.
        
        if(scenario == 3){
        argos.correction <- purrr::map(.x = mu.i[attached.tags==1], 
                                       .f = ~xy_error(argos.data = argos,
                                                      received.lvl = .x, 
                                                      source.lvl = source.level, 
                                                      multi = FALSE, plot.ellipse = FALSE))
        
        # mu.i[attached.tags==1] <- purrr::map_dbl(.x = argos.correction, "mean")
        sd.RL <- purrr::map_dbl(.x = argos.correction, "sd")
        }
                         
        if(scenario == 4){
        argos.correction <- purrr::map(.x = t.ij[attached.tags==1], 
                                       .f = ~xy_error(argos.data = argos,
                                                      received.lvl = .x, 
                                                      source.lvl = source.level, 
                                                      multi = FALSE, plot.ellipse = FALSE))
        
        # t.ij[attached.tags==1] <- purrr::map_dbl(.x = argos.correction, "mean")
        sd.RL <- purrr::map_dbl(.x = argos.correction, "sd")
        }
        
        if(scenario %in% c(3,4)){
        uncertainty.RL <- rep(dtag.sd, n.whales[nowh])
        uncertainty.RL[which(attached.tags==1)] <- sd.RL}
        
        # Final observations
        
        if(scenario == 1) y <- stats::rnorm(n = n.whales[nowh], mean = mu.i, sd = obs.param[obsval])
        if(scenario == 2) y <- stats::rnorm(n = n.trials, mean = t.ij, sd = obs.param[obsval])
        if(scenario == 3) y <- stats::rnorm(n = n.whales[nowh], mean = mu.i, sd = uncertainty.RL)
        if(scenario == 4) y <- stats::rnorm(n = n.trials, mean = t.ij, sd = uncertainty.RL)
        
        #'-------------------------------------------------
        # Censoring
        #'-------------------------------------------------
        
        if(scenario%in%c(1,3)) Rc <- runif(n = n.whales[nowh], min = censor.right[1], max = censor.right[2]) else Rc <- rep(runif(n = n.whales[nowh], min = censor.right[1], max = censor.right[2]), each = n.trials.per.whale)
        
        U <- rep(upper.bound, ifelse(scenario%in%c(1,3), n.whales[nowh], n.trials))

        is.censored <- ifelse(y>Rc, 1, 0)
        y.censored <- y; y.censored[is.censored==1]<-NA
        y.censored
      }
    
    y.dat <- name.list(scenario.id = scenario,
                       input.list = y.dat, 
                       Nwhales = n.whales,
                       Nsim = n.sim,
                       dose.or.ratio = obs.param)
    
    if(verbose) cat("Posterior predictive plots: ")
    
    # The idea behind posterior predictive checking is simple: If a model is a
    # good fit we should be able to use it to generate data that resemble the data
    # that we observed. To generate the data that are used for posterior predictive
    # checks we simulate from the posterior predictive distribution.
    # Posterior predictive checking is mostly qualitative.
    
    # First, sort out the data
    
    y.df <- y.dat %>%
      reshape2::melt() %>%
      tibble::as_tibble(.) %>%
      dplyr::rename(y = value, sim = L3, !!col_names[2] := L2, n = L1) %>% 
      dplyr::arrange_at(., vars(col_names[1:3])) %>% 
      removelabels(scenario.id = scenario, tbl = .) %>% 
      dplyr::select(., !!col_names[1:3], y)

    ppc.df <- expand.grid(n.whales, obs.param) %>% 
      tibble::as_tibble(.) %>% 
      dplyr::rename(n = Var1, !!col_names[2] := Var2) %>% 
      dplyr::arrange_at(., vars(col_names[1:2])) %>% 
      dplyr::mutate(n = addlzero(n), !!col_names[2] := addlzero(!!as.name(col_names[2])))
    
    # Number of plots per page
    
    if(nrow(ppc.df)<=2){
      nrow.per.page <- 2; ncol.per.page <- 1
    }else if(nrow(ppc.df)>2 & nrow(ppc.df)<=4){
      nrow.per.page <- 2; ncol.per.page <- 2
    }else if(nrow(ppc.df)>4){
      nrow.per.page <- 3; ncol.per.page <- 2
    }
    
    # Extract posterior samples for each simulation

    ppc.plots <- try(purrr::map(.x = seq_len(nrow(ppc.df)),
                                .f = ~{
                                  
                                  #' ---------------------------------
                                  # 'True' data y
                                  #' ---------------------------------
                                  
                                  y <- y.df %>% 
                                    dplyr::filter(!!as.name(col_names[2]) == as.character(ppc.df[.x,2]),
                                                  n == as.character(ppc.df$n[.x])) %>%
                                    dplyr::pull(y)
                                  
                                  #' ---------------------------------
                                  # y.rep:
                                  #' ---------------------------------
                                  # An S by N matrix of draws from the posterior 
                                  # predictive distribution, where S is the size of 
                                  # the posterior sample (or subset of the
                                  # posterior sample used to generate yrep) 
                                  # and N is the number of observations (the 
                                  # length of y).
                                  
                                  y.rep <- matrix(nrow = n.yrep, ncol = length(y))
                                  
                                  #' ---------------------------------
                                  # Generate y.rep
                                  #' ---------------------------------
                                  
                                  mu.temp <- mcmc.tbl %>% 
                                    dplyr::filter(n == as.character(ppc.df$n[.x]),
                                                  !!as.name(col_names[2]) == as.character(ppc.df[.x,2])) %>% 
                                    dplyr::select(n, !!col_names[2], sim, mu) %>% 
                                    dplyr::rename(value = mu)
                                  
                                  if(scenario %in% c(1,3)) omega.temp <- mcmc.tbl %>% 
                                    dplyr::filter(n == as.character(ppc.df$n[.x]),
                                                  !!as.name(col_names[2]) == as.character(ppc.df[.x,2])) %>% 
                                    dplyr::select(n, !!col_names[2], sim, omega) %>% 
                                    dplyr::rename(value = omega)
                                  
                                  if(scenario %in% c(2,4)){
                                    
                                    phi.temp <- mcmc.tbl %>% 
                                      dplyr::filter(n == as.character(ppc.df$n[.x]),
                                                    !!as.name(col_names[2]) == as.character(ppc.df[.x,2])) %>% 
                                    dplyr::select(n, !!col_names[2], sim, phi) %>% 
                                    dplyr::rename(value = phi)
                                    
                                    sigma.temp <- mcmc.tbl %>% 
                                      dplyr::filter(n == as.character(ppc.df$n[.x]),
                                                    !!as.name(col_names[2]) == as.character(ppc.df[.x,2])) %>% 
                                    dplyr::select(n, !!col_names[2], sim, sigma) %>% 
                                    dplyr::rename(value = sigma)
                                    
                                    beta.temp <- mcmc.tbl %>% 
                                      dplyr::filter(n == as.character(ppc.df$n[.x]),
                                                    !!as.name(col_names[2]) == as.character(ppc.df[.x,2])) %>%  
                                    dplyr::select(n, !!col_names[2], sim, beta) %>% 
                                    dplyr::rename(value = beta)
                                    
                                    alpha.temp <- mcmc.tbl %>% 
                                      dplyr::filter(n == as.character(ppc.df$n[.x]),
                                                    !!as.name(col_names[2]) == as.character(ppc.df[.x,2])) %>%  
                                      dplyr::select(n, !!col_names[2], sim, alpha) %>% 
                                      dplyr::rename(value = alpha)
                                  
                                    }
                                  
                                  #' ---------------------------------
                                  # Subset of posterior sample
                                  #' ---------------------------------
                                  
                                  for(h in 1:n.yrep){
                                    
                                    sample.ind <- sample(x = 1:nrow(mu.temp), 
                                                         size = length(y))
                                    
                                    mtemp <- mu.temp %>% 
                                      dplyr::pull(value) %>% 
                                      .[sample.ind]
                                    
                                    if(scenario %in% c(1,3)){
                                      otemp <- omega.temp %>% 
                                        dplyr::pull(value) %>% 
                                        .[sample.ind]
                                      }
                                    
                                    if(scenario %in% c(2,4)){
                                      
                                      ptemp <- phi.temp %>% 
                                        dplyr::pull(value) %>% 
                                        .[sample.ind]
                                      
                                      stemp <- sigma.temp %>% 
                                        dplyr::pull(value) %>% 
                                        .[sample.ind]
                                      
                                      btemp <- beta.temp %>% 
                                        dplyr::pull(value) %>% 
                                        .[sample.ind]
                                      
                                      atemp <- alpha.temp %>% 
                                        dplyr::pull(value) %>% 
                                        .[sample.ind]
                                      
                                    }
                                    
                                    #' ---------------------------------
                                    # Generate yrep
                                    #' ---------------------------------
                                    
                                    if(scenario %in% c(1,3)) mu.i <- truncnorm::rtruncnorm(n = 1, 
                                                                                    a = lower.bound, 
                                                                                    b = upper.bound, 
                                                                                    mean = mtemp, 
                                                                                    sd = otemp)
                                    
                                    if(scenario %in% c(2,4)) mu.i <- truncnorm::rtruncnorm(n = 1, 
                                                                                    a = lower.bound, 
                                                                                    b = upper.bound, 
                                                                                    mean = mtemp, 
                                                                                    sd = ptemp)
                                    
                                    #'-------------------------------------------------
                                    # Add covariate effects where appropriate
                                    #'-------------------------------------------------
                                    
                                    nwhales <- as.numeric(removelzero(ppc.df$n[.x]))
                                    
                                    if(scenario %in% c(2,4)){
                                      
                                      #'-----------------------------------------------------
                                      # Generate trials
                                      #'-----------------------------------------------------
                                      
                                      n.trials <- nwhales*n.trials.per.whale
                                      whale.id <- rep(rep(1:nwhales, each = n.trials.per.whale), n.sim)
                                      
                                      #'-----------------------------------------------------
                                      # Previous exposure + signal type
                                      #'-----------------------------------------------------
                                      
                                      is.exposed <- rep(rep(c(0, rep(1, n.trials.per.whale-1)), nwhales), n.sim)
                                      is.mfas <- sample.int(n = 2, size = n.sim*n.trials, replace = TRUE)-1
                                      is.lfas <- as.numeric(!is.mfas)
                                      
                                      mu.ij <- mu.i + 
                                        is.exposed*atemp - 
                                        is.mfas*(btemp/2) + # MFAS
                                        is.lfas*(btemp/2) # LFAS
                                      
                                      t.ij <- truncnorm::rtruncnorm(n = n.sim*n.trials, 
                                                                    a = lower.bound, 
                                                                    b = upper.bound, 
                                                                    mean = mu.ij, 
                                                                    sd = stemp)}
                                    
                                    #'-------------------------------------------------
                                    # Assign tag types to each whale according to tag ratio
                                    #'-------------------------------------------------
                                    # 0 = DTAG, 1 = SAT
                                    
                                    if(scenario %in% c(3,4)){
                                      
                                    attached.tags <- rep(0, n.sim*nwhales*n.trials.per.whale)
                                    attached.tags[sample(x = seq_along(attached.tags), 
                                    size = round(length(attached.tags)*obs.param[which(obs.param == as.numeric(removelzero(ppc.df[.x,2])))]/100),
                                    replace = FALSE)] <- 1}
                                    
                                    #'-------------------------------------------------
                                    # Positional uncertainty
                                    #'-------------------------------------------------
                                    
                                    if(scenario == 3){
                                    argos.correction <- purrr::map(.x = mu.i[attached.tags==1], 
                                                                     .f = ~xy_error(argos.data = argos,
                                                                                    received.lvl = .x, 
                                                                                    source.lvl = source.level, 
                                                                                    multi = FALSE, plot.ellipse = FALSE))
                                             
                                             # mu.i[attached.tags==1] <- purrr::map_dbl(.x = argos.correction, "mean")
                                             sd.RL <- purrr::map_dbl(.x = argos.correction, "sd")}
                                    
                                    if(scenario == 4){
                                      argos.correction <- purrr::map(.x = t.ij[attached.tags==1], 
                                                                     .f = ~xy_error(argos.data = argos,
                                                                                    received.lvl = .x, 
                                                                                    source.lvl = source.level, 
                                                                                    multi = FALSE, plot.ellipse = FALSE))
                                             
                                             # t.ij[attached.tags==1] <- purrr::map_dbl(.x = argos.correction, "mean")
                                             sd.RL <- purrr::map_dbl(.x = argos.correction, "sd")}
                                    
                                    if(scenario %in% c(3,4)){
                                    uncertainty.RL <- rep(dtag.sd, length(y))
                                    uncertainty.RL[which(attached.tags==1)] <- sd.RL}
                                    
                                    #'-------------------------------------------------
                                    # Observations
                                    #'-------------------------------------------------
                                    
                                    if(scenario == 1) yrep <- sapply(X = mu.i, 
                                    FUN = function(x) stats::rnorm(n = 1, mean = x, 
                                                                   sd = as.numeric(removelzero(ppc.df[.x,2]))))
                                    
                                    if(scenario == 2) yrep <- sapply(X = t.ij, 
                                    FUN = function(x) stats::rnorm(n = 1, mean = x, 
                                                                   sd = as.numeric(removelzero(ppc.df[.x,2]))))
                                    
                                    if(scenario == 3) yrep <- purrr::map2_dbl(.x = mu.i,
                                                                              .y = uncertainty.RL,
                                                                              .f = ~stats::rnorm(n = 1, mean = .x, sd = .y))
                                    
                                    if(scenario == 4) yrep <- purrr::map2_dbl(.x = t.ij,
                                                                              .y = uncertainty.RL,
                                                                              .f = ~stats::rnorm(n = 1, mean = .x, sd = .y))
                                    
                                    # Right-censoring
                
                                    Rc.rep <- runif(n = length(yrep), 
                                                    min = censor.right[1], 
                                                    max = censor.right[2])
                                    
                                    is.yrep.censored <- ifelse(yrep > Rc.rep, 1, 0)
                                    yrep.censored <- yrep
                                    yrep.censored[is.yrep.censored==1] <- NA
                                    
                                    y.rep[h,] <- yrep.censored
                                    
                                  } # End h loop

                                  #' ---------------------------------
                                  # Density plots
                                  #' ---------------------------------
                                 
                                  ppc_densoverlay(y = y, y.rep = y.rep, n.yrep = n.yrep, 
                                                   lbound = lower.bound, ubound = upper.bound) +
                                    ggplot2::ggtitle(paste0("n = ", as.numeric(removelzero(ppc.df$n[.x])), " whales\n", 
                                                            ifelse(scenario %in% c(1,2), "SD(dose) = ", "SAT = "),
                                                            ifelse(scenario %in% c(3,4), as.numeric(removelzero(ppc.df[.x,2])), as.numeric(removelzero(ppc.df[.x,2]))),
                                                            ifelse(scenario %in% c(1,2), " dB", " %"))) }) %>% 
                       gridExtra::marrangeGrob(grobs = ., 
                                               nrow = nrow.per.page, 
                                               ncol = ncol.per.page), 
                     silent = TRUE)

    #' ---------------------------------
    # Save plots to file
    #' ---------------------------------
    
    if(class(ppc.plots)[1]== "try-error"){
      ppcmsg <- "An error occurred. Plots could not be generated."
      if(verbose) cat(crayon::yellow("\u2717 ", ppcmsg, "\n"))
    }else{
      ppcmsg <- "Plots saved to disk."
      if(verbose) cat(crayon::green("\u2713 ", ppcmsg, "\n"))
      suppressMessages(ggpubr::ggexport(ppc.plots, filename = paste0(getwd(), "/out/scenario_", scenario, "/S", scenario, "_ppcPlots_", index.sim, ".pdf")))}
      
} # End if posterior.checks

# Outputs -----------------------------------------------------------------
  
  if(verbose){
    cat(crayon::bold(paste0("\nOutputs:\n")))
    cat(crayon::bold(paste0("-------------------------------------\n")))}
  
  #'--------------------------------------------------------------------
# Percent relative biases- -----------------------------------------------------------
  
  if(verbose) cat("Calculating percent relative (mean/median) bias ...")
    
    mcmc.results <- mcmc.results %>%
      dplyr::group_by(param) %>% 
      dplyr::mutate(truth = get(paste0("true.", param))) %>% 
      dplyr::mutate(prb = 100 * (post.mean-truth)/truth,
                    prmb = 100 * (post.median-truth)/truth,
                    err.prb = 100 * (err.mean - true.err)/true.err,
                    err.prmb = 100 * (err.median - true.err)/true.err) %>% 
      dplyr::ungroup() %>% 
      dplyr::select(-truth)
    
  if(verbose) cat(crayon::green(" \u2713\n"))
  
# Final output ------------------------------------------------------------

  mcmc.results <- mcmc.results %>% 
    dplyr::left_join(x = ., y = ess.tbl, by = c("n", col_names[2], "sim", "param"))

  mcmc.results <- mcmc.results %>% 
    dplyr::left_join(x = ., y = ppo.tbl, by = c("n", col_names[2], "sim", "param"))
  
  #'--------------------------------------------------------------------
  #' Summaries 
  #'--------------------------------------------------------------------
  
  if(verbose) cat("Compiling results ...")

  mcmc.summary <- mcmc.results %>% 
    dplyr::group_by_at(., vars(col_names[1:2], param)) %>% 
    dplyr::summarise(post.mean = mean(post.mean),
                     post.median = mean(post.median),
                     post.sd = mean(post.sd),
                     prb = mean(prb),
                     prmb = mean(prmb),
                     post.lower = mean(post.lower),
                     post.upper = mean(post.upper),
                     coverage.perc = 100*sum(param.coverage)/!!n.sim,
                     err.mean = mean(err.mean),
                     err.median = mean(err.median),
                     err.sd = mean(err.sd),
                     err.prb = mean(err.prb),
                     err.prmb = mean(err.prmb),
                     err.low = mean(err.low),
                     err.up = mean(err.up),
                     err.coverage.perc = 100*sum(err.coverage)/!!n.sim)
  
  #'--------------------------------------------------------------------
  #' Text file summary of the simulation
  #'--------------------------------------------------------------------  
  
  if(record.time){
    run.time <- tictoc::toc(quiet = TRUE)
    run.time <- as.numeric(round(run.time$toc-run.time$tic, 0))
    run.time <- hms::as_hms(run.time)}
  
  if(save.textsummary)  
  txt.summary <- tibble::tibble(item = c("SIMULATIONS:",
           "---------------",
           "Scenario:", 
           "Date/Time:",
           "Run time:",
           "",
           "PARAMETERS",
           "---------------",
           "Number of whales:",
           ifelse(scenario%in%c(1,2), "Uncertainty around dose (dB):", "Proportion of SAT tags (%):"),
           ifelse(scenario%in%c(3,4), "Species tagged: ", ""),
           "Mean response threshold (all whales):",
           "Minimum response threshold:",
           "Maximum response threshold:",
           "Right-censoring:",
           ifelse(scenario%in%c(1,3), "Overall (between and within-whale) variation:", 
                  "Between-whale variation:"),
           ifelse(scenario%in%c(1,3), "", "Within-whale variation:"),
           ifelse(scenario%in%c(1,3), "", "MFAS effect:"),
           ifelse(scenario%in%c(1,3), "", "Exposure effect:"),
           "",
           "MCMC",
           "---------------",
           "Nsim:",
           "Burn-in:",
           "Chains:",
           ifelse(!mcmc.auto, "Chain length:", "Max chain length:"),
           "Thinning rate:",
           "Number of posterior samples:",
           "Auto-correlation:",
           "Effective sample size(s):",
           "",
           "POSTERIOR",
           "---------------",
           "Convergence:",
           "Gelman-Rubin:",
           "Trace plots:",
           "Posterior predictive checks:",
           "PPC plots:"), 
           
           value = c("", "", 
                     scenario, 
                     as.character(Sys.time()), 
                     as.character(run.time),
                     "", "", "", 
                     paste0(n.whales, collapse = ", "),
                     ifelse(scenario%in%c(1,2), paste0(uncertainty.dose, collapse = ", "), 
                            paste0(prop.sat, collapse = ", ")),
                     ifelse(scenario%in%c(3,4), "species.argos", ""),
                     paste(true.mu, "dB"),
                     paste(lower.bound,  "dB"),
                     paste(upper.bound, "dB"),
                     paste0(censor.right[1], "-", censor.right[2], " dB"),
                     ifelse(scenario%in%c(1,3), paste(true.omega,  "dB"), paste(true.phi, "dB")),
                     ifelse(scenario%in%c(1,3), "", paste(true.sigma, "dB")),
                     ifelse(scenario%in%c(1,3), "", paste(true.beta, "dB", "(+/-", beta.sd, "SD)")),
                     ifelse(scenario%in%c(1,3), "", paste(true.alpha, "dB", "(+/-", alpha.sd, "SD)")),
                     "", "", "", 
                     n.sim, 
                     paste0(burn.in, collapse = "; "), 
                     mcmc.chains, 
                     ifelse(!mcmc.auto, mcmc.n, round(max(chain.lengths), 0)),
                     mcmc.thin, 
                     mcmc.n,
                     autocorr.msg,
                     as.character(paste0(params.monitored, " (", ess.msg, ")", collapse = "; ")),
                     "", "", "", 
                     ifelse(check.convergence, convergence.msg, ""),
                     mcmc.gelmanrubin,
                     ifelse(check.convergence, "Saved to disk", "Not saved"),
                     ifelse(posterior.checks, ppo.msg, ""),
                     ifelse(posterior.checks, "Saved to disk", "Not saved"))) %>% 
    
    write.table(x = ., file = paste0(getwd(),"/out/scenario_", scenario,"/sim_summary_", index.sim, ".txt"),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  #'--------------------------------------------------------------------
  # Combine everything into a list
  #'--------------------------------------------------------------------

  mcmc.final <- list(mcmc = mcmc.results,
                     summary = mcmc.summary,
                     dose.response = list(dose.range = dose.range,
                                          p = doseresp.avg,
                                          q = quants),
                     cor = mcmc.cor,
                     col_names = col_names,
                     ppo = ppo.tbl,
                     params = list(scenario = scenario,
                                   index = as.character(index.sim),
                                   params.monitored = params.monitored,
                                   n.sim = n.sim, 
                                   n.whales = n.whales, 
                                   true.mu = true.mu,
                                   lower.bound = lower.bound,
                                   upper.bound = upper.bound,
                                   true.ERR = true.err,
                                   burn.in = burn.in,
                                   mcmc.n = mcmc.n,
                                   mcmc.thin = mcmc.thin,
                                   mcmc.chains = mcmc.chains,
                                   mcmc.auto = mcmc.auto,
                                   mcmc.save = mcmc.save,
                                   chain.lengths = chain.lg,
                                   parallel.cores = parallel.cores))
  
  if(exists("internal.call")) mcmc.final$tbl = mcmc.tbl.saved
  
  if(scenario %in% c(1,2)) mcmc.final$params$uncertainty.dose <- uncertainty.dose
  if(scenario %in% c(3,4)) mcmc.final$params$prop.sat <- prop.sat
  
  if(scenario %in% c(1,3)) {
    mcmc.final$params$true.omega <- true.omega
    mcmc.final$params$omega.upper.bound <- omega.upper.bound} 
  
  if(scenario %in% c(2,4)) {
    mcmc.final$params$true.phi <- true.phi
    mcmc.final$params$true.sigma <- true.sigma
    mcmc.final$params$phi.upper.bound <- phi.upper.bound
    mcmc.final$params$sigma.upper.bound <- sigma.upper.bound
    mcmc.final$params$true.alpha <- true.alpha
    mcmc.final$params$alpha.sd <- alpha.sd
    mcmc.final$params$true.beta <- true.beta
    mcmc.final$params$beta.sd <- beta.sd}

  # if(mcmc.save) mcmc.final$samples <- mcmc.samples
  
  if(check.convergence){
    mcmc.final$convergence$gelmanR <- mcmc.gelmanrubin
    mcmc.final$convergence$rhat <- gelman.rubin
    mcmc.final$convergence$fail_count <- gelman.check
    mcmc.final$convergence$fail <- gelman.rubin %>% dplyr::filter(rhat>mcmc.gelmanrubin)}
  
  if(posterior.checks) mcmc.final$ppcheck <- list(ydat = y.df)

  if(verbose) cat(crayon::green(" \u2713\n"))
  if(verbose) cat(crayon::green("\nDone!\n"))
  
  if(record.time){
    mcmc.final$run.time <- run.time
    print(run.time)}
  
  #'--------------------------------------------------------------------
  #' Stop parallel cluster
  #'--------------------------------------------------------------------
  
  stop_cluster()
  
  #'--------------------------------------------------------------------
  #' Save results to disk if required
  #'--------------------------------------------------------------------
  
  if(save.results) saveRDS(mcmc.final, 
    file = paste0(getwd(),"/out/scenario_", scenario,"/mcmc.res_", index.sim, ".rds"))
  
  return(mcmc.final)

} # End run_scenario

#'--------------------------------------------------------------------
# Convenience function to run extra simulations if convergence not achieved
#'--------------------------------------------------------------------

extra_sim <- function(mcmc.object, replace.sims = TRUE, update.dr = FALSE){

  #'--------------------------------------------------------------------
  # PARAMETERS
  #'--------------------------------------------------------------------
  #' @param mcmc.object List. Output from the \code{run_scenario()} or the \code{compile_sim} functions.
  #' @param replace.sims Logical. If TRUE, update the results stored in mcmc.object.
  #' @param update.dr Logical. If TRUE, dose-response curves are re-calculated from the MCMC sample objects
  #' saved on disk (requires mcmc.save = TRUE in run_scenario()). 
  #'---------------------------------------------
  
  #'--------------------------------------------------------------------
  # Function checks
  #'--------------------------------------------------------------------
  
  f.checks <- ArgumentCheck::newArgCheck() # Create object to store error/warning messages
  
  # Errors
  
  if (is.null(mcmc.object$convergence)) 
    ArgumentCheck::addError(
      msg = "Convergence diagnostics not available. Set check.convergence to TRUE in run_scenario().",
      argcheck = f.checks
    )
  
  if(nrow(mcmc.object$convergence$fail)==0)
    ArgumentCheck::addError(
      msg = "All simulations have converged. No additional simulations needed.",
      argcheck = f.checks
    )
  
  if(any(mcmc.object$params$mcmc.save==0) & update.dr)
    ArgumentCheck::addError(
      msg = "MCMC samples cannot be found. update.dr must be set to FALSE.",
      argcheck = f.checks
    )
  
  ArgumentCheck::finishArgCheck(f.checks) # Return errors and warnings (if any)
  
  internal.call <<- TRUE
  
  #'--------------------------------------------------
  # Extract relevant parameters
  #'--------------------------------------------------
  
  scenario <- mcmc.object$params$scenario
  col_names <- mcmc.object$col_names
  dose.range <- mcmc.object$dose.response$dose.range
  params.monitored <- mcmc.object$params$params.monitored
  n.sim <- mcmc.object$params$n.sim
  lower.bound <- mcmc.object$params$lower.bound
  upper.bound <- mcmc.object$params$upper.bound
  mcmc.n <- max(mcmc.object$params$mcmc.n)
  
  #'--------------------------------------------------
  # Identify simulations in which convergence was not 
  # achieved for all parameters
  #'--------------------------------------------------

  not.converged <- mcmc.object$convergence$fail_count %>% 
    dplyr::filter(not.converged>0) %>% 
    dplyr::mutate(n = as.numeric(removelzero(n)),
                  !!col_names[2] := as.numeric(removelzero(!!as.name(col_names[2]))))
  
  #'--------------------------------------------------
  # Run additional simulations 
  #'--------------------------------------------------
  
  extra.sims <- list()
  
  for(i in 1:nrow(not.converged)){ 
    
    cat(paste0("Running ", not.converged$not.converged[i], 
               " additional simulations (n = ", not.converged$n[i], "; ",
               col_names[2], " = ", as.numeric(unlist(not.converged[,2])[i]), ")\n"))
    
    extra.sims[[i]] <- run_scenario(scenario = scenario,
                                    n.sim = not.converged$not.converged[i], 
                                    n.whales = not.converged$n[i],
                                    uncertainty.dose = as.numeric(unlist(not.converged[,2])[i]),
                                    prop.sat = as.numeric(unlist(not.converged[,2])[i]),
                                    mcmc.auto = TRUE,
                                    mcmc.n = max(mcmc.object$params$mcmc.n),
                                    burn.in = max(mcmc.object$params$burn.in),
                                    mcmc.thin = mcmc.object$params$mcmc.thin,
                                    mcmc.chains = mcmc.object$params$mcmc.chains,
                                    no.tracePlots = ,
                                    parallel.cores = mcmc.object$params$parallel.cores, 
                                    check.convergence = TRUE,
                                    save.trace = FALSE,
                                    verbose = FALSE,
                                    save.results = FALSE,
                                    save.textsummary = FALSE,
                                    record.time = FALSE)
  }

  match.sims <- extra.sims[[1]]$mcmc %>% 
    dplyr::select_at(., tidyselect::all_of(c("n", col_names[2], "sim"))) %>% 
    dplyr::distinct(.)
  
  match.mcmc <- mcmc.object$convergence$fail %>% 
    dplyr::select_at(., tidyselect::all_of(c("n", col_names[2], "sim"))) %>% 
    dplyr::distinct(.) %>% 
    dplyr::rename(init_sim = sim)
  
  matched <- cbind(match.mcmc, sim = match.sims$sim)
  matched$sim <- as.character(matched$sim)
  
  extra.sims[[1]]$mcmc <- dplyr::left_join(extra.sims[[1]]$mcmc, matched, by = c("n", col_names[2], "sim")) %>% 
    dplyr::select(-sim) %>% 
    dplyr::rename(sim = init_sim)
  
  extra.sims[[1]]$cor <- dplyr::left_join(extra.sims[[1]]$cor, matched, by = c("n", col_names[2], "sim")) %>% 
    dplyr::select(-sim) %>% 
    dplyr::rename(sim = init_sim)
  
  extra.sims[[1]]$ppo <- dplyr::left_join(extra.sims[[1]]$ppo, matched, by = c("n", col_names[2], "sim")) %>% 
    dplyr::select(-sim) %>% 
    dplyr::rename(sim = init_sim)
  
  extra.sims[[1]]$tbl <- dplyr::left_join(extra.sims[[1]]$tbl, matched, by = c("n", col_names[2], "sim")) %>% 
    dplyr::select(-sim) %>% 
    dplyr::rename(sim = init_sim)
  
  #'--------------------------------------------------
  # Update results
  #'--------------------------------------------------  
  
  if(replace.sims){
  
    revised.sims <- mcmc.object
    
    # mcmc tibble
    
    revised.sims$mcmc <- revised.sims$mcmc %>% 
      dplyr::mutate(rcomb = paste0(n, "-", !!as.name(col_names[2]), "-", sim))
    
    revised.sims$convergence$fail <- revised.sims$convergence$fail %>% 
      dplyr::mutate(rcomb = paste0(n, "-", !!as.name(col_names[2]), "-", sim))
    
    revised.sims$mcmc <- revised.sims$mcmc %>% 
      dplyr::filter(!rcomb%in%unique(revised.sims$convergence$fail$rcomb)) %>% 
      dplyr::select(-rcomb) %>% 
      rbind(., purrr::map(.x = extra.sims, "mcmc") %>% do.call(rbind, .))
    
    # Summary tibble  
    
    revised.sims$summary <- revised.sims$mcmc %>% 
      dplyr::group_by_at(., vars(col_names[1:2], param)) %>% 
      dplyr::summarise(post.mean = mean(post.mean),
                       post.median = mean(post.median),
                       post.sd = mean(post.sd),
                       prb = mean(prb),
                       prmb = mean(prmb),
                       post.lower = mean(post.lower),
                       post.upper = mean(post.upper),
                       coverage.perc = 100*sum(param.coverage)/!!n.sim,
                       err.mean = mean(err.mean),
                       err.median = mean(err.median),
                       err.sd = mean(err.sd),
                       err.prb = mean(err.prb),
                       err.prmb = mean(err.prmb),
                       err.low = mean(err.low),
                       err.up = mean(err.up),
                       err.coverage.perc = 100*sum(err.coverage)/!!n.sim)
    
    # Chain auto-correlation
    
    revised.sims$cor <- revised.sims$cor %>% 
      dplyr::mutate(rcomb = paste0(n, "-", !!as.name(col_names[2]), "-", sim)) %>% 
      dplyr::filter(!rcomb%in%unique(revised.sims$convergence$fail$rcomb)) %>% 
      dplyr::select(-rcomb) %>% 
      rbind(., purrr::map(.x = extra.sims, "cor") %>% do.call(rbind, .))
    
    # PPO
    
    revised.sims$ppo <- revised.sims$ppo %>% 
      dplyr::mutate(rcomb = paste0(n, "-", !!as.name(col_names[2]), "-", sim)) %>% 
      dplyr::filter(!rcomb%in%unique(revised.sims$convergence$fail$rcomb)) %>% 
      dplyr::select(-rcomb) %>% 
      rbind(., purrr::map(.x = extra.sims, "ppo") %>% do.call(rbind, .))
    
    # Dose-response functions

    if(update.dr){

      # First retrieve the MCMC samples
      
      list.samples <- list.files(path = paste0(getwd(),"/out/scenario_", scenario), pattern = "mcmc.samples")
      tbl.list <- purrr::map(.x = list.samples, 
                             .f = ~readRDS(paste0(getwd(),"/out/scenario_", scenario, "/", .x))) %>%
                               do.call(rbind, .)

      
    revised.sims$tbl <- tbl.list %>% 
      dplyr::mutate(rcomb = paste0(n, "-", !!as.name(col_names[2]), "-", sim)) %>% 
      dplyr::filter(!rcomb%in%unique(revised.sims$convergence$fail$rcomb)) %>% 
      dplyr::select(-rcomb) %>%
      rbind(., purrr::map(.x = extra.sims, "tbl") %>% do.call(rbind, .))
    
    revised.mcmc_tbl <- revised.sims$tbl %>% 
      split(., f = .$comb) %>%  
      purrr::map(.x = ., .f = ~split(.x, f = .x$sim)) %>% 
      purrr::map_depth(.x = ., .depth = 2, .f = ~split(.x, f = .x$param))

    revised.sims$tbl <- suppressWarnings(revised.sims$tbl  %>% 
                         tidyr::pivot_wider(., names_from = param, 
                                            values_from = value, 
                                            id_cols = tidyselect::all_of(c(!!col_names[1:3], "comb"))) %>% 
                                   tidyr::unnest(cols = params.monitored))
    
    revised.sims$tbl <- revised.sims$tbl %>% 
      dplyr::mutate(comb = paste0(n, "-", !!as.name(col_names[2]), "-", sim))
  
    dose.summary <- revised.sims$tbl %>%
      dplyr::group_by_at(vars(col_names[1:3])) %>% 
      dplyr::summarise_at(params.monitored, mean, na.rm = TRUE) %>% 
      dplyr::mutate(comb = paste0(n, "-", !!as.name(col_names[2]), "-", sim),
                    nR = paste0("n_", n, ifelse(scenario %in% c(1,2), "-RLsd_", "-sat_"), 
                                !!as.name(col_names[2])))
    
    start_cluster(revised.sims$params$parallel.cores)
    
    doseresp.values <- foreach::foreach(co = 1:length(revised.mcmc_tbl)) %:%
      foreach::foreach(pp = 1:n.sim, 
                       .packages = c("truncnorm")) %dopar%{
                         
                         if(scenario %in% c(1,3)){
                           truncnorm::ptruncnorm(q = rep(dose.range, each = mcmc.n),
                                                 a = lower.bound,
                                                 b = upper.bound,
                                                 mean = revised.mcmc_tbl[[co]][[pp]]$mu$value,
                                                 sd = revised.mcmc_tbl[[co]][[pp]]$omega$value)
                           
                         }else{
                           truncnorm::ptruncnorm(q = rep(dose.range, each = mcmc.n),
                                                 a = lower.bound,
                                                 b = upper.bound,
                                                 mean = revised.mcmc_tbl[[co]][[pp]]$mu$value,
                                                 sd = sqrt(revised.mcmc_tbl[[co]][[pp]]$phi$value^2+revised.mcmc_tbl[[co]][[pp]]$sigma$value^2))
                           
                         }}
    
    doseresp.values <- purrr::map_depth(.x = doseresp.values, 
                                        .depth = 2,
                                        .f = ~lapply(1:mcmc.n, FUN = function(x) nth_element(.x, x, mcmc.n))) %>% 
      purrr::set_names(x = ., nm = names(revised.mcmc_tbl)) %>% 
      purrr::map_depth(.x = ., .depth = 1, .f = ~purrr::set_names(x = .x, nm = names(revised.mcmc_tbl[[1]])))
    

    # Define quantiles
    
    quants <- mcmc.object$dose.response$q
    
    doseresp <- foreach::foreach(g = 1:nrow(dose.summary),
                                 .packages = c("magrittr", "tidyverse", "truncnorm"),
                                 .export = c("removelabels")) %dopar% {
                                   
                                   #'----------------------------------------------------
                                   # Extract row indices corresponding to each simulation    
                                   #'----------------------------------------------------
                                   
                                   # row.indices <- which(mcmc.tbl$comb==dose.summary$comb[g])
                                   
                                   gn <- paste0(dose.summary[g,]$n, "-", dose.summary[g,col_names[2]])
                                   gs <- dose.summary[g,]$sim
                                   
                                   #'----------------------------------------------------
                                   # Extract relevant dose-response curves  
                                   #'---------------------------------------------------- 
                                   
                                   # dose_resp <- doseresp.values[row.indices] %>% 
                                   #   do.call(rbind,.)
                                   
                                   dose_resp <- doseresp.values[[gn]][[gs]] %>% do.call(rbind, .)
                                   
                                   #'----------------------------------------------------
                                   # Calculate median dose-response curve   
                                   #'---------------------------------------------------- 
                                   
                                   p.median <- apply(X = dose_resp, MARGIN = 2, FUN = median)
                                   
                                   #'----------------------------------------------------
                                   # Calculate quantiles
                                   #'---------------------------------------------------- 
                                   
                                   q.low <- purrr::map(.x = quants, 
                                                       .f = ~apply(X = dose_resp, MARGIN = 2, 
                                                                   FUN = quantile, (50-.x/2)/100)) %>% 
                                     do.call(rbind,.)
                                   
                                   q.up <- purrr::map(.x = quants, 
                                                      .f = ~apply(X = dose_resp, MARGIN = 2, 
                                                                  FUN = quantile, (50+.x/2)/100)) %>% 
                                     do.call(rbind,.)
                                   
                                   #'----------------------------------------------------
                                   # Return results
                                   #'---------------------------------------------------- 
                                   
                                   list(doseresp.median = p.median, 
                                        doseresp.lower = q.low, 
                                        doseresp.upper = q.up)}
    
    doseresp <- doseresp %>% purrr::set_names(x = ., nm = dose.summary$nR)
    
    #'--------------------------------------------------------------------
    # Take average across simulations
    #'--------------------------------------------------------------------
    
    doseresp.avg <- foreach::foreach(g = unique(dose.summary$nR),
                                     .packages = c("dplyr", "magrittr")) %dopar% {
                                       
                                       #'----------------------------------------------------
                                       # Extract row indices corresponding to n x RL / ratios   
                                       #'----------------------------------------------------  
                                       
                                       # row.indices <- which(dose.summary$nR==g)
                                       sub.l <- doseresp[names(doseresp)==g]
                                       
                                       nR.mean <- purrr::map(.x = sub.l, .f = "doseresp.median") %>% 
                                         do.call(rbind, .) %>% 
                                         apply(X = ., MARGIN = 2, mean)
                                       
                                       # nR.mean <- purrr::map(.x = doseresp, 
                                       #                         .f = "doseresp.median")[row.indices] %>% 
                                       #    do.call(rbind, .) %>% 
                                       #     apply(X = ., MARGIN = 2, mean)
                                       
                                       nR.qlow <- purrr::map(.x = sub.l, .f = "doseresp.lower")
                                       
                                       #'----------------------------------------------------
                                       # Takes the mean over all matrices (by row)
                                       #'----------------------------------------------------  
                                       
                                       nR.q.low <- do.call(cbind, nR.qlow) %>% 
                                         array(., dim = c(dim(nR.qlow[[1]]), length(nR.qlow))) %>% 
                                         apply(., c(1, 2), mean, na.rm = TRUE)
                                       
                                       nR.qup <- purrr::map(.x = doseresp, .f = "doseresp.upper")
                                       
                                       nR.q.up <- do.call(cbind, nR.qup) %>% 
                                         array(., dim = c(dim(nR.qup[[1]]), length(nR.qup))) %>% 
                                         apply(., c(1, 2), mean, na.rm = TRUE)
                                       
                                       list(doseresp.mean = nR.mean,
                                            doseresp.lower = nR.q.low,
                                            doseresp.upper = nR.q.up)}
    
    # Assign names to list elements
    
    doseresp.avg <- purrr::set_names(doseresp.avg, unique(dose.summary$nR))
    
    invisible(stop_cluster())
    
    revised.sims$dose.response <- list(dose.range = dose.range,
                         p = doseresp.avg,
                         q = quants)
    
    } # End update.dr
    
    # Convergence
    
    revised.sims$convergence$rhat <- revised.sims$convergence$rhat %>% 
      dplyr::mutate(rcomb = paste0(n, "-", !!as.name(col_names[2]), "-", sim)) %>% 
      dplyr::filter(!rcomb%in%unique(revised.sims$convergence$fail$rcomb)) %>% 
      dplyr::select(-rcomb) %>% 
      rbind(., purrr::map(.x = extra.sims, "convergence") %>% 
              purrr::map(., "rhat") %>% do.call(rbind, .))
    
    revised.sims$convergence$fail_count <- revised.sims$convergence$fail_count %>% 
      rbind(., purrr::map(extra.sims, "convergence") %>% purrr::map(., "fail_count") %>% do.call(rbind, .)) %>% 
      dplyr::group_by_at(., vars(col_names[1:2])) %>% 
      dplyr::summarise(not.converged = min(not.converged)) %>% 
      dplyr::ungroup()
    
    revised.sims$convergence$fail <- purrr::map(extra.sims, "convergence") %>% 
      purrr::map(., "fail") %>% do.call(rbind, .)
    
    }
  
  cat("Done!\n")
  suppressWarnings(rm(internal.call))
  if(replace.sims) return(revised.sims) else return(extra.sims)
    
} # End extra_sim()

#'--------------------------------------------------------------------
# Function to create a forest plot of average posterior medians (+CI) 
# across a range of sample sizes and measurement error values
#'--------------------------------------------------------------------

# Each sample size is given a hue along the viridis colour scale.
# Each measurement error value is given a different shade,
# with lower values shown in lighter tones, and higher values in darker tones.
# Cannot use opacity (alpha) to achieve this effect - the trick here is 
# therefore to calculate the HEX colour value corresponding to different transparency
# levels of the same hue (see hexa2hex function)

plot_results <- function(mcmc.object, 
                         layout.ncol = 1,
                         pars.to.plot = NULL,
                         select.n = NULL,
                         select.obs = NULL,
                         summary.method = "mean",
                         start.shade = 0.2,
                         n.cols = 12,
                         darken.bars = FALSE,
                         save.to.disk = TRUE,
                         save.individual.plots = FALSE,
                         output.format = "pdf"){
  
  #'--------------------------------------------------------------------
  # PARAMETERS
  #'--------------------------------------------------------------------
  #' @param mcmc.object List. Output from the \code{run_scenario()} function.
  #' @param layout.ncol Integer between 1 and 3. Number of columns used in the final plot layout.
  #' @param pars.to.plot Parameter(s) of interest. By default, the function will produce plots for every monitored parameter (param = NULL). 
  #' @param select.n Subset of sample sizes to display. All values are shown when set to NULL (the default). 
  #' @param select.obs Subset of values to display for the observation model parameter (i.e. uncertainty.dose in scenarios 1 and 2, prop.sat in scenarios 3 and 4). All values are shown when set to NULL (the default). 
  #' @param summary.method Character vector. One of "mean" or "median. Whether to calculate the average or median value of posterior statistics across simulations.
  #' @param start.shade Number between 0 and 1 indicating the shade of the lightest to use on the Y-axis. Higher values indicate darker shades. Defaults to 0.2.
  #' @param n.cols Integer. Minimum number of colours used to define the colour palettes of the heat plots.
  #' @param darken.bars Logical. Whether to add dark lines to the plot to enhance the legibility of the lightest colours.  Defaults to FALSE.
  #' @param save.to.disk Logical. Whether to save the plots to disk. Defaults to FALSE.
  #' @param save.individual.plots Logical. If TRUE, individual heat plots are saved separately on disk.
  #' @param output.format Output file type. Defaults to "pdf".
  #'---------------------------------------------
  
  #'--------------------------------------------------------------------
  # Function checks
  #'--------------------------------------------------------------------
  
  f.checks <- ArgumentCheck::newArgCheck() # Create object to store error/warning messages
  
  if (!dir.exists(file.path(paste0(getwd(),"/out/scenario_", mcmc.object$params$scenario))))
    dir.create(path = file.path(paste0(getwd(),"/out/scenario_", mcmc.object$params$scenario)), showWarnings = FALSE)
  
  # Errors
  
  if (start.shade == 0) 
    ArgumentCheck::addError(
      msg = "start.shade must be greater than 0.",
      argcheck = f.checks
    )
  
  if (start.shade > 1) 
    ArgumentCheck::addError(
      msg = "start.shade cannot be greater than 1.",
      argcheck = f.checks
    )
  
  if (!summary.method %in% c("mean", "median")) 
    ArgumentCheck::addError(
      msg = "Unrecognised summary method.",
      argcheck = f.checks
    )
  
  if(!is.null(pars.to.plot)){
  if(mcmc.object$params$scenario %in% c(1,2)){
    if (!pars.to.plot %in% c("mu", "omega")) 
      ArgumentCheck::addError(
        msg = "Parameter(s) not recognised.",
        argcheck = f.checks
      )
  }}
  
  if(!is.null(pars.to.plot)){
    if(mcmc.object$params$scenario %in% c(1,2)){
      if (!pars.to.plot %in% c("mu", "phi", "sigma", "beta", "alpha")) 
        ArgumentCheck::addError(
          msg = "Parameter(s) not recognised.",
          argcheck = f.checks
        )
    }}
  
  if (layout.ncol <= 0 | layout.ncol >3)
    ArgumentCheck::addError(
      msg = "layout.ncol must be an integer between 1 and 3.",
      argcheck = f.checks
    )
  
  ArgumentCheck::finishArgCheck(f.checks) # Return errors and warnings (if any)
  
  #'--------------------------------------------------------------------
  # Retrieve simulation parameters
  #'--------------------------------------------------------------------
  
  scenario <- mcmc.object$params$scenario
  index <- mcmc.object$params$index

  n.sim <- mcmc.object$params$n.sim
  col_names <- mcmc.object$col_names
  if(is.null(pars.to.plot)) params.monitored <- mcmc.object$params$params.monitored else params.monitored <- pars.to.plot
  
  #'--------------------------------------------------------------------
  # Filter data where appropriate
  #'--------------------------------------------------------------------
  
  if(!is.null(select.n)) n.whales <- select.n else n.whales <- mcmc.object$params$n.whales
  
  if(!is.null(select.obs)) {
    obs.param <- select.obs
  }else{
    if(scenario %in% c(1,2)) obs.param <- mcmc.object$params$uncertainty.dose
    if(scenario %in% c(3,4)) obs.param <- mcmc.object$params$prop.sat 
  }
  
  mcmc.object$mcmc <- mcmc.object$mcmc %>% 
    dplyr::filter(n %in% addlzero(n.whales), !!as.name(col_names[2]) %in% addlzero(obs.param))
  
  #'--------------------------------------------------------------------
  # Retrieve all combinations of n.whales x uncertainty.dose
  # and assign appropriate colours to each
  #'--------------------------------------------------------------------
  
  ridge.colors <- expand.grid(n.whales, obs.param) %>% 
    dplyr::rename(n = Var1, !!col_names[2] := Var2) %>% 
    dplyr::mutate(n = addlzero(n), !!col_names[2] := addlzero(!!as.name(col_names[2]))) %>%
    dplyr::arrange_at(., vars(col_names[1:2]))
  
  #'--------------------------------------------------------------------
  # Define hues and opacity levels
  #'--------------------------------------------------------------------
  
  viridis.modified <- pals::viridis(n = length(unique(ridge.colors$n)))   # Modified viridis colour ramp
  viridis.modified[length(viridis.modified)] <- c("#ffad08") # ffbb00
  
  primary.colours <- tibble::lst(alpha = seq(start.shade, 1, 
                                             length.out = length(unique(ridge.colors[,col_names[2]]))),
                                 col = viridis.modified) %>% 
    purrr::cross_df()
  
  #'--------------------------------------------------------------------
  # Calculate colours (emulate opacity without actually making colours transparent)
  #'--------------------------------------------------------------------
  
  ridge.colors <- cbind(ridge.colors, primary.colours) %>% 
    dplyr::mutate(., tcol = purrr::map2_chr(.x = .$col, 
                                            .y = .$alpha, 
                                            .f = ~hexa2hex(input.colour = .x, opacity = .y, bg.colour = "white"))) %>% 
    
    dplyr::mutate(n = as.character(n), 
                  !!col_names[2] := as.character(!!as.name(col_names[2]))) %>% 
    tibble::as_tibble(.)
  
  #'--------------------------------------------------------------------
  # Add a darkening effect by emulating transparency over black background
  #'--------------------------------------------------------------------
  
  no.dark <- round(length(obs.param)/2, 0) # Number of bins affected from the top
  darkRL <- obs.param[(length(obs.param)-(no.dark-1)):length(obs.param)]
  
  ridge.colors$darkalpha <- rep(c(rep(1, length(obs.param)-no.dark), 
                                  seq(from = 0.95, to = 0.7, length.out = no.dark)), 
                                times = length(n.whales))
  
  ridge.colors <- dplyr::mutate(.data = ridge.colors, 
                                tcol = hexa2hex(input.colour = tcol, opacity = darkalpha, bg.colour = "black"))
  
  #'--------------------------------------------------------------------
  # Create a bivariate legend for the forest plots
  #'--------------------------------------------------------------------
  
  # n x SD(dose) / tag ratio combinations
  # Convert to sequential so that tiles are evenly spaced
  
  lgd <- expand.grid(n.whales, obs.param) %>% 
    tibble::as_tibble(.) %>% 
    dplyr::rename(n = Var1, !!col_names[2] := Var2) %>% 
    dplyr::arrange_at(., vars(col_names[1:2])) %>% 
    dplyr::mutate(n_num = as.numeric(as.factor(n))) %>% 
    dplyr::mutate(o_num = as.numeric(as.factor(!!as.name(col_names[2])))) %>% 
    dplyr::mutate(x = n_num - 0.5, y = o_num -0.5) %>% 
    dplyr::mutate(n = addlzero(n), !!col_names[2] := addlzero(!!as.name(col_names[2])))
  
  #'--------------------------------------------------------------------
  # Generate data for plotting
  #'--------------------------------------------------------------------
  
  cat(paste0("Preparing data ...\n"))
  
  params.monitored <- c(params.monitored, "ERR")
  
  make_dat <- function(input.data){
    
    purrr::map(.x = params.monitored,
               .f = ~{
                 
                 #'--------------------------------------------------------------------
                 # True values
                 #'--------------------------------------------------------------------
                 
                 if(!.x == "ERR") true.value <- input.data$params[[paste0("true.", .x)]]
                 if(.x == "ERR") true.value <- input.data$params$true.ERR
                 
                 #'--------------------------------------------------------------------
                 # Filter rows and columns
                 #'--------------------------------------------------------------------
                 
                 if(.x=="ERR"){
                   
                   post.est <- input.data$mcmc %>% 
                     dplyr::select(-contains("param"), -contains("post"), -prb, -prmb) %>% 
                     dplyr::rename(post.mean = err.mean, post.median = err.median, post.sd = err.sd,
                                   post.lower = err.low, post.upper = err.up, param.coverage = err.coverage,
                                   prb = err.prb, prmb = err.prmb)
                   
                 }else{
                   
                   post.est <- input.data$mcmc %>% 
                     dplyr::filter(param == .x) %>%
                     dplyr::select(-dplyr::contains("err"))
                   
                 }
                 
                 #'--------------------------------------------------------------------
                 # Match to alphas
                 #'--------------------------------------------------------------------
                 
                 post.est <- post.est %>%
                   dplyr::left_join(x = ., 
                                    y = ridge.colors[, c("n", col_names[2], "alpha")], 
                                    by = c("n", col_names[2]))
                 
                 #'--------------------------------------------------------------------
                 # Assign colours to data
                 #'--------------------------------------------------------------------
                 
                 suppressWarnings(post.est <- post.est %>% 
                                    dplyr::left_join(x = ., 
                                                     y = ridge.colors, by = c("n", col_names[2], "alpha")) %>% 
                                    dplyr::mutate(comb = paste0("n = ", removelzero(n), "; ", col_names[2], " = ", removelzero(!!as.name(col_names[2])))))
                 
                 #'--------------------------------------------------------------------
                 # Calculate mean values for plotting
                 #'--------------------------------------------------------------------
                 
                 if(summary.method=="mean"){
                   
                   post.forest <- post.est %>% 
                     dplyr::group_by(comb) %>% 
                     dplyr::mutate(dot.mean = mean(post.median),
                                      int.low = mean(post.lower),
                                      int.high = mean(post.upper)) %>% 
                     dplyr::ungroup()}
                 
                 if(summary.method=="median"){
                   
                   post.forest <- post.est %>% 
                     dplyr::group_by(comb) %>% 
                     dplyr::mutate(dot.mean = median(post.median),
                                      int.low = median(post.lower),
                                      int.high = median(post.upper)) %>% 
                     dplyr::ungroup()}
                 
                 
                 #'--------------------------------------------------------------------
                 # Define order for plotting
                 #'--------------------------------------------------------------------
                 
                 combs <- expand.grid(n = unique(post.est$n), 
                                      alpha = unique(post.est$alpha)) %>% 
                   tibble::as_tibble(.) %>% 
                   dplyr::arrange(n, alpha) %>% 
                   dplyr::mutate(index = 1:nrow(.)) %>% 
                   dplyr::mutate(n = as.character(n))
                 
                 post.forest <- post.forest %>% 
                   dplyr::left_join(x = ., y = combs, by = c('n', 'alpha'))
                 
                 #'--------------------------------------------------------------------
                 # Relabel factors for correct order during plotting
                 #'--------------------------------------------------------------------
                 
                 post.forest$ord <- factor(post.forest$tcol, 
                                           levels = unique(post.forest$tcol[order(as.numeric(post.forest$index))]))
                 post.forest$comb <- factor(post.forest$comb, 
                                            levels = unique(post.forest$comb[order(as.numeric(post.forest$index))]))
                 
                 #'--------------------------------------------------------------------
                 # Retrieve colours
                 #'--------------------------------------------------------------------
                 
                 lgd <- lgd %>%
                   dplyr::left_join(x = ., y = ridge.colors, by = c("n", col_names[2])) %>%
                   dplyr::mutate(index = 1:nrow(.))
                 
                 #'--------------------------------------------------------------------
                 # Make sure order of colours is correct
                 #'--------------------------------------------------------------------
                 
                 lgd$ord <- factor(lgd$tcol, levels = lgd$tcol[order(as.numeric(lgd$index))])
                 
                 #'--------------------------------------------------------------------
                 # Data for heat plots
                 #'--------------------------------------------------------------------
                 
                 # Credible interval widths
                 
                 ciw <- post.forest %>% 
                   dplyr::mutate(cred.width = post.upper - post.lower) %>% 
                   dplyr::group_by_at(., vars(col_names[1:2])) %>% 
                   dplyr::summarise(cred.width = mean(cred.width)) 
                 
                 # Percentage mean bias
                 
                 percmb <- post.forest %>% 
                   dplyr::group_by_at(., vars(col_names[1:2])) %>% 
                   dplyr::summarise(prb = mean(abs(prb))) 
                 
                 # PPO
                 
                 if(!is.null(input.data$ppo)){
                 ppo.summary <- input.data$ppo %>% 
                   dplyr::filter(param == .x) %>% 
                   dplyr::group_by_at(., vars(col_names[1:2])) %>% 
                   dplyr::summarise(ppo = mean(ppo))
                 }else{
                     ppo.summary <- NULL
                   }
                 
                 # Add to legend
                 
                 if(!is.null(input.data$ppo)){
                   
                 lgd <- dplyr::left_join(x = lgd, y = ciw, by = c("n", col_names[2])) %>% 
                   dplyr::left_join(x = ., y = percmb, by = c("n", col_names[2])) %>% 
                   dplyr::left_join(x = ., y = ppo.summary, by = c("n", col_names[2]))
                 
                 }else{
                
                lgd <- dplyr::left_join(x = lgd, y = ciw, by = c("n", col_names[2])) %>% 
                    dplyr::left_join(x = ., y = percmb, by = c("n", col_names[2]))  
                   }
                 
                 return(list(post.forest, lgd, trueval = true.value))
               })
  }
  
  plot.dat <- make_dat(input.data = mcmc.object) %>% 
    purrr::set_names(x = ., nm = params.monitored) %>% 
    purrr::map_depth(.x = ., .depth = 1, .f = ~purrr::set_names(.x, nm = c("post.forest", "lgd", "trueval")))

  
  if(layout.ncol>1) plot.dat.split <- purrr::map(.x = addlzero(n.whales),
                                                 .f = ~{
                                                   temp <- mcmc.object
                                                   temp$mcmc <- dplyr::filter(mcmc.object$mcmc, n == .x)
                                                   temp}) %>%
    purrr::map(.x = ., .f = ~make_dat(.x)) %>%
    purrr::set_names(x = ., nm = addlzero(n.whales)) %>%
    purrr::map_depth(.x = ., .depth = 1, .f = ~purrr::set_names(x = .x, nm = params.monitored)) %>%
    purrr::map_depth(.x = ., .depth = 2, .f = ~purrr::set_names(x = .x, nm = c("post.forest", "lgd", "trueval")))
  
  #'--------------------------------------------------------------------
  # Define functions for plotting
  #'--------------------------------------------------------------------
  
  make_forestplot <- function(dat, param, dark = darken.bars){
    
    #'--------------------------------------------------------------------
    # Set up x-axis
    #'--------------------------------------------------------------------
    
    xbks <- pretty(c(dat$post.forest$int.low, dat$post.forest$int.high), n = 10)
    xbks.labels <- xbks; xbks.labels[seq(2, length(xbks.labels), 2)] <- ""
    
    #'--------------------------------------------------------------------
    # Create bivariate legend
    #'--------------------------------------------------------------------
    
    f.legend <- ggplot(data = dat$lgd, aes(x, y)) +
      geom_tile(aes(fill = ord), col = "white", size = 0.25) +
      scale_fill_manual(values = dat$lgd$tcol) + 
      theme_ridges(font_family = "sans") +
      theme(legend.position = "none",
            panel.background = element_blank(),
            plot.margin = margin(t = 20, b = 10, l = 10)) +
      xlab("Sample size (N)") +
      {if(scenario %in% c(1,2)) ylab("SD of dose (dB)")} +
      {if(scenario %in% c(3,4)) ylab("Proportion of SAT tags (%)")} +
      theme(axis.title = element_text(color = "black"),
            axis.text.x = element_text(size = 12),
            axis.text.y = element_text(size = 12),
            text = element_text(size = 12)) +
      theme(plot.title = element_text(family = "sans", face = "bold", size = 12)) +
      suppressWarnings(scale_x_continuous(breaks = unique(dat$lgd$x), 
                                          labels = unique(removelzero(dat$lgd$n)), expand = c(0, 0))) +
      suppressWarnings(scale_y_continuous(breaks = unique(dat$lgd$y), 
                                          labels = unique(removelzero(unlist(dat$lgd[,2]))), expand = c(0, 0)))
  
    #'-------------------------------------------------
    # Define elements for plotting
    #'------------------------------------------------- 
    
    dat$post.forest <- dat$post.forest %>% 
      dplyr::mutate(cov.x = max(xbks))
    
    #'-------------------------------------------------
    # Define x axis breaks (with some margins either side)
    #'------------------------------------------------- 
    
    f.plot <- dat$post.forest %>% 
      dplyr::group_by_at(., vars(c("n", col_names[1:2]))) %>% 
      
      # Need to summarise first, otherwise the output PDF contains thousands of identical
      # layers on top of each other
      
      dplyr::summarise(dot.mean = mean(dot.mean),
                       int.low = mean(int.low),
                       int.high = mean(int.high),
                       comb = unique(comb)) %>% 
      ggplot(data = ., aes(x = comb, y = dot.mean, ymin = int.low, ymax = int.high)) +
      
      geom_hline(yintercept = dat$trueval, linetype = "dashed", alpha = 1) +
      
      {if(dark) geom_linerange(aes(col = comb), size = 1.5)} +
      {if(dark) geom_linerange(col = "black", size = 0.05)} +
      
      {if(!dark) geom_linerange(aes(col = comb), size = 1)} +
      
      geom_point(aes(x = comb, y = dot.mean, fill = comb), shape = 21, size = 4, stroke = 0.75) +
      
      coord_flip() +
      
      theme_ridges() +
      
      scale_fill_manual(name = ifelse(scenario %in% c(1,2), 
                                      "Uncertainty in dose (dB)", 
                                      "Proportion of SAT tags (%)"), 
                        values = as.character(unique(dat$post.forest$ord))) +
      
      scale_colour_manual(name = ifelse(scenario %in% c(1,2), 
                                        "Uncertainty in dose (dB)", 
                                        "Proportion of SAT tags (%)"), 
                          values = as.character(unique(dat$post.forest$ord))) +
      
      suppressWarnings(scale_x_discrete(labels = paste0("n = ", removelzero(dat$post.forest$n)))) +
      
      {if(param=="mu") ylab(mu ~ "(dB)")} +
      {if(param=="phi") ylab(phi ~ "(dB)")} +
      {if(param=="sigma") ylab(sigma ~ "(dB)")} +
      {if(param=="omega") ylab(omega ~ "(dB)")} +
      {if(param=="beta") ylab(beta ~ "(dB)")} +
      {if(param=="alpha") ylab(alpha ~ "(dB)")} +
      {if(param=="ERR") ylab("ERR (km)")} +
      
      xlab("") +
      
      scale_y_continuous(breaks = xbks, labels = xbks.labels, limits = range(xbks)) +
      ggnewscale::new_scale_fill() + # To allow multiple colour scales
      
      theme(axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            panel.grid.major = element_line(size = 0.2),
            panel.grid.minor = element_line(size = 0.2),
            legend.position = "none",
            plot.margin = margin(t = 10, b = 10, l = 10, r = 10))
    
    return(list(forest = f.plot, lgd = f.legend))
    
  } # End make_forestplot
  
  make_heatplot <- function(dat, param){
    
    #'--------------------------------------------------------------------
    # Parameters to produce a plot for
    #'--------------------------------------------------------------------
    
    heat.params <- c("cred.width", "prb")
    
    if(!param == "ERR"){
      if(!is.null(mcmc.object$ppo)) heat.params <- c(heat.params, "ppo")}

    heat.mat <- purrr::map(.x = heat.params, 
                           .f = ~{
                             
                             #'-----------------------------------------------
                             # Generate labels and data for colour scales
                             #'-----------------------------------------------
                             
                             heat.values <- dat$lgd %>% 
                               dplyr::pull(.x)
                             
                             heat.breaks <- pretty(x = heat.values, min.n = n.cols, n = n.cols +2) 
                             heat.labels <- cut(heat.values, heat.breaks)
                             
                             dat$lgd <- dat$lgd %>% dplyr::mutate(fct = heat.labels)
                             
                             if(length(unique(dat$lgd$fct))>=3){
                               if(.x == "cred.width") myColors <- rev(pals::brewer.spectral(n = length(unique(dat$lgd$fct))))
                               
                               if(.x == "ppo") myColors <- pals::brewer.blues(n = length(unique(dat$lgd$fct)))
                             
                             }else{
                               
                               if(.x == "cred.width") myColors <- rev(pals::cividis(n = length(unique(dat$lgd$fct))))
                               
                               if(.x == "ppo") myColors <- pals::coolwarm(n = length(unique(dat$lgd$fct)))
                             }
                             
                             if(.x == "prb") myColors <- pals::viridis(n = length(unique(dat$lgd$fct)))
                             
                             if(.x == "cred.width"){
                               if(!param == "ERR") add.x <- "dB" else add.x <- "km"
                               lgd.title <- paste0("Credible interval width (", add.x, ")")
                             }else if(.x == "prb"){
                               lgd.title <- "Percentage mean bias (%)"
                             }else if(.x == "ppo"){
                               lgd.title <- "Prior posterior overlap (%)"
                             }
                             
                             #'-----------------------------------------------
                             # Create plot
                             #'-----------------------------------------------
                             
                             heat.plot <- ggplot(data = dat$lgd, aes(x = x, y = y)) +
                               geom_tile(aes(fill = fct), col = "white", size = 0.25) +
                               scale_fill_manual(values = myColors) +
                               {if(scenario %in% c(1,2)) ylab("SD of dose (dB)")} +
                               {if(scenario %in% c(3,4)) ylab("Proportion of SAT tags (%)")} +
                               xlab("Sample size (N)") +
                               {if(param=="mu") ggtitle(mu ~ "")} +
                               {if(param=="omega") ggtitle(omega ~ "")} +
                               {if(param=="phi") ggtitle(phi ~ "")} +
                               {if(param=="sigma") ggtitle(sigma ~ "")} +
                               {if(param=="beta") ggtitle(beta ~ "")} +
                               {if(param=="ERR") ggtitle("ERR")} +
                               labs(fill = lgd.title) +
                               theme_ridges(font_family = "sans") +
                               theme(panel.background = element_blank(),
                                     panel.grid.major = element_blank(), 
                                     panel.grid.minor = element_blank(),
                                     panel.border = element_rect(colour = "black", fill = NA, size = 15),
                                     plot.margin = margin(t = 20, b = 10, l = 10)) +
                               theme(axis.title = element_text(color = "black"),
                                     axis.text.x = element_text(size = 12),
                                     axis.text.y = element_text(size = 12),
                                     text = element_text(size = 12)) +
                               suppressWarnings(scale_x_continuous(breaks = unique(dat$lgd$x), 
                                                                   labels = unique(removelzero(dat$lgd$n)), expand = c(0,0))) +
                               suppressWarnings(scale_y_continuous(breaks = unique(dat$lgd$y), 
                                                                   labels = unique(removelzero(unlist(dat$lgd[,col_names[2]]))), 
                                                                   expand = c(0,0))) +
                               theme(legend.position = "bottom", legend.text = element_text(size = 12), 
                                     legend.title = element_text(face = "bold")) +
                               guides(fill = guide_legend(nrow = round(n.cols/2), 
                                                          byrow = FALSE, 
                                                          title.position = "top", 
                                                          title.vjust = 1.5))
                             
                           }) %>% 
      purrr::set_names(x = ., nm = heat.params)
    
    return(heat.mat)
    
  } # End make_heatplot
  
  #'--------------------------------------------------------------------
  # Generate plots and save to disk
  #'--------------------------------------------------------------------
  
  cat(paste0("Creating plots ...\n"))
  
  # Forest plots +++
  
  pb <- dplyr::progress_estimated(n = length(params.monitored))
  
  forest.plots <- purrr::map2(.x = plot.dat, .y = names(plot.dat),
                              .f = ~{
                                pb$tick()$print()
                                make_forestplot(dat = .x, param = .y)})
  
  if(layout.ncol>1) forest.plots.split <- purrr::map_depth(.x = plot.dat.split, 
                                                           .depth = 2,
                                                           .y = names(plot.dat),
                                                           .f = ~make_forestplot(dat = .x, param = .y))
  
  # Combine forest plots
  
  if(layout.ncol == 1){
    
    combined.forest.plots <- purrr::map(.x = forest.plots, 
                                        .f = ~cowplot::plot_grid(.x$forest, 
                                                                 cowplot::plot_grid(.x$lgd, NULL,
                                                                                    NULL, NULL,
                                                                                    NULL, NULL, 
                                                                                    nrow = 3, ncol = 2, 
                                                                                    rel_widths = c(1, 0.1),
                                                                                    rel_heights = c(0.4, 0.5, 0.3)), 
                                                                 ncol = 2, 
                                                                 rel_widths = c(1,0.75)))
    
  }else{
    
    f.plots <- purrr::map(.x = params.monitored,
                          .f = ~{purrr::map_depth(.x = forest.plots.split, .depth = 1, .f = .x) %>% 
                              purrr::map_depth(.x = ., .depth = 1, .f = "forest")}) %>% 
      purrr::set_names(x = ., nm = params.monitored)
    
    # Retrieve legend
    
    f.plots <- purrr::map(.x = f.plots, .f = ~
                            append(.x, list(lgd = forest.plots.split[[1]][[1]]$lgd)))
    
    combined.forest.plots <- purrr::map(.x = params.monitored,
                                        .f = ~cowplot::plot_grid(plotlist = f.plots[[.x]], 
                                                                 ncol = layout.ncol,
                                                                 nrow = ceiling(length(f.plots[[.x]])/layout.ncol))) %>% 
      purrr::set_names(x = ., nm = params.monitored)
    
  }
  
  # Heat plots +++
  
  pb <- dplyr::progress_estimated(n = length(params.monitored))
  
  heat.plots <- purrr::map2(.x = plot.dat, .y = names(plot.dat),
                            .f = ~{pb$tick()$print()
                              make_heatplot(dat = .x, param = .y)})
  
  # Add blanks
  
  if(!is.null(mcmc.object$ppo)) heat.plots <- purrr::map_depth(.x = heat.plots, .depth = 1, .f = ~append(.x, list(blank = NULL))) else heat.plots <- purrr::map_depth(.x = heat.plots, .depth = 1, .f = ~append(.x, list(blank = NULL, blank = NULL)))
  
  if(scenario%in%c(1,3)){ # Break plots into two pages within same pdf when No params > 3
    
    combined.heat.plots <- cowplot::plot_grid(plotlist = purrr::flatten(heat.plots),
                                            nrow = length(params.monitored), ncol = 4, 
                                            rel_widths = c(0.33, 0.33, 0.33, 0.03),
                                            rel_heights = 1/length(params.monitored))
  }else{
    
    combined.heat.plots <- list(page.one = cowplot::plot_grid(plotlist = purrr::flatten(heat.plots[1:3]),
                                                         nrow = 3, ncol = 4, 
                                                         rel_widths = c(0.33, 0.33, 0.33, 0.03),
                                                         rel_heights = 1/3),
                                page.two = cowplot::plot_grid(plotlist = purrr::flatten(heat.plots[4:6]),
                                                               nrow = 3, ncol = 4, 
                                                               rel_widths = c(0.33, 0.33, 0.33, 0.03),
                                                               rel_heights = 1/3))
     }
  
  cat(paste0("\n"))
  
  # Save to disk 
  
  if(save.to.disk){
    
    cat(paste0("Saving to disk ...\n"))
    
    # Forest plots
    
    if(length(n.whales) > 4) {
      plot.height <- 11
      plot.width <- 10
      }else{
        plot.height <- 6
        plot.width <- 7
      } 
    
    purrr::walk(.x = params.monitored,
                .f = ~ggplot2::ggsave(plot = combined.forest.plots[[.x]], 
                                      filename = paste0(getwd(), "/out/scenario_", scenario, "/S", scenario, 
                                                        "_forestplot_", .x, "_", index, 
                                                        ".", output.format), 
                                      device = output.format, 
                                      height = plot.height, width = plot.width))
    
    # Heat maps (combined)
    
    if(scenario%in%c(1,3)){
      
      ggplot2::ggsave(plot = combined.heat.plots,
                      filename = paste0(getwd(), "/out/scenario_", scenario, "/S", scenario, 
                                        "_heatplots_combined", index, 
                                        ".", output.format), 
                      device = output.format, 
                      height = 15, width = 10)
      
    }else{
      
      pdf(paste0(getwd(), "/out/scenario_", scenario, "/S", scenario, 
                 "_heatplots_combined", index, 
                 ".", output.format), 
          height = 15, width = 10)
      invisible(lapply(combined.heat.plots, print))
      dev.off()
    }
    
    if(save.individual.plots){
    
    # Heat maps (individual)
    
    purrr::walk(.x = params.monitored,
                .f = ~{
                  tmp <- heat.plots[[.x]]
                  pp <- .x
                  purrr::map2(.x = tmp[which(!names(tmp)=="blank")],
                              .y = names(tmp)[which(!names(tmp)=="blank")],
                              .f = ~ cowplot::plot_grid(.x, NULL,
                                                        ncol = 2, 
                                                        rel_widths = c(1, 0.05)) %>% 
                                ggplot2::ggsave(plot = .,
                                                filename = paste0(getwd(), "/out/scenario_", scenario, "/S", scenario, 
                                                                  "_heatplot_", pp, "_", .y, "_", index, 
                                                                  ".", output.format), 
                                                device = output.format, 
                                                height = 5, width = 4))})
    }
    
  }else{
    
    purrr::walk(.x = params.monitored, .f = ~print(combined.forest.plots[[.x]]))
    print(combined.heat.plots)
    
  }
  
  cat("Done!")
  
} # End plot_forest

#'--------------------------------------------------------------------
# Function to plot dose-response functions
#'--------------------------------------------------------------------

plot_doseresponse <- function(mcmc.object, 
                              select.n = NULL,
                              select.obs = NULL,
                              concatenate = FALSE,
                              n.row = 1, 
                              n.col = 1,
                              save.to.disk = FALSE, 
                              output.format = "pdf",
                              plot.width = 850,
                              plot.height = 850,
                              plot.res = 300){
  
  #'--------------------------------------------------------------------
  # PARAMETERS
  #'--------------------------------------------------------------------
  #' @param mcmc.object List. Output from the \code{run_scenario()} function.
  #' @param select.n Subset of sample sizes to display. All values are shown when set to NULL (the default). 
  #' @param select.obs Subset of values to display for the observation model parameter (i.e. uncertainty.dose in scenarios 1 and 2, prop.sat in scenarios 3 and 4). All values are shown when set to NULL (the default). 
  #' @param concatenate Logical. By default, plots for each sample size are saved on separate pages when output to PDF, which can result in unnecessary empty space depending on the type of layout chosen (n.row x n.col). Set this argument to TRUE to combine plots across a minimum number of pages.
  #' @param n.row Number of rows used in the plot layout. Values greater than 1 (default) allow multiple plots to be arranged in the same plotting space.
  #' @param n.col Number of columns used in the plot layout. Values greater than 1 (default) allow multiple plots to be arranged in the same plotting space.
  #' @param save.to.disk Logical. Whether to save the plots to disk. Defaults to FALSE.
  #' @param output.format Output file type. Defaults to "pdf".
  #' @param plot.width Width of the output plot (in pixels).
  #' @param plot.height Height of the output plot (in pixels).
  #' @param plot.res Resolution of the output plot (in dpi).
  #'--------------------------------------------------------------------
  
  #'-------------------------------------------------
  # Function checks
  #'-------------------------------------------------
  
  if (!dir.exists(file.path(paste0(getwd(),"/out/scenario_", mcmc.object$params$scenario))))
    dir.create(path = file.path(paste0(getwd(),"/out/scenario_", mcmc.object$params$scenario)), showWarnings = FALSE)
  
  #'--------------------------------------------------------------------
  # Retrieve simulation parameters
  #'--------------------------------------------------------------------
  
  scenario <- mcmc.object$params$scenario
  index <- mcmc.object$params$index
  n.whales <- mcmc.object$params$n.whales
  col_names <- mcmc.object$col_names
  params.monitored <- mcmc.object$params$params.monitored
  if(scenario %in% c(1,2)) obs.param <- mcmc.object$params$uncertainty.dose
  if(scenario %in% c(3,4)) obs.param <- mcmc.object$params$prop.sat
  
  #'-------------------------------------------------
  # n x RL/ratio combinations
  #'-------------------------------------------------
  
  if(!is.null(select.n)){
    if(!is.null(select.obs)){
      use.n <- select.n; use.obs <- select.obs
    }else{
      use.n <- select.n; use.obs <- obs.param
    } # End if select.dose
  }else{
    if(!is.null(select.obs)){
      use.n <- n.whales; use.obs <- select.obs
    }else{
      use.n <- n.whales; use.obs <- obs.param
    } # End if select.dose
  } # End select.n
  
  combs <- expand.grid(use.n, use.obs) %>% 
    dplyr::rename(n = Var1, !!col_names[2] := Var2) %>% 
    dplyr::mutate(n = addlzero(n), !!col_names[2] := addlzero(!!as.name(col_names[2]))) %>% 
    dplyr::arrange_at(., vars(col_names[1:2]))
  
  #'-------------------------------------------------
  # Set up plot
  #'-------------------------------------------------
  
  if(save.to.disk & output.format=="pdf") pdf(paste0(getwd(), "/out/scenario_", scenario, "/S", 
                                                     scenario, "_doseresponse_plots_", index, ".pdf"))
  
  par(mfrow = c(n.row, n.col))

  combs <- combs %>% 
    dplyr::mutate(blank = 0) 
  
  combs.list <- dplyr::group_split(combs, n)
  nc <- purrr::map_dbl(.x = combs.list, .f = ~nrow(.x)) %>% unique()
  
  npages <- ceiling(nc/(n.row*n.col))
  ncells <- npages * n.row*n.col
  
  if(!concatenate){  
    combs <- dplyr::group_split(combs, n) %>% 
      purrr::map(.x = .,
                 .f = ~{
                   res <- data.frame(matrix(rep(c(NA, NA, 1), ncells-nc), byrow = TRUE, ncol = 3))
                   names(res) <- names(.x)
                   rbind(.x, res)
                 })
    
    if(class(combs)=="list") combs <- do.call(rbind, combs)}
  
  #'-------------------------------------------------
  # Create plots
  #'-------------------------------------------------
  
  purrr::map(.x = seq_len(nrow(combs)),
             .f = ~{
               
               # JPEG, TIFF and PNG are not multi-page file formats, so need to output one plot
               # per combination of sd(dose) x sample size
               
               if(save.to.disk) {
                 
                 if(output.format=="tiff") tiff(paste0(getwd(), "/out/scenario_", scenario, 
                                                       "/S", scenario, "_doseresponse_plots_", 
                                                       index, "_", .x, ".tiff"), 
                                                width = plot.width, height = plot.height, res = plot.res)
                 
                 if(output.format=="png") png(paste0(getwd(), "/out/scenario_", scenario,
                                                     "/S", scenario, "_doseresponse_plots_", 
                                                     index, "_", .x, ".png"), 
                                              res = plot.res, width = plot.width, height = plot.height)
                 
               }
               
               #'-------------------------------------------------
               # Start with an empty plot
               #'-------------------------------------------------
               
               if(combs$blank[.x]==1){
                 
                 plot.new()
                 
               }else{
                 
                 title.part.1 <- paste0("N = ", removelzero(gsub(pattern = "n_", replacement = "", x = combs$n[.x])), " |")
                 
                 if(scenario %in% c(1,2)){
                   
                   title.part.2 <- paste0(removelzero(gsub(pattern = "RLsd_", replacement = "", x = combs[.x, 2])), " dB")
                   plot.title <- bquote(bold(.(title.part.1)~delta == .(title.part.2)))}

                 if(scenario %in% c(3,4)) plot.title <- paste0("N = ", 
                                          removelzero(gsub(pattern = "n_", replacement = "", x = combs$n[.x])), 
                                          " | P(SAT) = ",
                                          as.numeric(removelzero(gsub(pattern = "sat_", replacement = "", x = combs[.x, 2])))," %")
                 
                 plot(x = mcmc.object$dose.response$dose.range,
                      y = seq(0,1, length = length(mcmc.object$dose.response$dose.range)), 
                      type = "n",
                      xlab = expression(paste("Dose (dB re 1", mu, "Pa)")), 
                      ylab = "p(response)",
                      ylim = c(0,1),
                      xlim = c(mcmc.object$params$lower.bound,
                               mcmc.object$params$upper.bound),
                      cex.lab = 1.2, 
                      cex.main = 1, 
                      cex.axis = 1.1,
                      main = plot.title)
                 
                 
                 #'-------------------------------------------------
                 # Plot polygons for each quantile, 
                 # in increasingly darker shades
                 #'-------------------------------------------------
                 
                 polygon.label <- paste0("n_", combs$n[.x], ifelse(scenario %in% c(1,2), "-RLsd_", "-sat_"), combs[.x, 2])
                 
                 purrr::walk(.x = 1:length(mcmc.object$dose.response$q),
                             .f = ~{
                               polygon(x = c(mcmc.object$dose.response$dose.range, 
                                             rev(mcmc.object$dose.response$dose.range)),
                                       y = c(mcmc.object$dose.response$p[[polygon.label]]$doseresp.lower[.x,], 
                                             rev(mcmc.object$dose.response$p[[polygon.label]]$doseresp.upper[.x,])), 
                                       col = hexa2hex(input.colour = "#066e9e",
                                                      opacity = 1-(mcmc.object$dose.response$q[.x]/100)), 
                                       border = NA)})
                 
                 #'-------------------------------------------------
                 # Add posterior mean
                 #'-------------------------------------------------
                 
                 nn <- names(mcmc.object$dose.response$p) %>% unique()
                 
                 lines(mcmc.object$dose.response$dose.range, 
                       mcmc.object$dose.response$p[[which(nn==polygon.label)]]$doseresp.mean,
                       type = "l",
                       lwd = 2, 
                       col = "#eaaf00")
               }
               
               if(save.to.disk & !output.format=="pdf") dev.off()
               
             })
  
  if(save.to.disk & output.format=="pdf") dev.off()
  
} # End plot_doseresponse

#'--------------------------------------------------------------------
# Function to calculate the effective response range (ERR)
#'--------------------------------------------------------------------

# Code taken from Tyack & Thomas (2019)
# The ERR is the radius at which as many animals respond beyond it as fail to respond within it.
# By definition, the total number of animals within this range (both responding and not responding)
# is exactly equal to the total number of animals responding.
# Transmission loss model assumed to be inverse‐square spherical spreading.

effective_range <- function(response.threshold = 141,
                            response.sd = 33.5, 
                            response.lowerbound = 60,
                            response.upperbound = 200,
                            received.level,
                            D = 1, 
                            n.bins = 10000,
                            maximum.rge
                            
){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param response.threshold Mean response threshold (received level at which a response occurs).
  #' @param response.sd Uncertainty around the response threshold.
  #' @param response.lowerbound Received level below which no animals respond.
  #' @param response.upperbound Received level at which all animals exhibit a response.
  #' @param received.level Received levels, based on a given source level and transmission loss model.
  #' @param D Animal density. Defaults to 1 animal per km2.
  #' @param n.bins Number of bins with which to divide the \code{maximum.rge}. The number of whales showing a response is evaluated in each bin. 
  #' @param maximum.rge Maximum range for sound propagation (km).
  #'---------------------------------------------
  
  cutpoints <- seq(0, maximum.rge, length = (n.bins+1))
  
  #'---------------------------------------------
  # Probabilities of response
  #'---------------------------------------------
  
  # CDF from a truncated Normal
  
  prob.response <- truncnorm::ptruncnorm(q = received.level, 
                                         a = response.lowerbound, 
                                         b = response.upperbound, 
                                         mean = response.threshold, 
                                         sd = response.sd)
  
  #'---------------------------------------------
  # Effective response range
  #'---------------------------------------------
  
  n.respond <- sum(D*pi*(cutpoints[-1]^2-cutpoints[1:n.bins]^2)*prob.response)
  ERR <- sqrt(D*n.respond/pi)
  
  return(ERR)
  
} # End effective_range

#'--------------------------------------------------------------------
# Function to estimate uncertainty in dose for sat tagged animals
#'--------------------------------------------------------------------

xy_error <- function(argos.data,
                     received.lvl, 
                     source.lvl, 
                     multi = FALSE,
                     n.ellipses = 100,
                     plot.ellipse = FALSE){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param argos.data Input ARGOS data.
  #' @param received.lvl Received level.
  #' @param source.lvl Sound pressure level of the noise source.
  #' @param multi Logical. If TRUE, repeat the estimation for multiple ellipses.
  #' @param n.ellipses Number of ellipses to simulate when multi = TRUE.
  #' @param plot.ellipse Logical. If TRUE, create plot a random realisation of the bivariate normal from which simulated errors are drawn.
  #'---------------------------------------------

  if(received.lvl>source.lvl) stop("Received level cannot exceed source level")
  
  mu.vector <- rep(received.lvl, ifelse(multi, n.ellipses, 1))
  
  # (1) Convert to estimate of range (km)
  
  r.i <- purrr::map_dbl(.x = mu.vector, 
                        .f = ~stats::optimize(f = range_finder, 
                                              interval = c(0,30000), 
                                              SL = source.lvl, 
                                              target.L = .x)$minimum)
  
  # (2) Convert to xy coordinates relative to source point
  
  a.i <- runif(n = length(r.i), min = 0, max = 360)
  rx.i <- cos(a.i*pi/180)*r.i
  ry.i <- sin(a.i*pi/180)*r.i
  
  # (3) Add some positional xy error according to bivariate Normal
  
  # The satellites carrying Argos equipment are polar orbiting, and as a result the true 
  # error around calculated positions is better represented by 2-dimensional ellipses rather 
  # than 1-dimensional circles. From the covariance matrix of the messages received by the satellite, 
  # Argos derives an error ellipse with semi-major and minor axes M and m, and an ellipse orientation.
  # Refer to McClintock et al. (2015) for arguments regarding the importance of using the
  # error ellipse as opposed to the error circle.
  # Here, relevant values are taken from a real-world dataset on tagged whales.

  # Generate random ARGOS ellipse based on real dataset
  
  xy.pts <- purrr::map(.x = seq_along(r.i), 
                       .f = ~{
                         
                         ellipse.sample <- argos.data[sample(x = 1:nrow(argos.data), size = 1),]
                         ellipse.M <- ellipse.sample$error_semi_major_axis/1000 # Semi-major axis
                         ellipse.m <- ellipse.sample$error_semi_minor_axis/1000 # Semi-minor axis
                         ellipse.alpha <- ellipse.sample$error_ellipse_orientation # Orientation
                         
                           drawFrom.ellipse <- function(ns = 10000){
    
    # See McClintock et al.
    
    ellipse.d1 <- ((ellipse.M/sqrt(2))^2)*(sin(ellipse.alpha*pi/180)^2)+((ellipse.m/sqrt(2))^2)*(cos(ellipse.alpha*pi/180)^2)
    
    ellipse.d2 <- ((ellipse.M/sqrt(2))^2)*(cos(ellipse.alpha*pi/180)^2)+((ellipse.m/sqrt(2))^2)*(sin(ellipse.alpha*pi/180)^2)
    
    ellipse.cov <- ((ellipse.M^2-ellipse.m^2)/2)*cos(ellipse.alpha*pi/180)*sin(ellipse.alpha*pi/180)
    
    # Parameters for bivariate normal distribution
    
    ellipse.sigma <- matrix(c(ellipse.d1, ellipse.cov, ellipse.cov, ellipse.d2),2) # Covariance matrix
    
    xy.pts <- MASS::mvrnorm(n = ns, mu = c(0,0), 
                            Sigma = ellipse.sigma)
    return(xy.pts)
   }
                         
                         # Draw positional errors
                         
                         drawFrom.ellipse()
                         
                       })
  
  # Shift xy point accordingly
  
  crx.i <- purrr::map(.x = xy.pts, .f = ~rx.i + .x[,1])
  cry.i <- purrr::map(.x = xy.pts, .f = ~ry.i + .x[,2])
  
  # Quick plot to check
  
  if(plot.ellipse){
     
    par(mar = c(0, 2, 2, 2)) 
    layout(matrix(c(1,2,3,4), ncol = 2, byrow = TRUE), heights = c(4, 1))
    
     ind <- sample(x = 1:length(xy.pts), size = 1)
     
     rr.i <- xy.pts[[ind]]
     rr.i[,1] <- rr.i[,1] + rx.i; rr.i[,2] <- rr.i[,2] + ry.i;
     
     bivn.kde <- MASS::kde2d(rr.i[,1], rr.i[,2], n = 50)
     
     min.x <- min(min(rr.i[,1]), -1)
     min.y <- min(min(rr.i[,2]), -1)
     max.x <- max(max(rr.i[,1]), 1)
     max.y <- max(max(rr.i[,2]), 1)
     
     plot.new()
     polygon(c(min.x, min.x,
               max.x, max.x),
             c(min.y, max.y,
               max.y, min.y), 
             col = "#FFFEC5")
     par(new = TRUE)
     image(bivn.kde, xlim = c(min.x, max.x),
           ylim = c(min.y, max.y), xlab = "x", ylab = "y")

     points(rx.i, ry.i, col = "black", bg = "darkorchid1", pch = 21, cex = 2)
     points(0,0, pch = 21, col = "black", bg = "green", cex = 2) # Noise source
     box()
     }
  
  # (4) Derive range
  
  rge.i <- purrr::map2(.x = crx.i, .y = cry.i, .f = ~sqrt(.x^2 + .y^2))
  
  # (5) Convert back to received levels
  # If the revised position is far enough that
  # the TL exceeds the source level, set received level to 0
  
  rl.i <- purrr::map(.x = rge.i, .f = ~{
    new.RL <- source.lvl-TL(rge = .x)
    ifelse(new.RL<0, 0, new.RL)})
  
  if(multi) rl.sd <- purrr::map(.x = rl.i, .f = ~sd(.x)) %>% do.call(c, .)
  
  RLsim <- do.call(c, rl.i)
  
  if(plot.ellipse){
    hist(RLsim, xlab = "Received level (dB)", main = "", col = "lightblue")
    # abline(v = received.lvl, col = "darkorange", lwd = 2)
    # abline(v = mean(RLsim), col = "royalblue3", lwd = 2)
    plot.new()
    legend("bottomleft", box.col = "transparent", 
           legend = c("Source", "Animal"), inset = .02,
           col = c("green", "darkorchid1"), pch = 16, cex = 1.2)
    # plot.new()
    # legend("bottomleft", box.col = "transparent", 
    #        legend = c("Initial threshold", "After XY error "), inset = .02,
    #        col = c("darkorange", "royalblue3"), lty = c(1,1), cex = 1.2)
  }
  
  # Reset graphical parameters
  
  par(mfrow = c(1,1))
  par(mar = c(5.1,4.1,4.1,2.1))
  
  res <- list(random = sample(x = RLsim, size = 1), 
              rl = received.lvl)
  
  if(multi) res <- append(x = res, values = list(sd = rl.sd)) else res <- append(x = res, values = list(mean = mean(RLsim), sd = sd(RLsim)))
  
  return(res)
  
} # End xy_error

#'--------------------------------------------------------------------
# Function to compute SD(dose) as a function of distance from source
#'--------------------------------------------------------------------

run_argos_example <- function(N = 500, 
                              range.min = 10, 
                              range.max = 240, 
                              range.increment = 10,
                              source.lvl = 210){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param N Number of simulated animals (i.e. error ellipses).
  #' @param range.min Minimum distance from the noise source. 
  #' @param range.maximum Maximum distance from the noise source.
  #' @param range.increment Increments by which to divide the range.
  #' @param source.lvl Level of the noise source.
  #'---------------------------------------------
  
  # SD(dose) at incremental distances from source
  
  dist.increments <- c(2, 5, seq(from = range.min, to = range.max, by = range.increment))
  
  # Calculate received levels at those distances
  
  dist.TLs <- TL(rge = dist.increments) # How much is lost
  dist.RLs <- 210-dist.TLs # Received levels given loss
  
  # Apply Argos error ellipses
  
  pb <- dplyr::progress_estimated(n = length(dist.increments))
  
  rl.dist <- purrr::map(.x = dist.RLs,
                        .f = ~{
                          pb$tick()$print()
                          xy_error(argos.data = argos, 
                                   received.lvl = .x, 
                                   source.lvl = source.lvl, 
                                   multi = TRUE, 
                                   n.ellipses = N,
                                   plot.ellipse = F)})
  
  # Extract relevant data and convert to tibble
  
  rl.dist.df <- purrr::map(.x = rl.dist, .f = "sd") %>% 
    purrr::set_names(., nm = as.character(dist.increments)) %>% 
    reshape2::melt(.) %>% 
    tibble::as_tibble(.) %>% 
    dplyr::rename(sd_dose = value, rge = L1) %>% 
    dplyr::mutate(rge_f = factor(rge, levels = unique(sort(as.numeric(rge))))) %>% 
    dplyr::mutate(rge = as.numeric(rge))
  
  # Return overall median and quantiles
  
  median(rl.dist.df$sd_dose)
  quantile(rl.dist.df$sd_dose, c(0.25, 0.5, 0.75))
  
  gg.opts <- theme(panel.grid.minor = element_blank(),
                   panel.background = element_blank(),
                   plot.title = element_text(size = 13, face = "bold"),
                   legend.title = element_text(size = 12),
                   legend.text = element_text(size = 11),
                   axis.text = element_text(size = 11, colour = "black"),
                   panel.grid.major = element_line(colour = 'lightgrey', size = 0.1),
                   legend.position = "bottom",
                   axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0)),
                   axis.title.x = element_text(margin = margin(t = 15, r = 0, b = 0, l = 0)),
                   axis.ticks = element_blank())
  
  rl.dist.df %>%
    ggplot(data = ., aes(x = rge_f, y = sd_dose)) +
    geom_boxplot(aes(fill = rge_f)) +
    scale_fill_manual(values = rev(pals::parula(nlevels(rl.dist.df$rge_f)))) +
    xlab("Range (km)") + ylab("sd(RL) after ARGOS correction") +
    gg.opts + xlab("Range from noise source (km)") +
    ylab("SD in received levels (dB re1\u03BCPa)") +
    theme(legend.position = "none")
  
}

# run_argos_example(N = 2000)

# ggplot2::ggsave(filename = "/Users/philippebouchet/Google Drive/Documents/postdoc/creem/dMOCHA/dose_response/simulation/dose_uncertainty/rmd/fig/fig_argos_sd.pdf", device = cairo_pdf, width = 8, height = 5)

#'--------------------------------------------------------------------
# Function to compile results from different simulation runs
#'--------------------------------------------------------------------

compile_sim <- function(scenario){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param scenario Scenario ID.
  #'---------------------------------------------
  
  #'-------------------------------------------
  # List RData files in the 'out' directory
  #'-------------------------------------------
  
  sdir <- paste0(getwd(), "/out/scenario_", scenario)
  fl <- list.files(path = sdir, pattern = "mcmc.res")
  
  #'-------------------------------------------
  # Import all data into a list
  #'-------------------------------------------
  
  mcmc.sims <- purrr::map(.x = fl, .f = ~readRDS(file.path(sdir, .x)))
  mcmc.sims <- purrr::set_names(mcmc.sims, 
                                gsub(pattern = ".rds", replacement = "", 
                                     gsub(pattern = "mcmcres_", replacement = "", fl)))
  
  col_names <- purrr::map(.x = mcmc.sims, .depth = 1, .f = "col_names") %>% unlist() %>% unique()
  
  #'-------------------------------------------
  # Calculate total run time
  #'-------------------------------------------
  
  totrun <- purrr::map_depth(.x = mcmc.sims, .depth = 1, "run.time") %>% 
    do.call(sum, .) %>% as.numeric() %>% lubridate::seconds_to_period(.)
  
  cat(paste0("Total run time: ", totrun))
  
  #'-------------------------------------------
  # Compile necessary elements
  #'-------------------------------------------
  
  mcmc.compiled <- list()
  
  mcmc.compiled$col_names <- col_names
  
  mcmc.compiled$mcmc <- purrr::map_dfr(.x = mcmc.sims, .f = "mcmc") %>% 
    dplyr::arrange_at(., vars(col_names[1:2], sim))
  
  mcmc.compiled$summary <- purrr::map_dfr(.x = mcmc.sims, .f = "summary") %>% 
    dplyr::arrange_at(., vars(col_names[1:2], param))
  
  mcmc.compiled$dose.response$dose.range <- mcmc.sims[[1]]$dose.response$dose.range
  mcmc.compiled$dose.response$q <- mcmc.sims[[1]]$dose.response$q
  
  mcmc.compiled$dose.response$p <- purrr::map(.x = mcmc.sims, .f = "dose.response") %>% 
    purrr::map(.x = ., .f = "p") %>% purrr::flatten(.)
  
  mcmc.compiled$cor <- purrr::map(.x = mcmc.sims, .f = "cor") %>% 
    purrr::map_dfr(.x = ., .f = ~dplyr::mutate(.x, lag = as.character(lag))) %>% 
    dplyr::arrange_at(., vars(col_names[1:2], sim))
  
  mcmc.compiled$ppo <- purrr::map(.x = mcmc.sims, .f = ~.x$ppo) %>% 
    do.call(rbind, .)
  
  mcmc.compiled$tbl <- purrr::map(.x = mcmc.sims, .f = ~.x$tbl) %>% 
    do.call(rbind, .)
  
  mcmc.compiled$convergence$gelmanR <- try(purrr::map(mcmc.sims, "convergence") %>% 
    purrr::map(., "gelmanR") %>% unlist() %>% unique() %>% sort())
  
  mcmc.compiled$convergence$rhat <- try(purrr::map(mcmc.sims, "convergence") %>% 
    purrr::map_dfr(.x = ., .f = "rhat") %>% 
    dplyr::arrange_at(., vars(col_names[1:2], sim)))
  
  mcmc.compiled$convergence$fail_count <- try(purrr::map(mcmc.sims, "convergence") %>% 
    purrr::map_dfr(.x = ., .f = "fail_count") %>% 
    dplyr::arrange_at(., vars(col_names[1:2])))
  
  mcmc.compiled$convergence$fail <- try(purrr::map(mcmc.sims, "convergence") %>% 
    purrr::map_dfr(.x = ., .f = "fail") %>% 
    dplyr::arrange_at(., vars(col_names[1:2])))
  
  mcmc.compiled$params$index <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "index") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$scenario <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "scenario") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$n.sim <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "n.sim") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$n.whales <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "n.whales") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$uncertainty.dose <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "uncertainty.dose") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$prop.sat <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "prop.sat") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$params.monitored <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "params.monitored") %>% unlist() %>% unique() %>% sort()
 
  mcmc.compiled$params$chain.lengths <- purrr::map_depth(.x = mcmc.sims, .depth = 1, "params") %>%
      purrr::map_depth(.x = ., .depth = 1, "chain.lengths")  %>% 
      do.call(rbind, .)
  
  mcmc.compiled$params$burn.in <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "burn.in") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$mcmc.thin <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "mcmc.thin") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$mcmc.chains <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "mcmc.chains") %>% unlist() %>% unique() %>% sort()

  mcmc.compiled$params$mcmc.auto <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "mcmc.auto") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$mcmc.save <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "mcmc.save") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$mcmc.n <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "mcmc.n") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$true.mu <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "true.mu") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$true.ERR <- 
    purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "true.ERR") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$lower.bound <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "lower.bound") %>% unlist() %>% unique() %>% sort()
  
  mcmc.compiled$params$upper.bound <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "upper.bound") %>% unlist() %>% unique() %>% sort()
  
  if(scenario %in% c(1,3)){
    
    mcmc.compiled$params$true.omega <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "true.omega") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$omega.upper.bound <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "omega.upper.bound") %>% unlist() %>% unique() %>% sort()
    
  }
  
  if(scenario %in% c(2,4)){
    
    mcmc.compiled$params$true.phi <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "true.phi") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$true.sigma <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "true.sigma") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$true.beta <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "true.beta") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$beta.sd <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "beta.sd") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$true.alpha <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "true.alpha") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$alpha.sd <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "alpha.sd") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$phi.upper.bound <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "phi.upper.bound") %>% unlist() %>% unique() %>% sort()
    
    mcmc.compiled$params$sigma.upper.bound <- purrr::map(mcmc.sims, "params") %>% 
      purrr::map(., "sigma.upper.bound") %>% unlist() %>% unique() %>% sort()
    
  }
  
  mcmc.compiled$params$parallel.cores <- purrr::map(mcmc.sims, "params") %>% 
    purrr::map(., "parallel.cores") %>% unlist() %>% unique() %>% sort()
  
  return(mcmc.compiled)
  
} # End compile_sim

#'--------------------------------------------------------------------
# Function to load results from one simulation
#'--------------------------------------------------------------------

load_sim <- function(scenario, index){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param scenario Scenario ID.
  #' @param index Simulation run ID.
  #'---------------------------------------------
  
  # List RData files in the 'out' directory
  
  sdir <- paste0(getwd(), "/out/scenario_", scenario)
  res <- readRDS(file = paste0(sdir, "/mcmcres_", addlzero(index), ".rds"))
  return(res)}

#'--------------------------------------------------------------------
# Function to extract the nth element from a vector, based on start position
#'--------------------------------------------------------------------

nth_element <- function(vector, starting.position, n) { 
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param vector Input vector.
  #' @param starting.position Starting position.
  #' @param n Integer. Sampling interval.
  #'---------------------------------------------
  
  vector[seq(starting.position, length(vector), n)] 
}

#'--------------------------------------------------------------------
# Sound transmission loss function
#'--------------------------------------------------------------------

# Note: This is how much is lost, * NOT * the sound level after loss.

TL <- function(rge, a = 0.185){ 
  
  #'--------------------------------------------------------------------
  # PARAMETERS
  #'--------------------------------------------------------------------
  #' @param rge Range in km.
  #' @param a Sound absorption coefficient, in dB per km. This is frequency-dependent, and takes a value of 0.185 for a 3 kHz signal under normal sea conditions.
  #'--------------------------------------------------------------------
  
  loss <- 20*log10(rge*1000)
  loss[loss<0] <- 0
  loss <- loss+a*rge
  return(loss)}

#'--------------------------------------------------------------------
# Function to calculate the range corresponding to a given RL
#'--------------------------------------------------------------------
# Return squared difference between target.TL and actual TL

range_finder <- function(rge, SL, target.L){
  
  #'--------------------------------------------------------------------
  # PARAMETERS
  #'--------------------------------------------------------------------
  #' @param rge Range in km.
  #' @param SL Level of the noise source.
  #' @param target.L Target noise level.
  #'--------------------------------------------------------------------
  
  return((SL-TL(rge)-target.L)^2)}

#'--------------------------------------------------------------------
# Function to find the HEX colour code corresponding to an input colour 
# with a set opacity level (i.e. emulate transparency)
#'--------------------------------------------------------------------

hexa2hex <- function(input.colour, 
                     opacity, 
                     bg.colour = "white"){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param input.colour Initial colour.
  #' @param opacity Desired level of transparency (number between 0 and 1).
  #' @param bg.colour Colour of the background. Defaults to 'white'.
  #'---------------------------------------------
  
  # White background
  
  bg <- grDevices::col2rgb(bg.colour, alpha = FALSE)
  
  # Convert input colour to RGB
  
  rgbcol <- grDevices::col2rgb(input.colour, alpha = FALSE)
  
  # Calculate red, green, blue values corresponding to input colour at chosen transparency level
  
  rc <- (1 - opacity) * bg[1,] + opacity * rgbcol[1,]
  gc <- (1 - opacity) * bg[2,] + opacity * rgbcol[2,]
  bc <- (1 - opacity) * bg[3,] + opacity * rgbcol[3,]
  
  # Convert back to hex
  
  rgb2hex <- function(r,g,b) rgb(r, g, b, maxColorValue = 255)
  return(rgb2hex(r = rc, g = gc, b = bc))
}

#'--------------------------------------------------------------------
# Stop and start parallel processing
#'--------------------------------------------------------------------

start_cluster <- function(n.cores){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param n.cores Number of cores to use for parallel processing.
  #'---------------------------------------------

  cl <<- parallel::makeCluster(n.cores)
  doParallel::registerDoParallel(cl) }
  
stop_cluster <- function(worker = cl){
  parallel::stopCluster(cl = worker) # Stop the cluster
  rm(cl, envir = .GlobalEnv)
  gc()
}

#'--------------------------------------------------------------------
# Quick convenience function to add/remove leading zeroes
#'--------------------------------------------------------------------

# Using leading zeroes is necessary to get correct order in plots etc. 

addlzero <- function(string){
  sapply(X = string, FUN = function(x){
  if(x%%1==0) res <- stringr::str_pad(string = x, width = 3, pad = "0")
  if(x%%1>0) {
    a <- do.call(rbind, strsplit(as.character(x),"\\."))
    b <- stringr::str_pad(string = a[,1], width = 3, pad = "0")
    res <- paste0(c(b, a[,2]), collapse = "-")}
  return(res)})
}

removelzero <- function(string){
  sapply(X = string, FUN = function(x){
  is.dash <- stringr::str_locate_all(string = x, pattern = "-") %>% unlist() %>% unique()
  if(length(is.dash)==0){
    res <- stringr::str_remove(string = x, pattern = "^0+")
  if(res=="") res <- "0"
  }else{
    a <- stringr::str_split(string = x, pattern = "-") %>% unlist()
    b <- stringr::str_remove(string = a[1], pattern = "^0+")
    res <- as.character((paste0(c(b, a[2]), collapse = ".")))
  }
  names(res) <- ""
  return(res)})
}

#'--------------------------------------------------------------------
# Convenience function to remove labels
#'--------------------------------------------------------------------

removelabels <- function(scenario.id, tbl){
  
  if(scenario.id %in% c(1,2))  tbl <- tbl %>% 
    dplyr::mutate(n = gsub(pattern = "n_", replacement = "", x = n),
                  RL = gsub(pattern = "RLsd_", replacement = "", x = RL),
                  sim = gsub(pattern = "sim_", replacement = "", x = sim))
  
  if(scenario.id %in% c(3,4))  tbl <- tbl %>% 
      dplyr::mutate(n = gsub(pattern = "n_", replacement = "", x = n),
                    ratio = gsub(pattern = "sat_", replacement = "", x = ratio),
                    sim = gsub(pattern = "sim_", replacement = "", x = sim))
  return(tbl)}

#'--------------------------------------------------------------------
# Modified version of bayesplot::ppc_dens_overlay that does not throw errors with NAS
#'--------------------------------------------------------------------

ppc_densoverlay <- function(y, y.rep, n.yrep, lbound, ubound){
  
  #'--------------------------------------------------------------------
  # PARAMETERS
  #'--------------------------------------------------------------------
  #' @param y 
  #' @param y.rep 
  #' @param n.yrep 
  #' @param lbound 
  #' @param ubound 
  #'--------------------------------------------------------------------
  
  #'-------------------------------------------------
  # Compile y and yrep into individual data.frames
  #'-------------------------------------------------
  
  ydat <- data.frame(y = y, type = "y")
  yrep.dat <- data.frame(y = as.vector(y.rep), type = "yrep")
  
  #'-------------------------------------------------
  # Combine y and yrep and calculate maximum value of kernel density
  #'-------------------------------------------------
  
  alldat <- rbind(ydat, yrep.dat)
  ydens <- NULL
  
  for(i in 1:n.yrep){
    dd <- density(na.omit(y.rep[i,]), kernel = "gaussian", adjust = 1)
    ydens <- c(ydens, max(dd$y))
  } 
  
  ydens.max <- max(ydens)
  
  #'-------------------------------------------------
  # Build density plot
  #'-------------------------------------------------
  
  # First the base frame
  
  p <- ggplot(data = alldat, aes(x = y)) + 
    scale_x_continuous(limits = extendrange(x = c(lbound, ubound), f = 0.25)) + 
    scale_y_continuous(limits = extendrange(x = c(0, ydens.max), f = c(0, 0.35)))
  
  # Then each replicate dataset yrep
  
  for (i in 1:n.yrep) { 
    loop_input = paste0("stat_density(data = data.frame(y = y.rep[",i,", ]), geom = 'line', position = 'identity', colour = '#98CAFF', size = 0.5, alpha = 0.75, adjust = 1, trim = FALSE, kernel = 'gaussian', n = 1024, na.rm = TRUE)") 
    p <- p + eval(parse(text = loop_input))  
  }
  
  # Finally the data y
  
  p + stat_density(data = ydat, geom = "line", position = "identity", colour = "#004C99",
                   size = 2, alpha = 1, adjust = 1, trim = FALSE, kernel = "gaussian", n = 1024, na.rm = TRUE) +
    ylab("") + xlab("") +
    theme_cowplot() +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text = element_text(size = 14)) +
    coord_cartesian(expand = FALSE)
  
}  # End ppc_densoverlay

#'--------------------------------------------------------------------
# Function to assign appropriate names to list elements
#'--------------------------------------------------------------------

name.list <- function(scenario.id,
                      input.list, 
                      Nwhales, 
                      Nsim,
                      dose.or.ratio){
  
  #'---------------------------------------------
  # PARAMETERS
  #'---------------------------------------------
  #' @param scenario.id Scenario ID.
  #' @param input.list Input list.
  #' @param Nwhales Number of whales (\code{n.whales} in \code{run_scenario()}).
  #' @param Nsim Number of simulations (\code{n.sim} in \code{run_scenario()}).
  #' @param dose.or.ratio Values of the observation model parameter (\code{uncertainty.dose} or \code{prop.sat}).
  #'---------------------------------------------
  
  # Top level is No. whales
  
  out.list <- purrr::set_names(input.list, paste0("n_", addlzero(Nwhales)))
  
  # Next level is uncertainty in dose / tag ratio
  
  if(scenario.id %in% c(1,2)) out.list <- purrr::map(.x = out.list, 
                         .f = ~purrr::set_names(., paste0("RLsd_", addlzero(dose.or.ratio))))
  
  if(scenario.id %in% c(3,4)) out.list <- purrr::map(.x = out.list, 
                                                  .f = ~purrr::set_names(., paste0("sat_", addlzero(dose.or.ratio))))
  
  # Last level is simulation run IDs
  
  if(!is.null(Nsim))
    out.list <- purrr::map(.x = out.list, 
                           .f = ~purrr::map(.x = .x, 
                                            .f = ~purrr::set_names(., paste0('sim_', 
                                                                             addlzero(1:Nsim)))))
  return(out.list)} # End name.list()

quiet <- function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 
