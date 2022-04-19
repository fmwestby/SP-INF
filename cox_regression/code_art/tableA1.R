#####################################################################
# Replication file for: "Simulating Duration Data for               #
# the Cox Model"                                                    #
#                                                                   #
# Jeffrey J. Harden                                                 #
# University of Notre Dame                                          #
# jeff.harden@nd.edu                                                #
#                                                                   #
# Jonathan Kropko                                                   #
# University of Virginia                                            #
# jkropko@virginia.edu                                              #
#                                                                   #
# Table A1 file                                                     #
# Last update: March 30, 2018                                       #
#####################################################################
## Load base packages ##
library(stats)
library(graphics)
library(grDevices)
library(utils)
library(datasets)
library(methods)
library(base)

## Survival package and random seed ##
library(survival)
set.seed(2552434)

## Define objects ##
T <- 100 # Number of time points
k <- 10 # Number of knots 
N <- 500 # Number of observations
p <- 3 # Number of covariates
sim <- 1000 # Number of simulations
cen <- .05 # Right-censor rate

beta <- as.matrix(c(1, .5, -.5, .75)) # True coefficient values
beta.aft <- -beta/5 # True coefficients in AFT parameterization [needed for the rweibull() function]
results <- matrix(NA, nrow = sim, ncol = 11) # Matrix to store results
nph.test <- array(NA, c(4, 2, sim))

## Simulation loop ##
for(i in 1:sim) {
	print(paste(c("Now working on iteration", i), collapse = " "))
	
# Generating event times #
	X <- cbind(1, matrix(rnorm(N*p, sd = .5), N, p)) # Covariates
	XB <- X%*%beta.aft
	
	lifetimes <- rweibull(N, shape = 5, scale = exp(XB)) # Draw event times
  censor <- runif(N) > cen # Randomly censor
	
	sim.data <- data.frame(lifetimes, X[ , -1]) # Create data objects
	sim.surv <- Surv(lifetimes, censor)

  # Exponential model #
  expo <- survreg(sim.surv ~ X1 + X2 + X3, data = sim.data, dist = "exponential")
  
  # Weibull model #
  weib <- survreg(sim.surv ~ X1 + X2 + X3, data = sim.data, dist = "weibull")

  # Cox model #
  coxm <- coxph(sim.surv ~ X1 + X2 + X3, data = sim.data, ties = "efron")
  
  # Results #
  results[i, ] <- c(-coef(expo), -coef(weib)/weib$scale, coef(coxm)) # Switch exponential and Weibull to PH estimates

 # Test for non-proportional hazards
  nph.test[ , , i] <- cox.zph(coxm, "log")$table[ , -1]
}

## Table A1 ##
# Coefficients #
tableA1.coef <- data.frame(Estimator = c("True", "Exponential", "Weibull", "Cox"),
                           b1 = round(c(beta[2], mean(results[ , 2]), mean(results[ , 6], na.rm = TRUE, trim = .01), mean(results[ , 9])), digits = 3),
                           b2 = round(c(beta[3], mean(results[ , 3]), mean(results[ , 7], na.rm = TRUE, trim = .01), mean(results[ , 10])), digits = 3),
                           b3 = round(c(beta[4], mean(results[ , 4]), mean(results[ , 8], na.rm = TRUE, trim = .01), mean(results[ , 11])), digits = 3))

# RMSE #
rmse <- function(x, true, tm = 0) sqrt(mean((x - true)^2, na.rm = TRUE, trim = tm))

tableA1.rmse <- data.frame(Estimator = c("Exponential", "Weibull", "Cox"),
                           b1 = round(c(rmse(results[ , 2], beta[2]), rmse(results[ , 6], beta[2]), rmse(results[ , 9], beta[2])), digits = 3),
                           b2 = round(c(rmse(results[ , 3], beta[3]), rmse(results[ , 7], beta[3]), rmse(results[ , 10], beta[3])), digits = 3),
                           b3 = round(c(rmse(results[ , 4], beta[4]), rmse(results[ , 8], beta[4]), rmse(results[ , 11], beta[4])), digits = 3))

sink("tableA1.txt", append = FALSE)
cat("Coefficient means (first 3 columns of Table A1) \n \n")
print(tableA1.coef[-1, ], row.names = FALSE)

cat("\n \n RMSE (last 3 columns of Table A1) \n \n")
print(tableA1.rmse, row.names = FALSE)
sink()

# save.image("tableA1.RData")

