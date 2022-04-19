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
# Table A3 file                                                     #
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

### Packages and functions ###
library(tidyverse)
library(survival)
library(PermAlgo)
source("rfunctions.R")

### Simulation parameters ###
m <- 1000
T <- 100
k <- 8
p <- 3
pars <- c(.5, .25, .75)

### Example simulation: Draw hazard randomly, with TVC, N = 100 ###
set.seed(200)
N <- 100

## Generate data ##
e100 <- sim.survdata(N = N, T = T, type = "tvc", hazard.fun = NULL, num.data.frames = m,
                   fixed.hazard = TRUE, knots = k, spline = TRUE, 
                   X = NULL, beta = pars, C = p, mu = 0, sd = .5, covariate = 1,
                   low = 0, high = 1, compare = median, censor = .05, 
                   censor.cond = FALSE)

## Model estimation ##
results100 <- matrix(NA, nrow = m, ncol = 3) # Matrices to store results
colnames(results100) <- c("cox.b1", "cox.b2", "cox.b3")

for(i in 1:m){
  cat(i, " ")
  if (i %% 10 == 0) cat("\n")
  
  sim.data <- data.frame(Start = e100[[i]]$data$Start, Stop = e100[[i]]$data$Stop, Event = e100[[i]]$data$Event, e100[[i]]$xdata) # Create data objects
  
  # Cox model #
  coxm <- coxph(Surv(Start, Stop, Event) ~ X1 + X2 + X3, data = sim.data, ties = "efron")
  
  # Results #
  results100[i, ] <- exp(coef(coxm))
}

## Table A3, row 1 ##
# Coefficients #
tableA3.coef.row1 <- data.frame(N = 100, name = c("True", "Cox Estimate"),
                                b1 = round(c(pars[1], mean(results100[ , 1])), digits = 3),
                                b2 = round(c(pars[2], mean(results100[ , 2])), digits = 3),
                                b3 = round(c(pars[3], mean(results100[ , 3])), digits = 3))

# RMSE #
rmse <- function(x, true) sqrt(mean((x - true)^2))

tableA3.rmse.row1 <- data.frame(N = 100, model = c("Cox"),
                                b1 = round(rmse(results100[ , 1], pars[1]), digits = 3),
                                b2 = round(rmse(results100[ , 2], pars[2]), digits = 3), 
                                b3 = round(rmse(results100[ , 3], pars[3]), digits = 3))

### Example simulation: Draw hazard randomly, with TVC, N = 500 ###
set.seed(8553)
N <- 500

## Generate data ##
e500 <- sim.survdata(N = N, T = T, type = "tvc", hazard.fun = NULL, num.data.frames = m,
                     fixed.hazard = TRUE, knots = k, spline = TRUE, 
                     X = NULL, beta = pars, C = p, mu = 0, sd = .5, covariate = 1,
                     low = 0, high = 1, compare = median, censor = .05, 
                     censor.cond = FALSE)

## Model estimation ##
results500 <- matrix(NA, nrow = m, ncol = 3) # Matrices to store results
colnames(results500) <- c("cox.b1", "cox.b2", "cox.b3")

for(i in 1:m){
  cat(i, " ")
  if (i %% 10 == 0) cat("\n")
  
  sim.data <- data.frame(Start = e500[[i]]$data$Start, Stop = e500[[i]]$data$Stop, Event = e500[[i]]$data$Event, e500[[i]]$xdata) # Create data objects
  
  # Cox model #
  coxm <- coxph(Surv(Start, Stop, Event) ~ X1 + X2 + X3, data = sim.data, ties = "efron")
  
  # Results #
  results500[i, ] <- exp(coef(coxm))
}

## Table A3, row 2 ##
# Coefficients #
tableA3.coef.row2 <- data.frame(N = 500, name = c("True", "Cox Estimate"),
                                b1 = round(c(pars[1], mean(results500[ , 1])), digits = 3),
                                b2 = round(c(pars[2], mean(results500[ , 2])), digits = 3),
                                b3 = round(c(pars[3], mean(results500[ , 3])), digits = 3))

# RMSE #
rmse <- function(x, true) sqrt(mean((x - true)^2))

tableA3.rmse.row2 <- data.frame(N = 500, model = c("Cox"),
                                b1 = round(rmse(results500[ , 1], pars[1]), digits = 3),
                                b2 = round(rmse(results500[ , 2], pars[2]), digits = 3), 
                                b3 = round(rmse(results500[ , 3], pars[3]), digits = 3))

### Example simulation: Draw hazard randomly, with TVC, N = 1000 ###
set.seed(33378)
N <- 1000

## Generate data ##
e1000 <- sim.survdata(N = N, T = T, type = "tvc", hazard.fun = NULL, num.data.frames = m,
                     fixed.hazard = TRUE, knots = k, spline = TRUE, 
                     X = NULL, beta = pars, C = p, mu = 0, sd = .5, covariate = 1,
                     low = 0, high = 1, compare = median, censor = .05, 
                     censor.cond = FALSE)

## Model estimation ##
results1000 <- matrix(NA, nrow = m, ncol = 3) # Matrices to store results
colnames(results1000) <- c("cox.b1", "cox.b2", "cox.b3")

for(i in 1:m){
  cat(i, " ")
  if (i %% 10 == 0) cat("\n")
  
  sim.data <- data.frame(Start = e1000[[i]]$data$Start, Stop = e1000[[i]]$data$Stop, Event = e1000[[i]]$data$Event, e1000[[i]]$xdata) # Create data objects
  
  # Cox model #
  coxm <- coxph(Surv(Start, Stop, Event) ~ X1 + X2 + X3, data = sim.data, ties = "efron")
  
  # Results #
  results1000[i, ] <- exp(coef(coxm))
}

## Table A3, row 3 ##
# Coefficients #
tableA3.coef.row3 <- data.frame(N = 1000, name = c("True", "Cox Estimate"),
                                b1 = round(c(pars[1], mean(results1000[ , 1])), digits = 3),
                                b2 = round(c(pars[2], mean(results1000[ , 2])), digits = 3),
                                b3 = round(c(pars[3], mean(results1000[ , 3])), digits = 3))

# RMSE #
rmse <- function(x, true) sqrt(mean((x - true)^2))

tableA3.rmse.row3 <- data.frame(N = 1000, model = c("Cox"),
                                b1 = round(rmse(results1000[ , 1], pars[1]), digits = 3),
                                b2 = round(rmse(results1000[ , 2], pars[2]), digits = 3), 
                                b3 = round(rmse(results1000[ , 3], pars[3]), digits = 3))

## Print results ##
tableA3.coef <- rbind(tableA3.coef.row1[2, ], tableA3.coef.row2[2, ], tableA3.coef.row3[2, ])
tableA3.rmse <- rbind(tableA3.rmse.row1, tableA3.rmse.row2, tableA3.rmse.row3)

sink("tableA3.txt", append = FALSE)
cat("Coefficient means (first 3 columns of Table A3) \n \n")
print(tableA3.coef[ , -2], row.names = FALSE)

cat("\n \n RMSE (last 3 columns of Table A3) \n \n")
print(tableA3.rmse[ , -2], row.names = FALSE)
sink()

# save.image("tableA3.RData")