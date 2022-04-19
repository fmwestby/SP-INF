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
# Figure A2 and Table A2 file                                       #
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
source("rfunctions.R")

### Simulation parameters ###
m <- 1000
T <- 100
k <- 8
N <- 500
p <- 3
pars <- c(.5, -.5, .75)

### Example simulation: Create a random monotonic hazard ###
set.seed(47647)

## Generate data ##
my.hazard <- function(t){
  T <- 100
  knots <- 8
  time <- 1:(T+1)
  k <- c(1,sort(sample(time[2:T], size=knots, replace=FALSE)), (T+1)) 
  heights <- c(0, sort(runif(knots)), 1)
  tk <- merge(data.frame(time), data.frame(time=k, heights), 
              by="time", all = TRUE)
  MonotonicSpline <- splinefun(x = tk$time, y = tk$heights, 
                               method = "hyman") 
  haz <- MonotonicSpline(time)[-1]
  return(haz[t])
}

e1 <- sim.survdata(N = N, T = T, hazard.fun = my.hazard, num.data.frames = m,
                   fixed.hazard = TRUE, knots = k, spline = TRUE, 
                   X = NULL, beta = pars, C = p, mu = 0, sd = .5, covariate = 1,
                   low = 0, high = 1, compare = median, censor = .05, 
                   censor.cond = FALSE)

## Graphs for Figure A2 ##
id <- 1
baseline <- gather(e1[[id]]$baseline, failure.PDF, failure.CDF, 
                   survivor, hazard, key = "type", value = "value")
baseline$type <- factor(baseline$type, 
                        levels = c("failure.PDF", "failure.CDF", "survivor", "hazard"),
                        labels = c("Failure PDF", "Failure CDF", "Survivor", "Hazard"))

# Panel (a) #
pdf("figureA2a.pdf")

ggplot(baseline[baseline$type == "Hazard", ], aes(x = time, y = value)) + 
  geom_line(lwd = 1, color = "gray50") +
  scale_y_continuous(breaks = seq(0, 1, .25)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("Hazard") + xlab("Time") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

d <- data.frame(quantity = "Simulated durations", value = e1[[id]]$data$y)
d <- rbind(d, data.frame(quantity = "Linear predictor",
                         value = e1[[id]]$xb))
d <- rbind(d, data.frame(quantity = "Exponentiated linear predictor",
                         value = e1[[id]]$exp.xb))

# Panel (b) #
pdf("figureA2b.pdf")

ggplot(d[d$quantity == "Simulated durations", ], aes(x = value)) +
  geom_histogram(binwidth = 1, color = "gray50") +
  xlab("Time") +
  ylab("Frequency") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

## Model estimation ##
results <- matrix(NA, nrow = m, ncol = 11) # Matrices to store results
colnames(results) <- c("expo.b0", "expo.b1", "expo.b2", "expo.b3", "weib.b0", "weib.b1", "weib.b2", "weib.b3", "cox.b1", "cox.b2", "cox.b3")
nph.test <- array(NA, c(4, 2, m))

for(i in 1:m){
  cat(i, " ")
  if (i %% 10 == 0) cat("\n")
  
  sim.data <- data.frame(y = e1[[i]]$data$y, # Create data objects
                         e1[[i]]$xdata,
                         failed = e1[[i]]$data$failed) 
  sim.surv <- Surv(sim.data$y, sim.data$failed)
  
  # Exponential, Weibull, and Cox models #
  expo <- survreg(sim.surv ~ X1 + X2 + X3, data = sim.data, dist = "exponential")
  weib <- survreg(sim.surv ~ X1 + X2 + X3, data = sim.data, dist = "weibull")
  coxm <- coxph(sim.surv ~ X1 + X2 + X3, data = sim.data, ties = "efron")
  
  # Results #
  results[i, ] <- c(-coef(expo), -coef(weib)/weib$scale, coef(coxm)) # Switch exponential and Weibull to PH estimates
  
  # Test for non-proportional hazards #
  nph.test[ , , i] <- cox.zph(coxm, "log")$table[ , -1]
}

## Table A2 ##
# Coefficients #
tableA2.coef <- data.frame(Estimator = c("True", "Exponential", "Weibull", "Cox"),
                           b1 = round(c(pars[1], mean(results[ , 2]), mean(results[ , 6], na.rm = TRUE, trim = .01), mean(results[ , 9])), digits = 3),
                           b2 = round(c(pars[2], mean(results[ , 3]), mean(results[ , 7], na.rm = TRUE, trim = .01), mean(results[ , 10])), digits = 3),
                           b3 = round(c(pars[3], mean(results[ , 4]), mean(results[ , 8], na.rm = TRUE, trim = .01), mean(results[ , 11])), digits = 3))

# RMSE #
rmse <- function(x, true, tm = 0) sqrt(mean((x - true)^2, na.rm = TRUE, trim = tm))

tableA2.rmse <- data.frame(Estimator = c("Exponential", "Weibull", "Cox"),
                           b1 = round(c(rmse(results[ , 2], pars[1]), rmse(results[ , 6], pars[1], tm = .01), rmse(results[ , 9], pars[1])), digits = 3),
                           b2 = round(c(rmse(results[ , 3], pars[2]), rmse(results[ , 7], pars[2], tm = .01), rmse(results[ , 10], pars[2])), digits = 3),
                           b3 = round(c(rmse(results[ , 4], pars[3]), rmse(results[ , 8], pars[3], tm = .01), rmse(results[ , 11], pars[3])), digits = 3))

sink("tableA2.txt", append = FALSE)
cat("Coefficient means (first 3 columns of Table A2) \n \n")
print(tableA2.coef[-1, ], row.names = FALSE)

cat("\n \n RMSE (last 3 columns of Table A2) \n \n")
print(tableA2.rmse, row.names = FALSE)
sink()

# save.image("figureA2-tableA2.RData")

