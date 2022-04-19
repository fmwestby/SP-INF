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
# Figure 3, Table 1, Table A4 (row 1), and Figure A3a file          #
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

### Example simulation: Draw hazard randomly ###
set.seed(999)

## Generate data ##
e1 <- sim.survdata(N=N, T=T, type="none", hazard.fun = NULL, num.data.frames = m,
                         fixed.hazard = TRUE, knots = k, spline = TRUE, 
                         X=NULL, beta=pars, C=p, mu=0, sd=.5, tvc = FALSE,
                         covariate=1, low=0, high=1, compare=median, 
                         censor = .05, censor.cond = FALSE)

## Graphs for Figure 3 ##
id <- 1
baseline <- gather(e1[[id]]$baseline, failure.PDF, failure.CDF, 
                   survivor, hazard, key = "type", value = "value")
baseline$type <- factor(baseline$type, 
                        levels = c("failure.PDF", "failure.CDF", "survivor", "hazard"),
                        labels = c("Failure PDF", "Failure CDF", "Survivor", "Hazard"))

# Panel (a) #
pdf("figure3a.pdf")

ggplot(baseline[baseline$type == "Hazard" & baseline$time!=T, ], aes(x = time, y = value)) + 
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
pdf("figure3b.pdf")

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

## Table 1 ##
# Coefficients #
table1.coef <- data.frame(Estimator = c("True", "Exponential", "Weibull", "Cox"),
                          b1 = round(c(pars[1], mean(results[ , 2]), mean(results[ , 6], na.rm = TRUE, trim = .01), mean(results[ , 9])), digits = 3),
                          b2 = round(c(pars[2], mean(results[ , 3]), mean(results[ , 7], na.rm = TRUE, trim = .01), mean(results[ , 10])), digits = 3),
                          b3 = round(c(pars[3], mean(results[ , 4]), mean(results[ , 8], na.rm = TRUE, trim = .01), mean(results[ , 11])), digits = 3))

# RMSE #
rmse <- function(x, true, tm = 0) sqrt(mean((x - true)^2, na.rm = TRUE, trim = tm))

table1.rmse <- data.frame(Estimator = c("Exponential", "Weibull", "Cox"),
               b1 = round(c(rmse(results[ , 2], pars[1]), rmse(results[ , 6], pars[1], tm = .01), rmse(results[ , 9], pars[1])), digits = 3),
               b2 = round(c(rmse(results[ , 3], pars[2]), rmse(results[ , 7], pars[2], tm = .01), rmse(results[ , 10], pars[2])), digits = 3),
               b3 = round(c(rmse(results[ , 4], pars[3]), rmse(results[ , 8], pars[3], tm = .01), rmse(results[ , 11], pars[3])), digits = 3))

sink("table1.txt", append = FALSE)
cat("Coefficient means (first 3 columns of Table 1) \n \n")
print(table1.coef[-1, ], row.names = FALSE)

cat("\n \n RMSE (last 3 columns of Table 1) \n \n")
print(table1.rmse, row.names = FALSE)
sink()

# NPH -- reported in Table A4 #
nph.p1 <- mean(nph.test[1, 2, ]) # Average p-value from NPH test for B1
nph.p2 <- mean(nph.test[2, 2, ]) # B2
nph.p3 <- mean(nph.test[3, 2, ]) # B3
nph.p4 <- mean(nph.test[4, 2, ]) # Global

tableA4.coef.row1 <- data.frame(Condition = "With PH",
                           b1 = round(mean(results[ , 9]), digits = 3),
                           b2 = round(mean(results[ , 10]), digits = 3),
                           b3 = round(mean(results[ , 11]), digits = 3))

tableA4.pval.row1 <- data.frame(Condition = "With PH",
                                x1 = round(nph.p1, digits = 3),
                                x2 = round(nph.p2, digits = 3),
                                x3 = round(nph.p3, digits = 3),
                                global = round(nph.p4, digits = 3))

sink("tableA4.txt", append = FALSE)
cat("Note: This output is the first row of Table A4. Row 2 gets added to this file after running tableA4-figureA3b.R. \n \n \n ")
cat("Coefficient means (first 3 columns of Table A4, row 1) \n \n")
print(tableA4.coef.row1, row.names = FALSE)

cat("\n \n Mean p-value (last 4 columns of Table A4, row 1) \n \n")
print(tableA4.pval.row1, row.names = FALSE)
cat("\n \n")
sink()

### Figure A3a ###
last.nph <- cox.zph(coxm, "log")
d1 <- data.frame(time = exp(last.nph$x), y = last.nph$y[ , 1])

pdf("figureA3a.pdf")

ggplot(d1, aes(x = time, y = y)) + 
  geom_point(color = "gray50", lwd = 1.5) +
  geom_smooth(color = "gray50", lwd = 2) +
  scale_y_continuous(breaks = seq(-7, 7, by = 1)) +
  ylab(expression("Scaled Schoenfeld Residuals "(X[1]))) + xlab("Time") +
  theme(legend.position = "none", axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1))

dev.off()

# save.image("figure3-table1-tableA4-figureA3a.RData")

