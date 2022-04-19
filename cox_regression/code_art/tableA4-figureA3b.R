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
# Table A4 and Figure A3b file                                      #
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

### Example simulation: Draw hazard randomly, with NPH violation ###
set.seed(28823)

## Generate data ##
sim.nph <- sim.survdata(N = N, T = T, type = "tvbeta", hazard.fun = NULL,
                   num.data.frames = m, fixed.hazard = TRUE,
                   knots = k, spline = TRUE, X = NULL, beta = pars,
                   C = p, mu = 0, sd = .5, covariate = 1,
                   low = 0, high = 1, compare = median, censor = .05, 
                   censor.cond = FALSE)

## Model estimation ##
results <- matrix(NA, nrow = m, ncol = 3) # Matrices to store results
colnames(results) <- c("cox.b1", "cox.b2", "cox.b3")
nph.test <- array(NA, c(4, 2, m))

for(i in 1:m){
  cat(i, " ")
  if (i %% 10 == 0) cat("\n")
  
  sim.data <- data.frame(y = sim.nph[[i]]$data$y, # Create data objects
                         X1 = sim.nph[[i]]$data$X1,
                         X2 = sim.nph[[i]]$data$X2,
                         X3 = sim.nph[[i]]$data$X3,
                         failed = sim.nph[[i]]$data$failed) 
  sim.surv <- Surv(sim.data$y, sim.data$failed)
  
  # Cox model #
  coxm <- coxph(sim.surv ~ X1 + X2 + X3, data = sim.data, ties = "efron")
  
  # Results #
  results[i, ] <- coef(coxm)

  # Test for non-proportional hazards #
  nph.test[ , , i] <- cox.zph(coxm, "log")$table[ , -1]
}

## Table A4, row 2 ##
# Coefficients -- without PH #
tableA4.coef.row2 <- data.frame(Condition = "Without PH",
                                b1 = round(mean(results[ , 1]), digits = 3),
                                b2 = round(mean(results[ , 2]), digits = 3),
                                b3 = round(mean(results[ , 3]), digits = 3))

# NPH -- without PH #
nph.p1 <- mean(nph.test[1, 2, ]) # Average p-value from NPH test for B1
nph.p2 <- mean(nph.test[2, 2, ]) # B2
nph.p3 <- mean(nph.test[3, 2, ]) # B3
nph.p4 <- mean(nph.test[4, 2, ]) # Global

tableA4.pval.row2 <- data.frame(Condition = "Without PH",
                                x1 = round(nph.p1, digits = 3),
                                x2 = round(nph.p2, digits = 3),
                                x3 = round(nph.p3, digits = 3),
                                global = round(nph.p4, digits = 3))

sink("tableA4.txt", append = TRUE)
cat("Note: This output is the second row of Table A4. Row 1 gets added to this file after running figure3-table1-tableA4-figureA3a.R. \n \n \n ")
cat("Coefficient means (first 3 columns of Table A4, row 2) \n \n")
print(tableA4.coef.row2, row.names = FALSE)

cat("\n \n Mean p-value (last 4 columns of Table A4, row 2) \n \n")
print(tableA4.pval.row2, row.names = FALSE)
sink()

### Figure A3b ###
last.nph <- cox.zph(coxm, "log")
d1 <- data.frame(time = exp(last.nph$x), y = last.nph$y[ , 1])

pdf("figureA3b.pdf")

ggplot(d1, aes(x = time, y = y)) + 
  geom_point(color = "gray50", lwd = 1.5) +
  geom_smooth(color = "gray50", lwd = 2) +
  scale_y_continuous(breaks = seq(-7, 7, by = 1)) +
  ylab(expression("Scaled Schoenfeld Residuals "(X[1]))) + xlab("Time") +
  theme(legend.position = "none", axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1))

dev.off()

# save.image("tableA4-figureA3b.RData")
