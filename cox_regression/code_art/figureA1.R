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
# Figure A1 file                                                    #
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
source("rfunctions.R")
library(tidyverse)
library(survival)

set.seed(22902)
iter <- 1000
maxtime <- 100
maxN <- 1000
steps <- 20

# Mixing a random spline and a Weibull
lambda <- 6
p <- 1.005

weib <- function(t){
     dweibull(t, shape=p, scale=lambda)/(1 - pweibull(t, shape=p, scale=lambda))
}

randhazard <- baseline.build()$hazard
randspline <- function(t) randhazard[t]

betas <- rnorm(3, sd=.5)

mixing.sim <- as.numeric()

for(sub in seq(0, maxN, by=round(maxN/steps))){
     print(paste(c("Generating ", sub, " from the random spline, and ", 1000 - sub, " from Weibull"), collapse=""))
     
     if(sub==0){
          simdata.weibull <- sim.survdata(N=(maxN-sub), T=maxtime, type="none", hazard.fun = weib, num.data.frames = iter,
                                          fixed.hazard = FALSE, knots = 8, spline = TRUE, 
                                          X=NULL, beta=betas, C=3, mu=0, sd=.5, covariate=1,
                                          low=0, high=1, compare=median, censor = .1, 
                                          censor.cond = FALSE)
          
          results <- sapply(1:iter, FUN=function(r){
               
               durations <- simdata.weibull[[r]]$data$y
               failed <- simdata.weibull[[r]]$data$failed
               
               survdata <- Surv(durations, event=failed)
               xdata <- simdata.weibull[[r]]$xdata
               
               cox <- coxph(survdata ~ X1 + X2 + X3, data=xdata)
               weibull <- survreg(survdata ~ X1 + X2 + X3, data=xdata, dist="weibull")
               
               cox.coef <- coef(cox)
               weibull.coef <- (-coef(weibull)/weibull$scale)[-1]
               
               cox.mse <- mean((cox.coef - betas)^2)
               weibull.mse <- mean((weibull.coef - betas)^2)
               
               if(is.nan(cox.mse) | is.nan(weibull.mse) | cox.mse > 10 | weibull.mse > 10){
                    cox.mse <- NA
                    weibull.mse <- NA
                    skip <- 1
               } else skip <- 0
               
               return(c(cox.mse, weibull.mse, skip))
          })
     } else if(sub==maxN) {
          simdata.spline <- sim.survdata(N=sub, T=maxtime, type="none", hazard.fun = randspline, num.data.frames = iter,
                                         fixed.hazard = FALSE, knots = 8, spline = TRUE, 
                                         X=NULL, beta=betas, C=3, mu=0, sd=.5, covariate=1,
                                         low=0, high=1, compare=median, censor = .1, 
                                         censor.cond = FALSE)
          
          results <- sapply(1:iter, FUN=function(r){
               
               durations <- simdata.spline[[r]]$data$y
               failed <- simdata.spline[[r]]$data$failed
               
               survdata <- Surv(durations, event=failed)
               xdata <- simdata.spline[[r]]$xdata
               
               cox <- coxph(survdata ~ X1 + X2 + X3, data=xdata)
               weibull <- survreg(survdata ~ X1 + X2 + X3, data=xdata, dist="weibull")
               
               cox.coef <- coef(cox)
               weibull.coef <- (-coef(weibull)/weibull$scale)[-1]
               
               cox.mse <- mean((cox.coef - betas)^2)
               weibull.mse <- mean((weibull.coef - betas)^2)
               
               if(is.nan(cox.mse) | is.nan(weibull.mse) | cox.mse > 10 | weibull.mse > 10){
                    cox.mse <- NA
                    weibull.mse <- NA
                    skip <- 1
               } else skip <- 0
               
               return(c(cox.mse, weibull.mse, skip))
          })
     } else {
          simdata.spline <- sim.survdata(N=sub, T=maxtime, type="none", hazard.fun = randspline, num.data.frames = iter,
                                         fixed.hazard = FALSE, knots = 8, spline = TRUE, 
                                         X=NULL, beta=betas, C=3, mu=0, sd=.5, covariate=1,
                                         low=0, high=1, compare=median, censor = .1, 
                                         censor.cond = FALSE)
          simdata.weibull <- sim.survdata(N=(maxN-sub), T=maxtime, type="none", hazard.fun = weib, num.data.frames = iter,
                                          fixed.hazard = FALSE, knots = 8, spline = TRUE, 
                                          X=NULL, beta=betas, C=3, mu=0, sd=.5, covariate=1,
                                          low=0, high=1, compare=median, censor = .1, 
                                          censor.cond = FALSE)
          
          results <- sapply(1:iter, FUN=function(r){
               
               durations <- c(simdata.spline[[r]]$data$y, simdata.weibull[[r]]$data$y)
               failed <- c(simdata.spline[[r]]$data$failed, simdata.weibull[[r]]$data$failed)
               
               survdata <- Surv(durations, event=failed)
               xdata <- rbind(simdata.spline[[r]]$xdata, simdata.weibull[[r]]$xdata)
               
               cox <- coxph(survdata ~ X1 + X2 + X3, data=xdata)
               weibull <- survreg(survdata ~ X1 + X2 + X3, data=xdata, dist="weibull")
               
               cox.coef <- coef(cox)
               weibull.coef <- (-coef(weibull)/weibull$scale)[-1]
               
               cox.mse <- mean((cox.coef - betas)^2)
               weibull.mse <- mean((weibull.coef - betas)^2)
               
               if(is.nan(cox.mse) | is.nan(weibull.mse) | cox.mse > 10 | weibull.mse > 10){
                    cox.mse <- NA
                    weibull.mse <- NA
                    skip <- 1
               } else skip <- 0
               
               return(c(cox.mse, weibull.mse, skip))
          })
     }
     
     res <- data.frame(prop.randspline=sub/maxN, 
                       Cox=sqrt(mean(results[1,], na.rm=TRUE)),
                       Weibull=sqrt(mean(results[2,], na.rm=TRUE)),
                       skip = sum(results[3,]))
     mixing.sim <- rbind(mixing.sim, res)
}

mixing.sim <- mixing.sim %>%
     mutate(ratio = Weibull/Cox) 

theme_set(theme_gray(base_size = 18))

pdf("figureA1.pdf", width=12, height=7)

g <- ggplot(mixing.sim, aes(x=prop.randspline, y=ratio)) +
     geom_line(lwd = 1) +
     geom_point(lwd = 3) +
     geom_hline(yintercept = 1, lty=2) +
     xlab("Proportion of observations from random spline hazard") +
     ylab("(Weibull RMSE / Cox RMSE)") +
     scale_x_continuous(breaks = seq(0, maxN, length=(steps+1))/maxN) +
     scale_y_continuous(limits = c(.85, 1.71), breaks = seq(.8, 2, by=.1))
g

dev.off()

# save.image("figureA1.RData")
