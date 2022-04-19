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
# Figure 1 and Figure 2 file                                        #
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

## tidyverse package and random seed ##
library(tidyverse)
set.seed(22902)

## Define objects ##
T <- 100
knots <- 8
spline <- TRUE

## Run baseline.build() function ##
time <- 1:(T+1)
k <- c(1,sort(sample(time[2:T], size=knots, replace=FALSE)), (T+1)) 
heights <- c(0, sort(runif(knots)), 1)
tk <- merge(data.frame(time), data.frame(time=k, heights), 
              by="time", all = !spline)
MonotonicSpline <- splinefun(x = tk$time, y = tk$heights, 
                               method = "hyman") 
bl.failure.CDF <- MonotonicSpline(time)
baseline <- data.frame(time = time[-(T+1)],
                         failure.PDF = diff(bl.failure.CDF),
                         failure.CDF = bl.failure.CDF[-1],
                         survivor = 1 - bl.failure.CDF[-1])
baseline$hazard <-baseline$failure.PDF/(1 - bl.failure.CDF[-(T+1)])

d1 <- data.frame(knots = k, heights = heights)
d2 <- data.frame(time = 1:T, cdf = baseline$failure.CDF, pdf = baseline$failure.PDF, survivor = baseline$survivor, hazard = baseline$hazard)

## Graphs ##

# Figure 1a
pdf("figure1a.pdf")

ggplot(d1, aes(x = knots, y = heights)) + 
  geom_point(shape = 16, position = "identity", color = "gray50", lwd = 4) +
  scale_y_continuous(breaks = seq(0, 1, .25)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("Cumulative P(Failure)") + xlab("Time") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

# Figure 1b
pdf("figure1b.pdf")

df <- rbind(c(1,0,0,0,0), d2)
ggplot(d1, aes(x = knots, y = heights)) + 
  geom_point(shape = 16, position = "identity", color = "gray50", lwd = 4) +
  geom_line(data = df, aes(x = time, y = cdf), color = "gray50", lwd = 1) +
  scale_y_continuous(breaks = seq(0, 1, .25)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("Cumulative P(Failure)") + xlab("Time") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

# Figure 1c
pdf("figure1c.pdf")

ggplot(d2, aes(x = time, y = pdf)) + 
  geom_line(color = "gray50", lwd = 1) +
  scale_y_continuous(breaks = seq(0, 1, .01)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("P(Failure)") + xlab("Time") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

# Figure 1d
pdf("figure1d.pdf")

ggplot(d2, aes(x = time, y = survivor)) + 
  geom_line(color = "gray50", lwd = 1) +
  scale_y_continuous(breaks = seq(0, 1, .1)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("P(Survival)") + xlab("Time") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

# Figure 1e
pdf("figure1e.pdf")

dh <- d2[-c(T-1,T),]
ggplot(dh, aes(x = time, y = hazard)) + 
  geom_line(color = "gray50", lwd = 1) +
  scale_y_continuous(breaks = seq(0, 1, .1)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("Hazard") + xlab("Time") +
  theme(legend.position = "none", axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

# Figure 2 
pdf("figure2.pdf", width=10, height=6)

d3 <- d2 %>%
      dplyr::select(time, survivor) %>%
      mutate(ind.surv = survivor^(1.5)) %>%
      rename(`exp(XB) = 1` = survivor,
             `exp(XB) = 1.5` = ind.surv) %>%
      gather(`exp(XB) = 1`, `exp(XB) = 1.5`, key="ELP",
             value = "value")
      

ggplot(d3, aes(x = time, y = value, lty=ELP)) + 
  geom_line(color = "gray50", lwd = 1) +
  scale_y_continuous(breaks = seq(0, 1, .1)) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  ylab("P(Survival)") + xlab("Time") +
      labs(lty="") +
      geom_hline(yintercept = .6247, lwd=.1) +
      geom_vline(xintercept = 46, lwd=.1) +
      geom_point(aes(x=46, y=.6247), size=3) +
      geom_text(aes(x=20, y=.6, label="Uniform draw = 0.6247"), size = 6.5) +
      geom_text(aes(x=46, y=0, label="Duration draw = 46"), size = 6.5) +
      #scale_linetype(labels = c("exp(X\u03B2) = 1","exp(X\u03B2) = 1.5")) +
  scale_linetype(labels = c(expression("exp(X"*beta*") = 1.0"), expression("exp(X"*beta*") = 1.5"))) +
  theme(legend.text=element_text(size=15), axis.text = element_text(size = 15), axis.title.y = element_text(size = 20, vjust = 1.5), axis.title.x = element_text(size = 20, vjust = -.1)) + labs(fill = "")

dev.off()

# save.image("figure1-figure2.RData")