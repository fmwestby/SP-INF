require(tidyverse)
require(PermAlgo)

baseline.build <- function(T=100, knots = 8, spline = TRUE){
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
  return(baseline)
}

user.baseline <- function(user.fun, T){
  baseline <- data.frame(time=1:T)
  baseline <- baseline %>%
    mutate(hazard = user.fun(time),
           cum.hazard = cumsum(hazard),
           survivor = exp(-cum.hazard),
           failure.CDF = 1 - survivor,
           failure.PDF = c(0, diff(failure.CDF))) %>%
    dplyr::select(-cum.hazard)
  return(baseline)
}

generate.lm <- function(baseline, X=NULL, beta=NULL, N=1000, C=3, mu=0, sd=1, type="none", T=100, censor=.1){
  if(type=="none"){
    if(is.null(X)) X <- cbind(matrix(rnorm(N*C, mean=mu, sd=sd), N, C))
    if(is.null(beta)) beta <- as.matrix(rnorm(ncol(X), mean=0, sd=.1))
    XB <- X%*%beta
    survival <- t(sapply(XB, FUN=function(x){baseline$survivor^exp(x)}, simplify=TRUE)) 		# S(t) = S_0(t) ^ exp(XB)
    y <- apply(survival, 1, FUN=function(x){
      which.max(diff(x < runif(1)))
    })
    data <- data.frame(X)
    data$y <- y
  } else if(type=="tvc"){
    X <- matrix(ncol = 3, nrow = N*T)
    X[ , 1] <- rnorm(N*T) # Time-varying covariates
    X[ , 2] <- rpois(N*T, lambda = 2)
    X[ , 3] <- rep(rbinom(N, 1, .3), each = T)  # Static binary covariate
    colnames(X) <- c("X1", "X2", "X3")
    if(is.null(beta)) beta <- as.matrix(rlnorm(ncol(X), meanlog=1, sdlog=.1))
    XB <- matrix(X%*%beta, N, T, byrow=TRUE)
    survival <- t(apply(XB, 1, FUN=function(x){baseline$survivor^exp(x)})) 		# S(t) = S_0(t) ^ exp(XB)
    lifetimes <- apply(survival, 1, FUN=function(x){
      which.max(diff(x < runif(1)))
    })
    cen <- lifetimes
    cen[runif(N) > censor] <- T
    data <- permalgorithm(N, T, X, 
                          XmatNames = colnames(X), 
                          eventRandom = lifetimes, 
                          censorRandom = cen, 
                          betas = log(beta), 
                          groupByD = FALSE)
  } else if(type=="tvbeta"){
    X <- cbind(matrix(rnorm(N*C, mean=mu, sd=sd), N, C))
    if(is.null(beta)) beta <- as.matrix(rnorm(ncol(X), mean=0, sd=.1))
    beta.mat <- data.frame(time = 1:T) %>%
      mutate(beta1 = beta[1]*log(time),
             beta2 = beta[2],
             beta3 = beta[3]) %>%
      dplyr::select(-time)
    XB <- apply(as.matrix(beta.mat), 1, FUN=function(b){
      as.matrix(X)%*%b
    })
    survival <- t(apply(XB, 1, FUN=function(x){baseline$survivor^exp(x)})) 		# S(t) = S_0(t) ^ exp(XB)
    lifetimes <- apply(survival, 1, FUN=function(x){
      which.max(diff(x < runif(1)))
    })
    data <- data.frame(X) %>%
      mutate(y = lifetimes,
             failed = !(runif(N) < censor))
  } else {stop("type must be one of 'none', 'tvc', or 'tvbeta'")}
  return(list(data=data, beta=beta, XB=XB, exp.XB = exp(XB),survmat=survival))
}

make.margeffect <- function(baseline, xb, beta=NULL, covariate=1, low=0, high=1, 
                            compare=median){
  X0 <- dplyr::select(xb$data, starts_with("X"))
  X1 <- dplyr::select(xb$data, starts_with("X"))
  
  X0[,covariate] <- low
  X1[,covariate] <- high
  
  beta <- xb$beta
  
  XB0 <- as.matrix(X0)%*%beta
  survival <- t(sapply(XB0, FUN=function(x){baseline$survivor^exp(x)}, simplify=TRUE)) 
  y0 <- apply(survival, 1, FUN=function(x){
    which.max(diff(x < runif(1)))
  })
  
  XB1 <- as.matrix(X1)%*%beta
  survival <- t(sapply(XB1, FUN=function(x){baseline$survivor^exp(x)}, simplify=TRUE)) 
  y1 <- apply(survival, 1, FUN=function(x){
    which.max(diff(x < runif(1)))
  })
  
  marg.effect <- compare(y1 - y0)
  data.low <- list(x = X0, y = y0)
  data.high <- list(x = X1, y = y1)
  
  return(list(marg.effect = marg.effect,
              data.low = data.low,
              data.high = data.high))
}

censor.x <- function(x, censor=.1){
  beta.cen <- as.matrix(rnorm(ncol(x), mean=0, sd=.1))
  xb.cen <- as.matrix(x)%*%beta.cen + rnorm(nrow(x), mean=0, sd=.1)
  cen <- (xb.cen > quantile(xb.cen, (1-censor)))
  return(cen)
}

sim.survdata <- function(N=1000, T=100, type="none", hazard.fun = NULL, num.data.frames = 1,
                         fixed.hazard = TRUE, knots = 8, spline = TRUE, 
                         X=NULL, beta=NULL, C=3, mu=0, sd=.5, tvc = FALSE,
                         covariate=1, low=0, high=1, compare=median, 
                         censor = .1, censor.cond = FALSE){
  ifelse(is.null(hazard.fun),
         baseline <- baseline.build(T=T, knots=knots, spline=spline),
         baseline <- user.baseline(hazard.fun, T))
  
  result <- lapply(1:num.data.frames, FUN=function(i){
    
    if(!fixed.hazard & is.null(hazard.fun)) baseline <- baseline.build(T=T, knots=knots, spline=spline)
    
    xb <- generate.lm(baseline, X=X, beta=beta, N=N, T=T, censor=censor, type=type)
    data <- xb$data
    me <- make.margeffect(baseline, xb, beta, covariate, low, high, compare)
    if(type=="none") ifelse(censor.cond, 
           data$failed <- !censor.x(xb$X, censor=censor),
           data$failed <- !(runif(N) < censor))
    
    return(list(data = data,
                xdata = dplyr::select(data, starts_with("X")),
                baseline=baseline,
                xb = xb$XB,
                exp.xb = xb$exp.XB,
                betas = xb$beta,
                ind.survive = xb$survmat,
                marg.effect = me$marg.effect,
                marg.effect.data = list(low = me$data.low,
                                        high = me$data.high)))            
  })
  ifelse(num.data.frames == 1, return(result[[1]]), return(result))
}

baseline.plot <- function(baseline, ...){
  require(ggplot2)
  require(tidyr)
  baseline <- gather(baseline, failure.PDF, failure.CDF, 
                     survivor, hazard, key="type", value="value")
  baseline$type <- factor(baseline$type, 
                          levels = c("failure.PDF", "failure.CDF", "survivor", "hazard"),
                          labels = c("Failure PDF", "Failure CDF", "Survivor", "Hazard"))
  g <- ggplot(baseline, aes(x=time, y=value)) +
    geom_line() +
    facet_wrap(~ type, scales="free") +
    xlab("Time") +
    ylab("Survival Model Function") +
    ggtitle("Simulated Baseline Functions")
  return(g)
}

data.plot <- function(data, xb, exp.xb, ...){
  d <- data.frame(quantity = "Simulated durations", value = data$y)
  d <- rbind(d, data.frame(quantity = "Linear predictor",
                           value = xb))
  d <- rbind(d, data.frame(quantity = "Exponentiated linear predictor",
                           value = exp.xb))
  g <- ggplot(d, aes(x = value)) +
    geom_histogram() +
    facet_wrap(~ quantity, scales = "free") +
    xlab("Value") +
    ylab("Frequency") +
    ggtitle("Histograms of Simulated Data")
  return(g)
}

plot.survsim <- function(survsim, type="both", ...){
  if(!is.null(type) & !(type %in% c("baseline", "hist", "both"))) 
    stop("type must be one of baseline, hist, or both")
  require(gridExtra)
  if(type=="baseline") g <- baseline.plot(survsim$baseline, ...)
  if(type=="hist") g <- data.plot(survsim$data, survsim$xb, survsim$exp.xb, ...)
  if(type=="both"){
    g1 <- baseline.plot(survsim$baseline, ...)
    g2 <- data.plot(survsim$data, survsim$xb, survsim$exp.xb, ...)
    g <- grid.arrange(g1, g2, nrow=2)
  } 
  return(g)
} 