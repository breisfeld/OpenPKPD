## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7
)

## ---- message=FALSE-----------------------------------------------------------
library(tidyr)
library(dplyr)
library(ggplot2)
library(xgxr)

## -----------------------------------------------------------------------------
# xgx_create_rmarkdown(type = "pk", open_file = FALSE)

## ---- fig.height=7------------------------------------------------------------
#if (sessionInfo()$otherPkgs$ggplot2$Version == "2.2.1") {
#   nsubj <- 50
#   ntime <- 8
#   time <- rep(c(1, 2, 4, 8, 12, 24, 36, 48), nsubj)
#   id <- sort(rep(seq(1, nsubj), ntime))
#   trt <- sort(rep(c(25, 50, 100, 150, 300), ntime * nsubj / 5))
#   ka <- rep(rlnorm(nsubj, -0.5, 0.3), each = ntime)
#   ke <- rep(rlnorm(nsubj, -3, 0.3), each = ntime)
#   conc <- trt * (ka * ke / (ka - ke)) * (exp(-time * ke) - exp(-time * ka)) * (rep(stats::rlnorm(ntime * nsubj, 0.3, 0.1)))
 
#   data <- data.frame(TIME = time, CONC = conc, ID = id, TRT = trt)
#   xgx_PK_summary(data = data, labels = list(TRT = "Dose"),
#                  units_dataset = list(TIME = "Hours", CONC = "ng/mL", TRT = "mg"))
#} else {
#  print("Currently only works with ggplot2 version 2.2.1 (on DaVinci), and not version 3")
#}

## -----------------------------------------------------------------------------
dirs <- list(
  parent_dir = tempdir(),
  rscript_dir = tempdir(),
  rscript_name = "example.R",
  results_dir = tempdir(),
  filename_prefix = "example_")
data <- data.frame(x = 1:1000, y = stats::rnorm(1000))
g <- xgx_plot(data = data, aes(x = x, y = y)) +
  geom_point()
xgx_save(width = 4, height = 4, dirs = dirs, filename_main = "example_plot", status = "DRAFT")

## -----------------------------------------------------------------------------
data <- data.frame(x = 1:1000, y = stats::rnorm(1000))
g <- xgx_plot(data = data, aes(x = x, y = y)) +
  geom_point()
filename = file.path(tempdir(), "png_example.png")
ggsave(filename, plot = g, height = 4, width = 4, dpi = 75)
xgx_annotate_status_png(filename, "./ExampleScript.R")

## -----------------------------------------------------------------------------
x <- data.frame(ID = c(1, 2), SEX = c("male", "female"))
data <- xgx_save_table(x, dirs = dirs, filename_main = "ExampleTable")
knitr::kable(data)

## -----------------------------------------------------------------------------
xgx_plot(mtcars, aes(x = cyl, y = mpg)) + geom_point()

## -----------------------------------------------------------------------------
theme_set(xgx_theme())

## Alternative, equivalent function:
xgx_theme_set()

## ---- fig.width=4, fig.height=2-----------------------------------------------
# time <- rep(seq(1,10),5)
# id <- sort(rep(seq(1,5), 10))
# conc <- exp(-time)*sort(rep(rlnorm(5),10))
# 
# data <- data.frame(time = time, concentration  = conc, id = factor(id))
# xgx_plot() + xgx_geom_spaghetti(data = data, mapping = aes(x = time, y = concentration, group = id, color = id))
# 
# xgx_spaghetti(data = data, mapping = aes(x = time, y = concentration, group = id, color = id))

## ---- fig.width=4, fig.height=2-----------------------------------------------
data <- data.frame(x = rep(c(1, 2, 3), each = 20),
                   y = rep(c(1, 2, 3), each = 20) + stats::rnorm(60),
                   group = rep(1:3, 20))

xgx_plot(data,aes(x = x, y = y)) +
    xgx_stat_ci(conf_level = .95)

xgx_plot(data,aes(x = x, y = y)) +
    xgx_stat_pi(percent = .95)

xgx_plot(data,aes(x = x, y = y)) +
    xgx_stat_ci(conf_level = .95, geom = list("pointrange","line"))

xgx_plot(data,aes(x = x, y = y)) +
    xgx_stat_ci(conf_level = .95, geom = list("ribbon","line"))

xgx_plot(data,aes(x = x, y = y, group = group, color = factor(group))) + 
    xgx_stat_ci(conf_level = .95, alpha =  0.5,
                position = position_dodge(width = 0.5))

## ---- fig.width=4, fig.height=2-----------------------------------------------
# plotting lognormally distributed data
data <- data.frame(x = rep(c(1, 2, 3), each = 20),
                   y = 10^(rep(c(1, 2, 3), each = 20) + stats::rnorm(60)),
                   group = rep(1:3, 20))
xgx_plot(data, aes(x = x, y = y)) + 
 xgx_stat_ci(conf_level = 0.95, distribution = "lognormal")
 
# note: you DO NOT need to use both distribution = "lognormal" and scale_y_log10()
xgx_plot(data,aes(x = x, y = y)) + 
 xgx_stat_ci(conf_level = 0.95) + xgx_scale_y_log10()

# plotting binomial data
data <- data.frame(x = rep(c(1, 2, 3), each = 20),
                  y = rbinom(60, 1, rep(c(0.2, 0.6, 0.8), each = 20)),
                  group = rep(1:3, 20))
xgx_plot(data, aes(x = x, y = y)) + 
 xgx_stat_ci(conf_level = 0.95, distribution = "binomial")


# Example plotting the percent of subjects in a categorical covariate group by treatment.

set.seed(12345)
data = data.frame(x = 120*exp(rnorm(100,0,1)),
                 response = sample(c("Trt1", "Trt2", "Trt3"), 100, replace = TRUE),
                 covariate = factor(sample(c("White","Black","Asian","Other"), 100, replace = TRUE), 
                                    levels = c("White", "Black", "Asian", "Other")))

xgx_plot(data = data) +
 xgx_stat_ci(mapping = aes(x = response, response = covariate),
             distribution = "ordinal") +
 xgx_stat_ci(mapping = aes(x = 1, response = covariate), geom = "hline",
             distribution = "ordinal") +
 scale_y_continuous(labels = scales::percent_format()) + 
 facet_wrap(~covariate) + 
 xlab("Treatment group") + ylab("Percent of subjects by category")


## -----------------------------------------------------------------------------

# plotting 
set.seed(12345)
data = data.frame(x = 120*exp(rnorm(100,0,1)),
              response = sample(c("Mild","Moderate","Severe"), 100, replace = TRUE),
              covariate = sample(c("Male","Female"), 100, replace = TRUE)) %>%
  mutate(y = (50 + 20*x/(200 + x))*exp(rnorm(100, 0, 0.3)))

# plotting a lognormally distributed variable by quartiles of x
xgx_plot(data = data) +
  xgx_stat_ci(mapping = aes(x = x, y = y, colour = covariate),
              distribution = "lognormal", bins = 4)

# plotting ordinal or multinomial data, by quartiles of x
xgx_plot(data = data) +
  xgx_stat_ci(mapping = aes(x = x, response = response, colour = covariate),
              distribution = "ordinal", bins = 4) +
  scale_y_continuous(labels = scales::percent_format()) + facet_wrap(~response)

xgx_plot(data = data) +
  xgx_stat_ci(mapping = aes(x = x, response = response, colour = response),
              distribution = "ordinal", bins = 4) +
  scale_y_continuous(labels = scales::percent_format()) + facet_wrap(~covariate)


## -----------------------------------------------------------------------------
set.seed(123456)

Nsubj <- 10
Doses <- c(0, 25, 50, 100, 200)
Ntot <- Nsubj*length(Doses)
times <- c(0,14,30,60,90)

dat1 <- data.frame(ID = 1:(Ntot),
                   DOSE = rep(Doses, Nsubj),
                   E0 = 50*rlnorm(Ntot, 0, 0.3),
                   Emax = 100*rlnorm(Ntot, 0, 0.3),
                   ED50 = 50*rlnorm(Ntot, 0, 0.3)) %>%
  dplyr::mutate(Response = (E0 + Emax*DOSE/(DOSE + ED50))*rlnorm(Ntot, 0, 0.3)  ) %>%
  merge(data.frame(ID = rep(1:(Ntot), each = length(times)), Time = times), by = "ID") 

gg <- xgx_plot(data = dat1, aes(x = DOSE, y = Response))
gg <- gg + geom_point()
gg

gg + geom_smooth(method = "nlsLM", 
                 formula = y ~ E0 + Emax*x/(ED50 + x),
                 method.args = list(start = list(E0 = 1, ED50 = 1, Emax = 1),
                                    lower = c(-Inf, 0, -Inf)))

## -----------------------------------------------------------------------------
gg + xgx_geom_smooth_emax()

gg + 
  xgx_geom_smooth_emax(geom = "ribbon", color = "black", fill = NA, linetype = "dashed") + 
  xgx_geom_smooth_emax(geom = "line", color = "red")


## -----------------------------------------------------------------------------
mod <- nlsLM(formula = Response ~ E0 + Emax * DOSE / (ED50 + DOSE),
             data = dat1,
             start = list(E0 = 1, ED50 = 1, Emax = 1),
                                    lower = c(-Inf, 0, -Inf))

predict(mod,
            newdata = data.frame(DOSE = c(0, 25, 50, 100, 200)),
            se.fit = TRUE)

predict(mod,
            newdata = data.frame(DOSE = c(0, 25, 50, 100, 200)),
            se.fit = TRUE, interval = "confidence", level = 0.95)


## -----------------------------------------------------------------------------

# example with ordinal data (method = "polr")
set.seed(12345)
data = data.frame(x = 120*exp(stats::rnorm(100,0,1)),
                  response = sample(c("Mild","Moderate","Severe"), 100, replace = TRUE),
                  covariate = sample(c("Male","Female"), 100, replace = TRUE)) %>%
  dplyr::mutate(y = (50 + 20*x/(200 + x))*exp(stats::rnorm(100, 0, 0.3)))

# example coloring by the response categories
xgx_plot(data = data) +
  xgx_stat_smooth(mapping = ggplot2::aes(x = x, response = response,
                                         colour = response, fill = response),
                  method = "polr") +
  ggplot2::scale_y_continuous(labels = scales::percent_format())

# example faceting by the response categories, coloring by a different covariate
xgx_plot(data = data) +
  xgx_stat_smooth(mapping = ggplot2::aes(x = x, response = response,
                                         colour = covariate, fill = covariate),
                  method = "polr", level = 0.80) +
  ggplot2::facet_wrap(~response) +
  ggplot2::scale_y_continuous(labels = scales::percent_format())


## -----------------------------------------------------------------------------
df <- data.frame(x = c(0, stats::rlnorm(1000, 0, 1)),
                 y = c(0, stats::rlnorm(1000, 0, 3)))
xgx_plot(data = df, aes(x = x, y = y)) + 
  geom_point() + 
  xgx_scale_x_log10() + 
  xgx_scale_y_log10()

## ---- fig.height=3.5, warning=FALSE-------------------------------------------
conc <- 10^(seq(-3, 3, by = 0.1))
ec50 <- 1
data <- data.frame(concentration = conc, 
                  bound_receptor = 1 * conc / (conc + ec50))
gy <- xgx_plot(data, aes(x = concentration, y = bound_receptor)) + 
  geom_point() + 
  geom_line() + 
  xgx_scale_x_log10() +
  xgx_scale_y_reverselog10()

gx <- xgx_plot(data, aes(x = bound_receptor, y = concentration)) + 
  geom_point() + 
  geom_line() + 
  xgx_scale_y_log10() +
  xgx_scale_x_reverselog10()

gridExtra::grid.arrange(gy, gx, nrow = 1)

## ---- fig.height=3.5, warning=FALSE-------------------------------------------

Nsubj <- 10
Doses <- c(0, 25, 50, 100, 200)
Ntot <- Nsubj*length(Doses)
times <- c(0,14,30,60,90)

dat1 <- data.frame(ID = 1:(Ntot),
                   DOSE = rep(Doses, Nsubj),
                   PD0 = rlnorm(Ntot, log(100), 1),
                   Kout = exp(rnorm(Ntot,-2, 0.3)),
                   Imax = 1,
                   ED50 = 25) %>%
  dplyr::mutate(PDSS = PD0*(1 - Imax*DOSE/(DOSE + ED50))*exp(rnorm(Ntot, 0.05, 0.3))  ) %>%
  merge(data.frame(ID = rep(1:(Ntot), each = length(times)), Time = times), by = "ID") %>%
  dplyr::mutate(PD = ((PD0 - PDSS)*(exp(-Kout*Time)) + PDSS), 
                PCHG = (PD - PD0)/PD0)

ggplot2::ggplot(dat1 %>% subset(Time == 90), 
                ggplot2::aes(x = DOSE, y = PCHG, group = DOSE)) +
  ggplot2::geom_boxplot() +
  xgx_theme() +
  xgx_scale_y_percentchangelog10() +
  ylab("Percent Change from Baseline") +
  xlab("Dose (mg)")

ggplot2::ggplot(dat1, 
                ggplot2::aes(x = Time, y = PCHG, group = ID, color = factor(DOSE))) +
  ggplot2::geom_line() +
  xgx_theme() +
  xgx_scale_y_percentchangelog10() +
  guides(color = guide_legend(title = "Dose (mg)")) +
  ylab("Percent Change from Baseline")

dat2 <- data.frame(ID = 1:(Ntot),
                  DOSE = rep(Doses, Nsubj),
                  PD0 = rlnorm(Ntot, log(100), 1),
                  Kout = exp(rnorm(Ntot,-2, 0.3)),
                  Emax = 50*rlnorm(Ntot, 0, 0.3),
                  ED50 = 300) %>%
 dplyr::mutate(PDSS = PD0*(1 + Emax*DOSE/(DOSE + ED50))*exp(rnorm(Ntot, -1, 0.3))  ) %>%
  merge(data.frame(ID = rep(1:(Ntot), each = length(times)), Time = times), by = "ID") %>%
  dplyr::mutate(PD = ((PD0 - PDSS)*(exp(-Kout*Time)) + PDSS), 
                PCHG = (PD - PD0)/PD0)

ggplot2::ggplot(dat2, ggplot2::aes(x = DOSE, y = PCHG, group = DOSE)) +
  ggplot2::geom_boxplot() +
  xgx_theme() +
  xgx_scale_y_percentchangelog10() +
  ylab("Percent Change from Baseline") +
  xlab("Dose (mg)")

ggplot2::ggplot(dat2, 
                ggplot2::aes(x = Time, y = PCHG, group = ID, color = factor(DOSE))) +
  ggplot2::geom_line() +
  xgx_theme() +
  xgx_scale_y_percentchangelog10() +
  guides(color = guide_legend(title = "Dose (mg)")) +
  ylab("Percent Change from Baseline")


## ---- fig.height=7------------------------------------------------------------
data <- data.frame(x = 1:1000, y = stats::rnorm(1000))
g <- xgx_plot(data = data, aes(x = x, y = y)) + 
  geom_point()
g1 <- g + xgx_scale_x_time_units(units_dataset = "hours", units_plot = "hours")
g2 <- g + xgx_scale_x_time_units(units_dataset = "hours", units_plot = "days")
g3 <- g + xgx_scale_x_time_units(units_dataset = "hours", units_plot = "weeks")
g4 <- g + xgx_scale_x_time_units(units_dataset = "hours", units_plot = "months")

gridExtra::grid.arrange(g1, g2, g3, g4, nrow = 2)

## ---- message=FALSE-----------------------------------------------------------
data <- mad_missing_duplicates %>%
  filter(CMT %in% c(1, 2, 3)) %>%
  rename(DV      = LIDV,
         YTYPE   = CMT,
         USUBJID = ID)
covariates <- c("WEIGHTB", "SEX")
check <- xgx_check_data(data, covariates)

knitr::kable(check$summary)
knitr::kable(head(check$data_subset))

## -----------------------------------------------------------------------------
covar <- xgx_summarize_covariates(data,covariates)
knitr::kable(covar$cts_covariates)
knitr::kable(covar$cat_covariates)

