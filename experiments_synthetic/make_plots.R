options(width=160)

library(tidyverse)


idir <- "results_hpc/setup1/"
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
    df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

plot.fdr <- TRUE

if(plot.fdr) {
    results <- results.raw %>%
        mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
} else {
    results <- results.raw %>%
        mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
}

alpha.nominal <- 0.1
df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal)

results.fdr.models <- results %>%
    group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
    summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

results.fdr.oracle <- results %>%
    filter(Method %in% c("Binary", "One-Class")) %>%
    group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
    summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
    group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
    summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle], TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
    select(-idx.oracle)

df <- results.fdr.models %>%
    filter(Method %in% c("Ensemble", "Ensemble (weighted)")) %>%
    rbind(results.fdr.oracle)
results.fdr.mean <- df %>%
    gather(Power, TypeI, key="Metric", value="Mean") %>%
    select(-Power.se, -TypeI.se)
results.fdr.se <- df %>%
    gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
    select(-Power, -TypeI) %>%
    mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
           Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
results.fdr <- results.fdr.mean %>% inner_join(results.fdr.se)


results.fdr %>%
    filter(p==1000, Metric=="Power", Alpha==alpha.nominal) %>%
    ggplot(aes(x=n, y=Mean, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line(alpha=0.75) +
    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
    geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
    facet_grid(Signal~Purity) +
    scale_x_log10() +
    xlab("Sample size") +
    ylab("") +
    theme_bw()
