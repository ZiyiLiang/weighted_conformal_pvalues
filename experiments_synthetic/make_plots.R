options(width=160)

library(tidyverse)

plot.1 <- FALSE
plot.2 <- FALSE
plot.3 <- FALSE
plot.4 <- FALSE
plot.5 <- TRUE
plot.6 <- FALSE


#############
## Setup 1 ##
#############

if(plot.1) {

    idir <- "results_hpc/setup1/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    plot.fdr <- FALSE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        rbind(results.fdr.oracle)
    results.fdr.mean <- df %>%
        gather(Power, TypeI, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se)
    results.fdr.se <- df %>%
        gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
        select(-Power, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    results.fdr <- results.fdr.mean %>% inner_join(results.fdr.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))

    pp <- results.fdr %>%
        filter(Data=="circles-mixed", n>10, p==1000, Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10(breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_1_n_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")

}

#############
## Setup 2 ##
#############

if(plot.2) {

    idir <- "results_hpc/setup2/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    plot.fdr <- TRUE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))


    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        rbind(results.fdr.oracle)
    results.fdr.mean <- df %>%
        gather(Power, TypeI, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se)
    results.fdr.se <- df %>%
        gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
        select(-Power, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    results.fdr <- results.fdr.mean %>% inner_join(results.fdr.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(n=sprintf("n = %d", n))

    pp.2 <- results.fdr %>%
        filter(Data=="circles-mixed", n>10, p==1000, Purity<1, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=1-Purity, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
                                        #    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.01) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~n) +
        scale_x_continuous(breaks=c(0,0.25,0.5), labels = scales::number_format(accuracy = 0.01)) +
        scale_y_continuous(lim=c(0,0.85), breaks=c(0,0.25,0.5,0.75), labels = scales::number_format(accuracy = 0.1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Fraction of outliers in the data") +
        ylab("") +
        theme_bw()
    pp.2 %>% ggsave(file=sprintf("figures/experiment_2_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")
}

#############
## Setup 3 ##
#############

if(plot.3) {

    idir <- "results_hpc/setup3/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    plot.fdr <- TRUE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        rbind(results.fdr.oracle)
    results.fdr.mean <- df %>%
        gather(Power, TypeI, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se)
    results.fdr.se <- df %>%
        gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
        select(-Power, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    results.fdr <- results.fdr.mean %>% inner_join(results.fdr.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))

    pp <- results.fdr %>%
        filter(Data=="circles-mixed", n>10, p==1000, Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
                                        #    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10(breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_3_n_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")


#############################
    ## Setup 3: oracle details ##
#############################

    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble (one-class, unweighted)", "One-Class"))
    df.mean <- df %>%
        gather(Power, TypeI, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se)
    df.se <- df %>%
        gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
        select(-Power, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    df <- inner_join(df.mean, df.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    model.values <- c("Oracle", "Ensemble", "IF", "LOF", "SVM")
    model.labels <- c("Oracle", "Ensemble", "IF", "LOF", "SVM")
    color.scale <- c("red", "deeppink", "orange", "orange", "orange", "orange")
    shape.scale <- c(3, 17, 3, 3, 3)
    linetype.scale <- c(1, 1, 3, 4, 5)
    pp.1 <- df %>%
        filter(Data=="circles-mixed", Purity==0.5, n>10, p==1000, Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(OCC=Model) %>%
        ggplot(aes(x=n, y=Mean, color=OCC, shape=OCC, linetype=OCC)) +
        geom_point() +
        geom_line() +
                                        #    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Purity~Metric) +
        scale_x_log10(breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"))
    pp.1 %>% ggsave(file=sprintf("figures/experiment_3_n_oracle_binary_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5, height=2, units="in")

}


## model.values <- c("Oracle", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
## model.labels <- c("Oracle", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
## color.scale <- c("blue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue")
## shape.scale <- c(1, 1, 1, 1, 1, 1, 1, 1)
## alpha.scale <- c(1, 1, 1, 1, 1, 1, 1, 1)


#############
## Setup 4 ##
#############

if(plot.4) {

    idir <- "results_hpc/setup4/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    plot.fdr <- TRUE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        rbind(results.fdr.oracle)
    results.fdr.mean <- df %>%
        gather(Power, TypeI, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se)
    results.fdr.se <- df %>%
        gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
        select(-Power, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    results.fdr <- results.fdr.mean %>% inner_join(results.fdr.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))

    pp <- results.fdr %>%
        filter(Data=="binomial", n>10, p==100, Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10(breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_4_n_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")

}


#####################
## Setup 5 (power) ##
#####################

if(plot.5) {

    idir <- "results_hpc/setup5/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    plot.fdr <- FALSE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="Z", Mean=1)# %>%
#        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        mutate(Z=E_U1_Y0/`1/log(n1+1)`) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(E_U1_Y0=mean(E_U1_Y0), RHS=mean(`1/log(n1+1)`), Z=mean(Z),
                  Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        gather(Power, Z, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI, -TypeI.se, E_U1_Y0, -RHS) %>%
        #mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))

    pp <- df %>%
        filter(Data=="circles-mixed", n>10, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=Signal, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
#        scale_x_log10(breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Signal") +
        ylab("") +
        theme_bw()
    pp

}

######################
## Setup 6 (greedy) ##
######################


if(plot.6) {

    idir <- "results_hpc/setup4/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    plot.fdr <- TRUE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI")
        metric.labels <- c("TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    results.fdr.greedy <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha, Seed, Repetition) %>%
        summarise(idx.greedy = which.max(Power), Model="Greedy", Power=Power[idx.greedy], TypeI=TypeI[idx.greedy]) %>%
        select(-idx.greedy) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        rbind(results.fdr.greedy)
    results.fdr.mean <- df %>%
        gather(Power, TypeI, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se)
    results.fdr.se <- df %>%
        gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
        select(-Power, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    results.fdr <- results.fdr.mean %>% inner_join(results.fdr.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))

    pp <- results.fdr %>%
        filter(Data=="binomial", n>10, p==100, Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10(breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp

}
