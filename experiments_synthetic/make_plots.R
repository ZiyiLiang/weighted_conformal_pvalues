options(width=160)

library(tidyverse)

plot.1 <- FALSE
plot.2 <- FALSE
plot.3 <- FALSE
plot.4 <- FALSE
plot.5 <- FALSE
plot.6 <- FALSE
plot.7 <- FALSE
plot.8 <- FALSE
plot.9 <- FALSE
plot.10 <- TRUE

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

    idir <- "results_hpc/setup_power1/"
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

    z.lab <- parse(text=latex2exp::TeX("$\\frac{E( \\hat{u}_1(X) \\,|\\, Y=0 )}{1/\\log(n_1+1)}$"))
    
    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="Z", Mean=1) %>%
        mutate(Metric = factor(Metric, c("Power", "Z"), c("Power", z.lab)))

    results.fdr.models <- results %>%
        mutate(Z=E_U1_Y0 / `1/log(n1+1)`) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha, Gamma) %>%
        summarise(Z.se=2*sd(Z)/sqrt(n()), Z=mean(Z),
                  Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    df.mean <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        gather(Power, Z, key="Metric", value="Mean") %>%
        select(-TypeI, -TypeI.se)
    df.se <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        gather(Power.se, Z.se, key="Metric", value="SE") %>%
        select(-Z, -Power, -TypeI, -TypeI.se) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="Z.se", "Z", Metric))
    df <- inner_join(df.mean, df.se) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))

    pp <- df %>%
        filter(Data=="circles-mixed", n==200, Signal==0.7, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Metric = factor(Metric, c("Power", "Z"), c("Power", z.lab))) %>%
        ggplot(aes(x=Gamma, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity, scales="free", labeller=label_parsed) +
        scale_x_log10() +
#        scale_y_continuous(lim=c(0,1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("SVM gamma") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_power_1_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.75, height=3, units="in")
    

}

######################
## Setup 6 (greedy) ##
######################


if(plot.6) {

    idir <- "results_hpc/setup_greedy1/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (naive)", "Binary (naive)")
    color.scale <- c("darkviolet", "red", "blue", "darkgreen", "green")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 0.5)

    plot.fdr <- FALSE

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(Discoveries=`Storey-BH-Rejections`, TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Discoveries", "Power", "TypeI")
        metric.labels <- c("Discoveries", "Power", "FDR")
    } else {
        results <- results.raw %>%
            mutate(Discoveries=`Fixed-Rejections`, TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Discoveries", "Power", "TypeI")
        metric.labels <- c("Discoveries", "TPR", "FPR")
    }

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, `Num-models`, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.greedy <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha, `Num-models`, Seed, Repetition) %>%
        summarise(idx.greedy = which.max(Discoveries), Model="Greedy", Power=Power[idx.greedy], TypeI=TypeI[idx.greedy]) %>%
        select(-idx.greedy) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha, `Num-models`) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble")) %>%
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
        filter(Data=="circles-mixed", n==1000, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=`Num-models`, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity, scales="free") +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of models") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_greedy_1_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")

}


###########################
## Setup 7 (correlation) ##
###########################


if(plot.7) {

    idir <- "results_hpc/setup_corr1/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    key.values <- c("Integrative", "OCC", "OCC-Theory")
    key.labels <- c("Integrative", "OCC", "OCC(Theory)")
    color.scale <- c("darkviolet", "deeppink", "black")
    shape.scale <- c(8, 17, 15, 3, 1, 1)
    alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)
    linetype.scale <- c(1,2,3)

    df <- results.raw %>%
        group_by(Setup, Data, n, p, Signal, Purity) %>%
        summarise(Integrative=mean(Integrative), OCC=mean(OCC), `OCC-Theory`=mean(`OCC-Theory`)) %>%
        gather(Integrative, OCC, `OCC-Theory`, key="Method", value="Value")
    
    pp <- df %>%
        mutate(Data = factor(Data, c("circles-mixed", "binomial"), c("Data distribution 1", "Data distribution 2"))) %>%
        filter(n>10) %>%
        filter(Method %in% key.values) %>%
        mutate(Method = factor(Method, key.values, key.labels)) %>%
        mutate(n=n/4) %>%
        ggplot(aes(x=n, y=Value, color=Method, linetype=Method)) +
#        geom_point() +
        geom_line() +
        scale_x_log10() +
        scale_y_log10() +
        facet_grid(.~Data) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Calibration set size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_corr_1.pdf"), width=5.5, height=2.5, units="in")

}

###################
## Setup 8 (CV+) ##
###################


if(plot.8) {

    idir <- "results_hpc/setup_cv1/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

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
    
    method.values <- c("Ensemble (CV+)", "Ensemble (split)", "One-Class (CV+)", "Binary (CV+)")
    method.labels <- c("Integrative (TCV+)", "Integrative", "One-Class (oracle, CV+)", "Binary (oracle, CV+)")
    color.scale <- c("#da00da", "darkviolet", "red", "blue", "darkgreen", "green")
    shape.scale <- c(5, 8, 3, 1, 1)
    alpha.scale <- c(1, 1, 0.5, 0.5)

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary (CV+)", "One-Class (CV+)")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)
  
    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble (CV+)", "Ensemble (split)")) %>%
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
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_cv_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")

}


if(plot.9) {

    idir <- "results_hpc/setup_cv2/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

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
    
    method.values <- c("Ensemble (CV+)", "Ensemble (split)", "One-Class (CV+)", "Binary (CV+)")
    method.labels <- c("Integrative (TCV+)", "Integrative", "One-Class (oracle, CV+)", "Binary (oracle, CV+)")
    color.scale <- c("#da00da", "darkviolet", "red", "blue", "darkgreen", "green")
    shape.scale <- c(5, 8, 3, 1, 1)
    alpha.scale <- c(1, 1, 0.5, 0.5)

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="TypeI", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary (CV+)", "One-Class (CV+)")) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Setup, Data, n, p, Signal, Purity, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)
  
    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble (CV+)", "Ensemble (split)")) %>%
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
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_binomial_cv_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.5, height=3, units="in")

}

if(plot.10) {

    idir <- "results_hpc/setup_fdr1/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    }))

    results <- results.raw %>%
        group_by(Setup, Data, p, Signal, Purity, Alpha, n, Method) %>%
        summarise(FDR.se=2*sd(FDP)/sqrt(n()), Power.se=2*sd(Power)/sqrt(n()), FDR=mean(FDP), Power=mean(Power))
    
    ## method.values <- c("Ensemble", "Ensemble (one-class, unweighted)", "Ensemble (binary, unweighted)",  "One-Class", "Binary")
    ## method.labels <- c("Integrative", "OCC (ensemble)", "Binary (ensemble)", "OCC (oracle)", "Binary (oracle)")
    ## color.scale <- c("darkviolet", "deeppink", "slateblue", "red", "blue", "darkgreen", "green")
    ## shape.scale <- c(8, 17, 15, 3, 1, 1)
    ## alpha.scale <- c(1, 0.5, 1, 0.75, 0.75)

    ## plot.fdr <- FALSE

    ## if(plot.fdr) {
    ##     results <- results.raw %>%
    ##         mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
    ## } else {
    ##     results <- results.raw %>%
    ##         mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
    ##     metric.values <- c("Power", "TypeI")
    ##     metric.labels <- c("TPR", "FPR")
    ## }

    metric.values <- c("Power", "FDR")
    metric.labels <- c("Power", "FDR")

    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="FDR", Mean=alpha.nominal) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))
  
    df.mean <- results %>%
        gather(Power, FDR, key="Metric", value="Mean") %>%
        select(-Power.se, -FDR.se)
    df.se <- results %>%
        gather(Power.se, FDR.se, key="Metric", value="SE") %>%
        select(-Power, -FDR) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="FDR.se", "FDR", Metric))
    df <- df.mean %>% inner_join(df.se) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels)) %>%
        mutate(Purity = sprintf("Inliers: %.2f", Purity))


    pp <- df %>%
        filter(Data=="circles-mixed", Alpha==alpha.nominal) %>%
#        filter(Method %in% method.values) %>%
#        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n, y=Mean, color=Method, shape=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Purity) +
        scale_x_log10(breaks=c(30, 300, 3000)) +
    #    scale_color_manual(values=color.scale) +
    #    scale_shape_manual(values=shape.scale) +
    #    scale_alpha_manual(values=alpha.scale) +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file="figures/experiment_fdr_1.pdf", width=6.5, height=3, units="in")
}
