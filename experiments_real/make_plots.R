options(width=160)

library(tidyverse)

plot.1 <- TRUE

#############
## Setup 1 ##
#############
 
if(plot.1) {

    idir <- "results_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "cifar-100_")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
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
        group_by(Data, n_in, n_out, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Data, n_in, n_out, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Data, n_in, n_out, Method, Alpha) %>%
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
        mutate(Metric = factor(Metric, metric.values, metric.labels))

    pp <- results.fdr %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Metric %in% c("TPR", "Power")) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
#        facet_grid(Metric~Data) +
        facet_grid(Data~n_in, scales="free") +
        scale_x_log10() +#breaks=c(30, 300, 3000)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real.pdf", ifelse(plot.fdr, "bh", "fixed")), width=7, height=4, units="in")

}
