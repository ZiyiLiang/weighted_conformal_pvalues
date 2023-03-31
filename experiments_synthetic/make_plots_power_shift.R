options(width=160)

library(tidyverse)

plot.1 <- TRUE

#############
## Setup 1 ##
#############


if(plot.1) {

    idir <- "results_hpc/setup_power_shift/"
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

    z.lab <- parse(text=latex2exp::TeX("Informativeness ($\\hat{\\Xi}$)"))
    
    alpha.nominal <- 0.1
    df.nominal <- tibble(Metric="Z", Mean=1) %>%
        mutate(Metric = factor(Metric, c("Power", "Z"), c("Power", z.lab)))

    results.fdr.models <- results %>%
        mutate(Z=`informativeness`/E_U1_Y0_approx) %>%
        group_by(Setup, Data, n, p, Signal, Shift, Purity, Method, Model, Alpha, Gamma) %>%
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


#    purity.labs <- c(parse(text=latex2exp::TeX("$n_1=50$")), parse(text=latex2exp::TeX("$n_1=25$")), parse(text=latex2exp::TeX("$n_1=10$")))
    df <- inner_join(df.mean, df.se) %>%
        mutate(Shift.lab = sprintf("Outlier shift: %.2f", Shift)) %>%
        mutate(Shift.lab = factor(Shift.lab, sprintf("Outlier shift: %.2f", unique(df$Shift))))
#        mutate(Purity = sprintf("Inliers: %.2f", Purity))
#        mutate(Purity = factor(Purity, c(0.5,0.75,0.9), purity.labs))

    pp <- df %>%
        filter(Data=="circles-mixed", n==200, Signal==0.7, Purity==0.5, abs(Shift)<0.5, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Metric = factor(Metric, c("Power", "Z"), c("Power", z.lab))) %>%
        ggplot(aes(x=Gamma, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Shift.lab, scales="free", labeller= labeller(Metric = label_parsed, `Shift.lab` = label_value)) +
        scale_x_log10(breaks=c(1e-6,1e-3,1)) +
#        scale_y_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("SVM gamma") +
        ylab("") +
        theme_bw() +
        theme(legend.position="bottom", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
    pp %>% ggsave(file=sprintf("figures/experiment_power_shift_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.75, height=4, units="in")


    
    results.fdr.models <- results %>%
        mutate(Z=`1/log(n1+1)`/E_U1_Y0_approx) %>%
        group_by(Setup, Data, n, p, Signal, Shift, Purity, Method, Model, Alpha, Gamma) %>%
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


#    purity.labs <- c(parse(text=latex2exp::TeX("$n_1=50$")), parse(text=latex2exp::TeX("$n_1=25$")), parse(text=latex2exp::TeX("$n_1=10$")))
    df <- inner_join(df.mean, df.se) %>%
        mutate(Shift.lab = sprintf("Outlier shift: %.2f", Shift)) %>%
        mutate(Shift.lab = factor(Shift.lab, sprintf("Outlier shift: %.2f", unique(df$Shift))))
#        mutate(Purity = sprintf("Inliers: %.2f", Purity))
#        mutate(Purity = factor(Purity, c(0.5,0.75,0.9), purity.labs))

    pp <- df %>%
        filter(Data=="circles-mixed", n==200, Signal==0.7, Purity==0.5, abs(Shift)<0.5, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Metric = factor(Metric, c("Power", "Z"), c("Power", z.lab))) %>%
        ggplot(aes(x=Gamma, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~Shift.lab, scales="free", labeller= labeller(Metric = label_parsed, `Shift.lab` = label_value)) +
        scale_x_log10(breaks=c(1e-6,1e-3,1)) +
#        scale_y_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("SVM gamma") +
        ylab("") +
        theme_bw() +
        theme(legend.position="bottom", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
    pp %>% ggsave(file=sprintf("figures/experiment_power_shift_noshift_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.75, height=4, units="in")


}
