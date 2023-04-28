options(width=160)

library(tidyverse)
library(patchwork)

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
    xi.lab <- parse(text=latex2exp::TeX("True ($\\Xi$)"))
    xi.hat.lab <- parse(text=latex2exp::TeX("Estimated ($\\hat{\\Xi}$), exch."))
    xi.hat.ne.lab <- parse(text=latex2exp::TeX("Estimated ($\\hat{\\Xi}$), non-exch."))

    if(plot.fdr) {
        results <- results.raw %>%
            mutate(TypeI=`Storey-BH-FDP`, Power=`Storey-BH-Power`)
        metric.values <- c("Power", "TypeI", "Xi", "Xi-hat", "Xi-hat-ne")
        metric.labels <- c("Power", "FDR", xi.lab, xi.hat.lab, xi.hat.ne.lab)
    } else {
        results <- results.raw %>%
            mutate(TypeI=`Fixed-FPR`, Power=`Fixed-TPR`)
        metric.values <- c("Power", "TypeI", "Xi", "Xi-hat", "Xi-hat-ne")
        metric.labels <- c("TPR", "FPR", xi.lab, xi.hat.lab, xi.hat.ne.lab)
    }
   
    alpha.nominal <- 0.1

    results.fdr.models <- results %>%
        group_by(Setup, Data, n, p, Signal, Shift, Purity, Method, Model, Alpha, Gamma) %>%
        summarise(Xi=mean(xi), Xi.se=2*sd(xi)/sqrt(n()), Xi.hat=mean(`xi-2-hat`), Xi.hat.se=2*sd(`xi-2-hat`)/sqrt(n()),
                  Xi.hat.ne=mean(`xi-3-hat`), Xi.hat.ne.se=2*sd(`xi-3-hat`)/sqrt(n()),
                  Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))
    
    df.mean <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        gather(Power, TypeI, Xi, Xi.hat, Xi.hat.ne, key="Metric", value="Mean") %>%
        select(-Power.se, -TypeI.se, -Xi.hat.se, -Xi.hat.ne.se)
    df.se <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
        gather(Power.se, TypeI.se, Xi.se, Xi.hat.se, Xi.hat.ne.se, key="Metric", value="SE") %>%
        select(-Power, -Xi, -Xi.hat, -Xi.hat.ne, -TypeI) %>%
        mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
               Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric),
               Metric = ifelse(Metric=="Xi.se", "Xi", Metric),
               Metric = ifelse(Metric=="Xi.hat.se", "Xi.hat", Metric),
               Metric = ifelse(Metric=="Xi.hat.ne.se", "Xi.hat.ne", Metric))


#    purity.labs <- c(parse(text=latex2exp::TeX("$n_1=50$")), parse(text=latex2exp::TeX("$n_1=25$")), parse(text=latex2exp::TeX("$n_1=10$")))
    df <- inner_join(df.mean, df.se) %>%
        mutate(Shift.lab = sprintf("Shift: %.2f", Shift)) %>%
        mutate(Shift.lab = factor(Shift.lab, sprintf("Shift: %.2f", unique(df$Shift))))
#        mutate(Purity = sprintf("Inliers: %.2f", Purity))
#        mutate(Purity = factor(Purity, c(0.5,0.75,0.9), purity.labs))

    df.nominal <- tibble(Metric=c("TypeI"), Mean=c(alpha.nominal)) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))
    pp.1 <- df %>%
        filter(Data=="circles-mixed", n==200, Signal==0.7, Purity==0.5, abs(Shift)<0.5, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        filter(Metric %in% c("Power", "TypeI")) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Metric = factor(Metric, c("TypeI", "Power"), c("FDR", "Power"))) %>%
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
        theme(legend.position="right", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.key.width = unit(1,"cm"))

    df.nominal <- tibble(Metric=c("Xi"), Mean=c(1)) %>%
        mutate(Metric = factor(Metric, metric.values, metric.labels))
    pp.2 <- df %>%
        filter(Data=="circles-mixed", n==200, Signal==0.7, p==1000, Alpha==alpha.nominal) %>%
        filter(Method %in% c("Ensemble")) %>%
        filter( Metric %in% c("Xi", "Xi.hat", "Xi.hat.ne")) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Metric = factor(Metric, c("Xi", "Xi.hat", "Xi.hat.ne"), c(xi.lab, xi.hat.lab, xi.hat.ne.lab))) %>%
        ggplot(aes(x=Gamma, y=Mean, linetype=Metric, shape=Metric)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~Shift.lab, scales="free") +
        labs(shape = "Informativeness ratio", linetype = "Informativeness ratio") + 
        scale_x_log10() +
        scale_y_sqrt() +
        scale_shape_manual(values=c(8, 19, 11), labels = c(unname(latex2exp::TeX(c("True ($\\Xi$)"))),
                                                       unname(latex2exp::TeX(c("Estimated ($\\hat{\\Xi}$)"))),
                                                       unname(latex2exp::TeX(c("Estimated ($\\hat{\\Xi}$), non-exch.")))
                                                       )) +
        scale_linetype_manual(values=c(1,2,3), labels = c(unname(latex2exp::TeX(c("True ($\\Xi$)"))),
                                                        unname(latex2exp::TeX(c("Estimated ($\\hat{\\Xi}$)"))),
                                                        unname(latex2exp::TeX(c("Estimated ($\\hat{\\Xi}$), non-exch.")))
                                                        )) +
    xlab("SVM gamma") +
        ylab("") +
        theme_bw() +
        theme(legend.position="right", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.key.width = unit(1,"cm"))
    

    pp <- pp.1+pp.2 + plot_layout(ncol = 1, heights=c(1.8,1))      
    pp %>% ggsave(file=sprintf("figures/experiment_power_shift_%s.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6.75, height=5, units="in")   
}
