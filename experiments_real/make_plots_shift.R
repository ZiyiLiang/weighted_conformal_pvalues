options(width=160)

library(tidyverse)

plot.2 <- TRUE  # Animals (shift)


idir <- "results_shift_hpc/"
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
    if(startsWith(ifile, "images_animals")) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
    } else {
        df <- tibble()
    }
}))


make.plot.shift <- function(n_in.plot, outlier.group, y.max=0.75) {
    method.values <- c("Ensemble", "One-Class", "Binary")
    method.labels <- c("Integrative", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "red", "blue")
    shape.scale <- c(8, 3, 1, 1)
    alpha.scale <- c(1, 0.75, 0.75)

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
        group_by(Data, n_in, n_out, shift, shift_group, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class")) %>%
        group_by(Data, n_in, n_out, shift, shift_group, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Data, n_in, n_out, shift, shift_group, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble", "Ensemble (one-class)", "Ensemble (one-class, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
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

    n.out.vals <- sort(unique(results.fdr$n_out))
    n.out.labels <- sprintf("Outliers: %d", n.out.vals)

    df.nominal <- tibble(Mean=0.1, Metric="FDR", n_out=n.out.vals) %>%
        mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
        mutate(n_out = factor(n_out, n.out.labels)) %>%
        mutate(Metric = factor(Metric, c("Power", "FDR")))

    df.limits <- tibble(Mean=c(0.2), Metric="FDR", n_out=n.out.vals) %>%
        rbind(tibble(Mean=c(0.75), Metric="Power", n_out=n.out.vals)) %>%
        mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
        mutate(n_out = factor(n_out, n.out.labels)) %>%
        mutate(Metric = factor(Metric, c("Power", "FDR")))

    
    pp <- results.fdr %>%
        filter(n_in==n_in.plot, shift_group==outlier.group) %>%
        filter(Data=="images_animals", Metric %in% c("Power", "FDR")) %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(n_in = sprintf("Inliers: %d", n_in)) %>% 
        mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
        mutate(n_out = factor(n_out, n.out.labels)) %>%
        mutate(Metric = factor(Metric, c("Power", "FDR"))) %>%
        ggplot(aes(x=shift, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(Metric~n_out, scale="free_y") +
        scale_x_continuous(lim=c(0,1), breaks=c(0,0.5,1)) +
        scale_y_continuous(lim=c(0,y.max)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Outlier shift") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_animals_shift_%s_nin_%s.pdf", outlier.group, n_in.plot), width=6.5, height=3, units="in")
}

make.plot.shift(1000, 1, y.max=0.6)
make.plot.shift(1000, 2, y.max=0.6)
make.plot.shift(1000, 3, y.max=0.6)

make.plot.shift(7824, 1, y.max=0.75)
make.plot.shift(7824, 2, y.max=0.75)
make.plot.shift(7824, 3, y.max=0.75)

    

make.plot.shift.single <- function(n_in.plot, outlier.group, y.max=0.75) {

method.values <- c("Integrative (one-class)", "Ensemble", "One-Class", "Binary")
method.labels <- c("Integrative (OCC)", "Integrative", "OCC (oracle)", "Binary (oracle)")
color.scale <- c("violet", "darkviolet", "red", "blue")
shape.scale <- c(7, 8, 3, 1, 1)
alpha.scale <- c(1, 1, 0.75, 0.75)

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
    group_by(Data, n_in, n_out, shift, shift_group, Method, Model, Alpha) %>%
    summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

results.fdr.oracle <- results %>%
    filter(Method %in% c("Binary", "One-Class")) %>%
    group_by(Data, n_in, n_out, shift, shift_group, Method, Model, Alpha) %>%
    summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
    group_by(Data, n_in, n_out, shift, shift_group, Method, Alpha) %>%
    summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
              TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
    select(-idx.oracle)

df <- results.fdr.models %>%
    filter(Method %in% c("Integrative (one-class)", "Ensemble", "Ensemble (one-class)", "Ensemble (one-class, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
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

n.out.vals <- sort(unique(results.fdr$n_out))
n.out.labels <- sprintf("Outliers: %d", n.out.vals)

df.nominal <- tibble(Mean=0.1, Metric="FDR", n_out=n.out.vals) %>%
    mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
    mutate(n_out = factor(n_out, n.out.labels)) %>%
    mutate(Metric = factor(Metric, c("Power", "FDR")))

df.limits <- tibble(Mean=c(0.2), Metric="FDR", n_out=n.out.vals) %>%
    rbind(tibble(Mean=c(0.75), Metric="Power", n_out=n.out.vals)) %>%
    mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
    mutate(n_out = factor(n_out, n.out.labels)) %>%
    mutate(Metric = factor(Metric, c("Power", "FDR")))


pp <- results.fdr %>%
    filter(n_in==n_in.plot, shift_group==outlier.group) %>%
    filter(Data=="images_animals", Metric %in% c("Power", "FDR")) %>%
    filter(Alpha==alpha.nominal) %>%
    filter(Method %in% method.values) %>%
    mutate(Method = factor(Method, method.values, method.labels)) %>%
    mutate(n_in = sprintf("Inliers: %d", n_in)) %>% 
    mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
    mutate(n_out = factor(n_out, n.out.labels)) %>%
    mutate(Metric = factor(Metric, c("Power", "FDR"))) %>%
    ggplot(aes(x=shift, y=Mean, color=Method, shape=Method, alpha=Method)) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
    geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
    facet_grid(Metric~n_out, scale="free_y") +
    scale_x_continuous(lim=c(0,1), breaks=c(0,0.5,1)) +
    scale_y_continuous(lim=c(0,y.max)) +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    scale_alpha_manual(values=alpha.scale) +
    xlab("Outlier shift") +
    ylab("") +
    theme_bw()

    pp %>% ggsave(file=sprintf("figures/experiment_real_animals_shift_%s_nin_%s_single.pdf", outlier.group, n_in.plot), width=6.5, height=3, units="in")

}

make.plot.shift.single(7824, 3, y.max=0.75)


## n_in.plot=7824
## outlier.group = 3
## y.max=0.75


## pp



## df <- results.fdr.models %>%
##     rbind(results.fdr.oracle) %>%
##     filter(Method %in% c("Ensemble", "Binary"))
## df.mean <- df %>%
##     gather(Power, TypeI, key="Metric", value="Mean") %>%
##     select(-Power.se, -TypeI.se)
## df.se <- df %>%
##     gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
##     select(-Power, -TypeI) %>%
##     mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
##            Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
## df <- inner_join(df.mean, df.se) %>%
##     mutate(Metric = factor(Metric, metric.values, metric.labels))

## model.values <- c("Ensemble", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
## model.labels <- c("Integrative", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
## color.scale <- c("darkviolet", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue")
## shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
## linetype.scale <- c(1, 2, 4, 2, 2, 3, 5, 2) 
## alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)


## df <- results.fdr.models %>%
##     rbind(results.fdr.oracle) %>%
##     filter(Method %in% c("Ensemble", "One-Class"))
## df.mean <- df %>%
##     gather(Power, TypeI, key="Metric", value="Mean") %>%
##     select(-Power.se, -TypeI.se)
## df.se <- df %>%
##     gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
##     select(-Power, -TypeI) %>%
##     mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
##            Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
## df <- inner_join(df.mean, df.se) %>%
##     mutate(Metric = factor(Metric, metric.values, metric.labels))

## model.values <- c("Ensemble", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
## model.labels <- c("Integrative", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
## color.scale <- c("darkviolet", "orange", "orange", "orange", "orange", "orange", "orange", "orange")
## shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
## linetype.scale <- c(1, 2, 2, 2, 2, 3, 2, 2)
## alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)


## pp.1 <- df %>%
##     filter(n_in==n_in.plot, shift_group==outlier.group) %>%
##     filter(Alpha==alpha.nominal, Metric=="Power") %>%
##     filter(Metric=="Power") %>%
##     filter(Model %in% model.values) %>%
##     mutate(Model = factor(Model, model.values, model.labels)) %>%
##     mutate(OCC=Model) %>% 
##     mutate(n_in = sprintf("Inliers: %d", n_in)) %>% 
##     mutate(n_out = sprintf("Outliers: %d", n_out)) %>%
##     mutate(n_out = factor(n_out, n.out.labels)) %>%
##     mutate(Metric = factor(Metric, c("Power", "FDR"))) %>%
##     ggplot(aes(x=shift, y=Mean, color=OCC, shape=OCC, linetype=OCC, alpha=OCC)) +
##     geom_point() +
##     geom_line() +
##     ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
##     geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
##     facet_grid(Metric~n_out, scale="free_y") +
##     scale_x_log10() +
##     scale_color_manual(values=color.scale) +
##     scale_shape_manual(values=shape.scale) +
##     scale_alpha_manual(values=alpha.scale) +
##     scale_linetype_manual(values=linetype.scale) +
##     xlab("Number of outliers") +        
##     ylab("Power") +
##     ggtitle("(a)") +
##     theme_bw() +
##     theme(legend.key.width = unit(1,"cm"),
##           plot.title = element_text(hjust = -0.11, vjust = -4))
## pp.1



##pp %>% ggsave(file=sprintf("figures/experiment_real_animals.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")


    
    ## df <- results.fdr.models %>%
    ##     rbind(results.fdr.oracle) %>%
    ##     filter(Method %in% c("Ensemble", "One-Class"))
    ## df.mean <- df %>%
    ##     gather(Power, TypeI, key="Metric", value="Mean") %>%
    ##     select(-Power.se, -TypeI.se)
    ## df.se <- df %>%
    ##     gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
    ##     select(-Power, -TypeI) %>%
    ##     mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
    ##            Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    ## df <- inner_join(df.mean, df.se) %>%
    ##     mutate(Metric = factor(Metric, metric.values, metric.labels))
    
    ## model.values <- c("Ensemble", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    ## model.labels <- c("Integrative", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    ## color.scale <- c("darkviolet", "orange", "orange", "orange", "orange", "orange", "orange", "orange")
    ## shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    ## linetype.scale <- c(1, 2, 2, 2, 2, 3, 2, 2)
    ## alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
   
    ## pp.1 <- df %>%
    ##     filter(Alpha==alpha.nominal, Metric=="Power") %>%
    ##     filter(Metric=="Power") %>%
    ##     filter(Model %in% model.values) %>%
    ##     mutate(Model = factor(Model, model.values, model.labels)) %>%
    ##     mutate(OCC=Model) %>% 
    ##     mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
    ##     ggplot(aes(x=n_out, y=Mean, color=OCC, shape=OCC, linetype=OCC, alpha=OCC)) +
    ##     geom_point() +
    ##     geom_line() +
    ##     ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
    ##     geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
    ##     facet_grid(.~n_in) +
    ##     scale_x_log10() +
    ##     scale_color_manual(values=color.scale) +
    ##     scale_shape_manual(values=shape.scale) +
    ##     scale_alpha_manual(values=alpha.scale) +
    ##     scale_linetype_manual(values=linetype.scale) +
    ##     xlab("Number of outliers") +        
    ##     ylab("Power") +
    ##     ggtitle("(a)") +
    ##     theme_bw() +
    ##     theme(legend.key.width = unit(1,"cm"),
    ##           plot.title = element_text(hjust = -0.11, vjust = -4))
    ## pp.1 %>% ggsave(file=sprintf("figures/experiment_real_animals_oracle_occ.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")
    

    ## df <- results.fdr.models %>%
    ##     rbind(results.fdr.oracle) %>%
    ##     filter(Method %in% c("Ensemble", "Binary"))
    ## df.mean <- df %>%
    ##     gather(Power, TypeI, key="Metric", value="Mean") %>%
    ##     select(-Power.se, -TypeI.se)
    ## df.se <- df %>%
    ##     gather(Power.se, TypeI.se, key="Metric", value="SE") %>%
    ##     select(-Power, -TypeI) %>%
    ##     mutate(Metric = ifelse(Metric=="Power.se", "Power", Metric),
    ##            Metric = ifelse(Metric=="TypeI.se", "TypeI", Metric))
    ## df <- inner_join(df.mean, df.se) %>%
    ##     mutate(Metric = factor(Metric, metric.values, metric.labels))
    
    ## model.values <- c("Ensemble", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    ## model.labels <- c("Integrative", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    ## color.scale <- c("darkviolet", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue")
    ## shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    ## linetype.scale <- c(1, 2, 4, 2, 2, 3, 5, 2) 
    ## alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
   
    ## pp.2 <- df %>%
    ##     filter(Alpha==alpha.nominal) %>%
    ##     filter(Metric=="Power") %>%
    ##     filter(Model %in% model.values) %>%
    ##     mutate(Model = factor(Model, model.values, model.labels)) %>%
    ##     mutate(BC=Model) %>%
    ##     mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
    ##     ggplot(aes(x=n_out, y=Mean, color=BC, shape=BC, linetype=BC, alpha=BC)) +
    ##     geom_point() +
    ##     geom_line() +
    ##     ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
    ##     geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
    ##     facet_grid(.~n_in) +
    ##     scale_x_log10() +
    ##     scale_color_manual(values=color.scale) +
    ##     scale_shape_manual(values=shape.scale) +
    ##     scale_alpha_manual(values=alpha.scale) +
    ##     scale_linetype_manual(values=linetype.scale) +
    ##     xlab("Number of outliers") +        
    ##     ylab("Power") +
    ##     ggtitle("(b)") +
    ##     theme_bw() +
    ##     theme(legend.key.width = unit(1,"cm"),
    ##           plot.title = element_text(hjust = -0.11, vjust = -4))
    ## pp.2 %>% ggsave(file=sprintf("figures/experiment_real_animals_oracle_bin.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")

#}
