options(width=160)

library(tidyverse)

plot.1 <- FALSE  # Flowers
plot.2 <- TRUE  # Animals
plot.3 <- FALSE  # Cars
plot.4 <- FALSE  # Tabular
plot.5 <- FALSE  # Flowers-CV
plot.6 <- FALSE  # Tabular-CV

#############
## Setup 1 ##
#############

if(plot.1) {
    idir <- "results_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "images_flowers")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
    }))

    method.values <- c("Ensemble (one-class)", "One-Class", "Binary")
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
        filter(Method %in% c("Ensemble", "Ensemble (one-class)", "Ensemble (one-class, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
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
        filter(Data=="images_flowers") %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_wrap(Metric~.) +
        scale_x_log10() +
        scale_y_continuous(lim=c(0,1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_flowers.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")

    
    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble", "One-Class"))
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
    
    model.values <- c("Ensemble", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    model.labels <- c("Integrative", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    color.scale <- c("darkviolet", "orange", "orange", "orange", "orange", "orange", "orange", "orange")
    shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    linetype.scale <- c(1, 2, 2, 2, 2, 3, 2, 2)
    alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)

    pp.1 <- df %>%
        filter(Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(OCC=Model) %>%
        ggplot(aes(x=n_out, y=Mean, color=OCC, shape=OCC, linetype=OCC, alpha=OCC)) +
        geom_point() +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~Metric) +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Number of outliers") +        
        ylab("") +
        ggtitle("(a)") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"),
              plot.title = element_text(hjust = -0.11, vjust = -4))
    pp.1 %>% ggsave(file=sprintf("figures/experiment_real_flowers_oracle_occ.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")
    

    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble", "Binary"))
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
    
    model.values <- c("Ensemble", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    model.labels <- c("Integrative", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    color.scale <- c("darkviolet", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue")
    shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    linetype.scale <- c(1, 2, 4, 2, 2, 3, 5, 2)
    
    pp.2 <- df %>%
        filter(Alpha==alpha.nominal) %>%
                                        #    filter(Metric=="Power") %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(BC=Model) %>%
        ggplot(aes(x=n_out, y=Mean, color=BC, shape=BC, linetype=BC, alpha=BC)) +
        geom_point() +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~Metric) +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Number of outliers") +        
        ylab("") +
        ggtitle("(b)") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"),
              plot.title = element_text(hjust = -0.11, vjust = -4))
    pp.2 %>% ggsave(file=sprintf("figures/experiment_real_flowers_oracle_bin.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")
}



if(plot.2) {
    idir <- "results_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "images_animals")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
    }))

    method.values <- c("Ensemble (one-class)", "One-Class", "Binary")
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
        filter(Method %in% c("Ensemble", "Ensemble (one-class)", "Ensemble (one-class, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
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
        filter(Data=="images_animals", Metric=="Power") %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_wrap(.~n_in) +
        scale_x_log10() +
        scale_y_continuous(lim=c(0,0.75), breaks=c(0,0.25,0.5,0.75)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("Power") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_animals.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")


    
    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble", "One-Class"))
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
    
    model.values <- c("Ensemble", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    model.labels <- c("Integrative", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    color.scale <- c("darkviolet", "orange", "orange", "orange", "orange", "orange", "orange", "orange")
    shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    linetype.scale <- c(1, 2, 3, 4, 5, 6, 7, 8) 
    alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
   
    pp.1 <- df %>%
        filter(Alpha==alpha.nominal, Metric=="Power") %>%
        filter(Metric=="Power") %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(OCC=Model) %>% 
        mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
        ggplot(aes(x=n_out, y=Mean, color=OCC, shape=OCC, linetype=OCC, alpha=OCC)) +
        geom_point() +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~n_in) +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Number of outliers") +        
        ylab("Power") +
        ggtitle("(a)") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"),
              plot.title = element_text(hjust = -0.11, vjust = -4))
    pp.1 %>% ggsave(file=sprintf("figures/experiment_real_animals_oracle_occ.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")
    

    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble", "Binary"))
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
    
    model.values <- c("Ensemble", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    model.labels <- c("Integrative", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    color.scale <- c("darkviolet", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue")
    shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    linetype.scale <- c(1, 2, 3, 4, 5, 6, 7, 8) 
    alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
   
    pp.2 <- df %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Metric=="Power") %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(BC=Model) %>%
        mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
        ggplot(aes(x=n_out, y=Mean, color=BC, shape=BC, linetype=BC, alpha=BC)) +
        geom_point() +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        ##geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~n_in) +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Number of outliers") +        
        ylab("Power") +
        ggtitle("(b)") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"),
              plot.title = element_text(hjust = -0.11, vjust = -4))
    pp.2 %>% ggsave(file=sprintf("figures/experiment_real_animals_oracle_bin.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")

}




if(plot.3) {
    idir <- "results_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "images_cars")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
    }))

    method.values <- c("Ensemble (one-class)", "One-Class", "Binary")
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

    alpha.nominal <- 0.01
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
        filter(Method %in% c("Ensemble", "Ensemble (one-class)", "Ensemble (one-class, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
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
        filter(Data=="images_cars", n_in==10000) %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_wrap(Metric~.) +
        scale_x_log10() +
        scale_y_continuous(lim=c(0,1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_cars.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")

}


if(plot.4) {
    idir <- "results_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "annthyroid") | startsWith(ifile, "mammography")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
    }))

    data.values <- c("annthyroid", "mammography")
    data.labels <- c("Thyroid disease", "Mammography")
    
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

    alpha.nominal <- 0.01
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
        filter(Method %in% c("Ensemble", "Ensemble (one-class)", "Ensemble (one-class, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (mixed, unweighted)", "Ensemble (binary, unweighted)", "Ensemble (one-class, unweighted)")) %>%
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
        filter(Metric=="Power", n_in %in% c(6399, 10000)) %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Data = factor(Data, data.values, data.labels)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_wrap(.~Data, scales="free") +
        scale_x_log10() +
        scale_y_continuous(lim=c(0,1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_tabular.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")


    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble", "One-Class"))
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
    
    model.values <- c("Ensemble", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    model.labels <- c("Integrative", "IF", "LOF", "SVM-pol", "SVM-rbf", "SVM-sgd", "SVM-sig")
    color.scale <- c("darkviolet", "orange", "orange", "orange", "orange", "orange", "orange", "orange")
    shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    linetype.scale <- c(1, 2, 2, 2, 2, 3, 2, 2)
    alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
   
    pp.1 <- df %>%
        filter(Alpha==alpha.nominal, Metric=="Power") %>%
        filter(Metric=="Power", n_in %in% c(6399, 10000)) %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(Data = factor(Data, data.values, data.labels)) %>%
        mutate(OCC=Model) %>% 
        mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
        ggplot(aes(x=n_out, y=Mean, color=OCC, shape=OCC, linetype=OCC, alpha=OCC)) +
        geom_point() +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~Data, scales="free") +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Number of outliers") +        
        ylab("Power") +
        ggtitle("(a)") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"),
              plot.title = element_text(hjust = -0.11, vjust = -4))
    pp.1 %>% ggsave(file=sprintf("figures/experiment_real_tabular_oracle_occ.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")
    

    df <- results.fdr.models %>%
        rbind(results.fdr.oracle) %>%
        filter(Method %in% c("Ensemble", "Binary"))
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
    
    model.values <- c("Ensemble", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    model.labels <- c("Integrative", "KNN", "MLP", "NB", "QDA", "RF", "SVC")
    color.scale <- c("darkviolet", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue")
    shape.scale <- c(8, 3, 3, 3, 3, 3, 3, 3)
    linetype.scale <- c(1, 2, 4, 2, 2, 3, 5, 2) 
    alpha.scale <- c(1,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
   
    pp.2 <- df %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Metric=="Power", n_in %in% c(6399, 10000)) %>%
        filter(Model %in% model.values) %>%
        mutate(Model = factor(Model, model.values, model.labels)) %>%
        mutate(Data = factor(Data, data.values, data.labels)) %>%
        mutate(BC=Model) %>%
        mutate(n_in = sprintf("Inliers: %d", n_in)) %>%
        ggplot(aes(x=n_out, y=Mean, color=BC, shape=BC, linetype=BC, alpha=BC)) +
        geom_point() +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_grid(.~Data, scales="free") +
        scale_x_log10() +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        scale_linetype_manual(values=linetype.scale) +
        xlab("Number of outliers") +        
        ylab("Power") +
        ggtitle("(b)") +
        theme_bw() +
        theme(legend.key.width = unit(1,"cm"),
              plot.title = element_text(hjust = -0.11, vjust = -4))
    pp.2 %>% ggsave(file=sprintf("figures/experiment_real_tabular_oracle_bin.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=2.25, units="in")

}




if(plot.5) {
    idir <- "results_cv_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "images_flowers")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
    }))
   
    method.values <- c("Ensemble (CV)", "Ensemble", "One-Class (CV)", "Binary (CV)")
    method.labels <- c("Integrative (TCV+)", "Integrative", "One-Class (oracle, CV+)", "Binary (oracle, CV+)")
    color.scale <- c("#da00da", "darkviolet", "red", "blue", "darkgreen", "green")
    shape.scale <- c(5, 8, 3, 1, 1)
    alpha.scale <- c(1, 1, 0.5, 0.5)

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
        group_by(Data, n_in, n_out, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class", "Binary (CV)", "One-Class (CV)")) %>%
        group_by(Data, n_in, n_out, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Data, n_in, n_out, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (CV)")) %>%
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
#        filter(Metric %in% c("Power","TPR")) %>%
        filter(n_in %in% c(513)) %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
#        mutate(Data = factor(Data, data.values, data.labels)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_wrap(.~Metric) +
        scale_x_log10() +
        scale_y_continuous(lim=c(0,1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_cv_flowers.pdf", ifelse(plot.fdr, "bh", "fixed")), width=6, height=2.25, units="in")



}



if(plot.6) {
    idir <- "results_cv_hpc/"
    ifile.list <- list.files(idir)

    results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
        if(startsWith(ifile, "annthyroid") | startsWith(ifile, "mammography")) {
            df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
        } else {
            df <- tibble()
        }
    }))

    data.values <- c("annthyroid", "mammography")
    data.labels <- c("Thyroid disease", "Mammography")
    
    method.values <- c("Ensemble (CV)", "One-Class (CV)", "Binary (CV)", "Ensemble", "One-Class", "Binary")
    method.labels <- c("Integrative (CV)", "OCC (CV, oracle)", "Binary (CV, oracle)", "Integrative", "OCC (oracle)", "Binary (oracle)")
    color.scale <- c("darkviolet", "red", "blue", "darkviolet", "red", "blue")
    shape.scale <- c(8, 3, 1, 8, 3, 1)
    alpha.scale <- c(1, 0.75, 0.75, 0.5, 0.25, 0.25)

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
        group_by(Data, n_in, n_out, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI))

    results.fdr.oracle <- results %>%
        filter(Method %in% c("Binary", "One-Class", "Binary (CV)", "One-Class (CV)")) %>%
        group_by(Data, n_in, n_out, Method, Model, Alpha) %>%
        summarise(Power.se=2*sd(Power)/sqrt(n()), Power=mean(Power), TypeI.se=2*sd(TypeI)/sqrt(n()), TypeI=mean(TypeI)) %>%
        group_by(Data, n_in, n_out, Method, Alpha) %>%
        summarise(idx.oracle = which.max(Power), Model="Oracle", Power=Power[idx.oracle], Power.se=Power.se[idx.oracle],
                  TypeI=TypeI[idx.oracle], TypeI.se=TypeI.se[idx.oracle]) %>%
        select(-idx.oracle)

    df <- results.fdr.models %>%
        filter(Method %in% c("Ensemble", "Ensemble (CV)")) %>%
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
#        filter(Metric %in% c("Power","TPR")) %>%
        filter(n_in %in% c(6399, 10000)) %>%
        filter(Alpha==alpha.nominal) %>%
        filter(Method %in% method.values) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        mutate(Data = factor(Data, data.values, data.labels)) %>%
        ggplot(aes(x=n_out, y=Mean, color=Method, shape=Method, alpha=Method)) +
        geom_point() +
        geom_line() +
#        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
#        geom_hline(aes(yintercept=Mean), data=df.nominal, linetype=2) +
        facet_wrap(Metric~Data) +
        scale_x_log10() +
        scale_y_continuous(lim=c(0,1)) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_alpha_manual(values=alpha.scale) +
        xlab("Number of outliers") +
        ylab("") +
        theme_bw()
    pp %>% ggsave(file=sprintf("figures/experiment_real_cv_tabular.pdf", ifelse(plot.fdr, "bh", "fixed")), width=5.5, height=4, units="in")



}
