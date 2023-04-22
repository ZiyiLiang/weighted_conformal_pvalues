options(width=160)

library(tidyverse)

psi <- function(n, nu) {
    if(nu<1) {
        out <- (n+1) * ( VGAM::zeta(2-nu) + 1/(nu-1) * (n+1)^(nu-1) ) / ( VGAM::zeta(1-nu) + (n+1)^(nu) / nu )
    } else if (nu==1) {
        out <- log(n+1)
    } else {
        out <- nu / (nu-1)
    }
    return(out)
}

compute.Xi <- function(epsilon, gamma, nu0, nu1, n1) {
    out <- (epsilon + gamma - 2 * epsilon * gamma) * ( epsilon*gamma + (1-epsilon)*(1-gamma) )
    out <- out / ( (1-epsilon) * gamma * nu0 / (1+nu0) + epsilon * (1-gamma) * nu1/(1+nu1) )
    out <- out / ( epsilon * gamma * psi(n1, nu0) + (1-epsilon) * (1-gamma) * psi(n1, nu1) )
    return(out)
}

find.nu0 <- function(epsilon, gamma, nu1, n1) {
    if(compute.Xi(epsilon, gamma, 1/(1+n1), nu1, n1) < 1) {
        return(0)
    }
    f <- function(nu0) {
        xi <- compute.Xi(epsilon, gamma, nu0, nu1, n1)
        return(xi-1)
    }
    out <- cmna::bisection(f, 0, nu1, tol = 1/(n1*100), m = 100)
    if(out<1/(1+n1)) {
        return(0)
    }
    return(out)
}

nu0.seq <- seq(0,1,length.out=100)
mean0.seq <- nu0.seq / (1 + nu0.seq)

gamma = 0.5
df <- tibble()
for(epsilon in c(0,0.01,0.1)) {
    for(n1 in c(100,200,1000)) {
        for(nu1 in c(0.5, 0.75, 0.9,1)) {
            xi.seq <- sapply(nu0.seq, function(nu0) compute.Xi(epsilon, gamma, nu0, nu1, n1))
            df.tmp <- tibble(nu0 = nu0.seq, mean0 = mean0.seq, xi=xi.seq, nu1=nu1, mean1=nu1/(1+nu1), n1=n1, epsilon=epsilon, nu0.min=1/(n1+1), mean0.min=nu0.min/(nu0.min+1))
            df <- rbind(df, df.tmp)
        }
    }
}

eps.labs <- c(parse(text=latex2exp::TeX("$\\epsilon=0$")), parse(text=latex2exp::TeX("$\\epsilon=0.01$")), parse(text=latex2exp::TeX("$\\epsilon=0.1$")))    
n1.labs <- c(parse(text=latex2exp::TeX("$n_1=100$")), parse(text=latex2exp::TeX("$n_1=200$")), parse(text=latex2exp::TeX("$n_1=1000$")))

pp <- df %>%
    mutate(mean1 = as.factor(round(mean1,3))) %>%
    mutate(nu1 = as.factor(round(nu1,3))) %>%
    mutate(epsilon = factor(epsilon, c(0,0.01,0.1), eps.labs)) %>%
    mutate(n1 = factor(n1, c(100,200,1000), n1.labs)) %>%
    ggplot(aes(x=mean0, y=xi, color=mean1)) +
    geom_line() +
    facet_grid(epsilon~n1, labeller= labeller(epsilon = label_parsed, n1 = label_parsed)) +
    coord_cartesian(ylim=c(0.5,2)) +
#    scale_y_continuous(trans='log') +
    geom_vline(aes(xintercept=mean0.min), color="black", linetype=2) +
    geom_hline(yintercept=1, color="black", linetype=2) +
    scale_x_continuous(breaks=c(0,0.25,0.5)) +
    xlab(latex2exp::TeX("$\\nu_0/(1+\\nu_0)$")) +
    ylab(latex2exp::TeX("$\\Xi$")) +
    labs(color=latex2exp::TeX("$\\nu_1/(1+\\nu_1)$")) +
#    labs(color=latex2exp::TeX("$\\nu_1$")) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
pp %>% ggsave(file=sprintf("figures/power_analysis_xi_gamma0.5.pdf"), width=5, height=4, units="in")


gamma = 0.5
df <- tibble()
for(epsilon in c(0,0.01,0.1)) {
    for(n1 in c(100,200,1000)) {
        for(nu1 in seq(1/(n1+1),1,length.out=20)) {
            nu0 <- find.nu0(epsilon, gamma, nu1, n1)
            df.tmp <- tibble(nu0 = nu0, mean0 = nu0/(1+nu0), mean1 = nu1/(1+nu1), xi=xi.seq, nu1=nu1, n1=n1, epsilon=epsilon, gamma=gamma,
                             nu0.min=1/(n1+1), mean0.min=nu0.min/(nu0.min+1))
            df <- rbind(df, df.tmp)
        }
    }
}

pp <- df %>% 
    mutate(n1 = factor(n1, c(100,200,1000), n1.labs)) %>%
    mutate(epsilon = factor(epsilon, c(0,0.01,0.1), eps.labs)) %>%
    ggplot(aes(x=mean1, y=mean0)) +
    geom_point() +
    geom_line() +
    facet_grid(epsilon~n1, labeller= labeller(epsilon = label_parsed, n1 = label_parsed)) +
    geom_hline(aes(yintercept=mean0.min), color="red", linetype=2) +
    geom_vline(aes(xintercept=mean0.min), color="red", linetype=2) +
    scale_y_continuous(trans='sqrt') +
    scale_x_continuous(trans='sqrt', breaks=c(0.1,0.25,0.5)) +
    coord_cartesian(ylim=c(0,0.5)) +
    xlab(latex2exp::TeX("$\\nu_1/(1+\\nu_1)$")) +
    ylab(latex2exp::TeX("$\\nu_0/(1+\\nu_0)$")) +
    theme_bw()
pp %>% ggsave(file=sprintf("figures/power_analysis_gamma0.5.pdf"), width=5, height=4.5, units="in")



if(FALSE) {
    epsilon = 0.1
    df <- tibble()
    for(gamma in c(0.1,0.25,0.5)) {
        for(n1 in c(10,100,1000)) {
            for(nu1 in seq(1/(n1+1),1,length.out=20)) {
                nu0 <- find.nu0(epsilon, gamma, nu1, n1)
                df.tmp <- tibble(nu0 = nu0, mean0 = nu0/(1+nu0), mean1 = nu1/(1+nu1), xi=xi.seq, nu1=nu1, n1=n1, epsilon=epsilon, gamma=gamma,
                                 nu0.min=1/(n1+1), mean0.min=nu0.min/(nu0.min+1))
                df <- rbind(df, df.tmp)
            }
        }
    }

    df %>%
        ggplot(aes(x=mean1, y=mean0)) +
        geom_point() +
        geom_line() +
        facet_grid(gamma~n1, labeller = label_both) +
        geom_hline(aes(yintercept=mean0.min), color="red", linetype=2) +
        geom_vline(aes(xintercept=mean0.min), color="red", linetype=2) +
        scale_y_continuous(trans='sqrt') +
        scale_x_continuous(trans='sqrt') +
        coord_cartesian(ylim=c(0,0.5)) +
        xlab(latex2exp::TeX("$\\nu_1/(1+\\nu_1)$")) +
        ylab(latex2exp::TeX("$\\nu_0/(1+\\nu_0)$")) +
        theme_bw()
}
