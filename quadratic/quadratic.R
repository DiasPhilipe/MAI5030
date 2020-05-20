library(ggplot2)
library(data.table)
library(dplyr)
library(reticulate)

###
py_run_file(file = "/media/philipe/5650A03850A020AB/quadratic/quadratic.py")

data.table(Geracão = 1:100, cga = unlist(py$progress_cga), sga = unlist(py$progress_sga)) %>%
  melt.data.table(id.vars = 'Geracão', variable.name = "Algorítimo", value.name = "Ajuste") %>%
  ggplot(aes(x = Geracão, y = Ajuste, color = Algorítimo)) + 
  geom_line() + 
  scale_color_brewer(palette = "Set1")

###
py_run_file(file = "/media/philipe/5650A03850A020AB/quadratic/quadratic_comp.py")

for(i in 1:length(py$sizes)){
  sga_acc = round(sum(unlist(py$fitness_sga[[i]])==py$global_fitness)/py$tests, digits = 3)*100
  cga_acc = round(sum(unlist(py$fitness_cga[[i]])==py$global_fitness)/py$tests, digits = 3)*100
  
  print(
    data.table(Execuções = 1:py$tests, cga = unlist(py$fitness_cga[[i]]), sga = unlist(py$fitness_sga[[i]])) %>%
      melt.data.table(id.vars = 'Execuções', variable.name = "Acurácia", value.name = "Ajuste") %>%
      mutate(Acurácia = if_else(Acurácia=="cga", paste0(Acurácia, " - ", cga_acc, "%"), paste0(Acurácia, " - ", sga_acc, "%"))) %>%
      ggplot(aes(x = Execuções, y = Ajuste, color = Acurácia)) + 
      ylim(c(0,20)) +
      geom_line() + 
      scale_color_brewer(palette = "Set1") +
      ggtitle(paste0("População de tamanho ", py$sizes[i]))
  )
}

###
py_run_file(file = "/media/philipe/5650A03850A020AB/quadratic/quadratic_cov.py")

melt(py$sga_cm$icm, value.name = "Cov") %>%
  ggplot(aes(x=Var1, y=Var2, fill=Cov)) +
  geom_tile() +
  scale_fill_continuous(limits=c(-20, 20)) +
  ggtitle("sga - População inicial") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())

melt(py$sga_cm$tcm, value.name = "Cov") %>%
  ggplot(aes(x=Var1, y=Var2, fill=Cov)) +
  geom_tile() +
  scale_fill_continuous(limits=c(-20, 20)) +
  ggtitle("sga - População intermediária") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())

melt(py$sga_cm$fcm, value.name = "Cov") %>%
  ggplot(aes(x=Var1, y=Var2, fill=Cov)) +
  geom_tile() +
  scale_fill_continuous(limits=c(-20, 20)) +
  ggtitle("sga - População final") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())

melt(py$cga_cm$icm, value.name = "Cov") %>%
  ggplot(aes(x=Var1, y=Var2, fill=Cov)) +
  geom_tile() +
  scale_fill_continuous(limits=c(-20, 20)) +
  ggtitle("cga - População inicial") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())

melt(py$cga_cm$tcm, value.name = "Cov") %>%
  ggplot(aes(x=Var1, y=Var2, fill=Cov)) +
  geom_tile() +
  scale_fill_continuous(limits=c(-20, 20)) +
  ggtitle("cga - População intermediária") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())

melt(py$cga_cm$fcm, value.name = "Cov") %>%
  ggplot(aes(x=Var1, y=Var2, fill=Cov)) +
  geom_tile() +
  scale_fill_continuous(limits=c(-20, 20)) +
  ggtitle("cga - População final") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())
