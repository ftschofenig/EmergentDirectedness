rm(list=ls());gc()
library(dplyr); library(ggplot2); library(tidyverse);
library(lubridate); library(sjPlot); library(sjmisc)
library(lmtest); library(sandwich); library(gtsummary)
library(flextable); library(reshape2)
theme_set(theme_sjplot())

savepath<-"" #set path for where you want figures to be saved
datapath<-"" #set path for where data is saved locally

#Load Data
all_dt<-read.csv(paste(datapath, "df_all.csv", sep=""))
all_dt$net_id<-as.numeric(as.factor(all_dt$ID))
all_dt$tie_group<-all_dt$neighborhood_overlap_relative_quantile
all_dt$Dataset<-all_dt$Type
all_dt$net_id<-as.factor(all_dt$net_id)
all_dt$tie_group<-factor(all_dt$tie_group, levels=c("weak", "medium", "strong"))
levels(all_dt$tie_group)<-c("Weak","Medium", "Strong")
all_dt$Dataset<-as.factor(all_dt$Dataset)
levels(all_dt$Dataset)<-c("AddHealth","Banerjee", "Power law")
all_dt$threshold<-all_dt$T

#Analysis
all_dt_simp<-all_dt %>% select(threshold, tie_group, Dataset, net_id, CPC_max)

#MAX CPC
mod_maxCPC<-lm(CPC_max ~ tie_group + Dataset + net_id + threshold, data = all_dt_simp_sub)

summary(mod_maxCPC)

plot_model(mod_maxCPC, type = "pred", terms = c("tie_group", "Dataset"), 
           show.p = F, wrap.title = 100, wrap.labels = 100, 
           line.size = 1, grid.breaks = FALSE, 
           value.size=10, se = TRUE) + 
  theme_bw() + ylab("Tie Importance") + xlab("Tie Type") +
  theme(
    axis.title.x=element_text(size=35),
    axis.title.y=element_text(size=35),
    plot.title=element_blank(), 
    legend.text=element_text(size=35),
    legend.title=element_blank(),
    legend.position="top",
    legend.background = element_blank(),
    legend.box.background = element_rect(colour = "black",fill="white", linewidth=1.4),
    axis.text.x=element_text(size = 35, vjust=0.8),axis.text.y=element_text(size = 35),
    axis.line = element_line(colour = "black")) + 
  geom_line(size=1, position=position_dodge(0.1)) + 
  geom_point(size=1, position=position_dodge(0.1)) + 
  scale_y_continuous(breaks = seq(0.15, 0.27, 0.01))

ggsave("tie_type_model_by_dataset_and_T.png", width=12, height=12, path = savepath)

plot_model(mod_maxCPC, type = "pred", terms = c("tie_group"), 
           show.p = F, wrap.title = 100, wrap.labels = 100, 
           line.size = 1, grid.breaks = FALSE, 
           value.size=10, se = TRUE, color="black") + 
  theme_bw() + ylab("Tie Importance") + xlab("Tie Type") +
  theme(
    axis.title.x=element_text(size=35),
    axis.title.y=element_text(size=35),
    plot.title=element_blank(), 
    legend.text=element_text(size=35),
    legend.title=element_blank(),
    legend.position="top",
    legend.background = element_blank(),
    legend.box.background = element_rect(colour = "black",fill="white", linewidth=1.4),
    axis.text.x=element_text(size = 35, vjust=0.8),axis.text.y=element_text(size = 35),
    axis.line = element_line(colour = "black")) + 
  geom_line(linewidth=1, position=position_dodge(0.1), color="black") + 
  geom_point(size=8, position=position_dodge(0.1), color="black") 

ggsave("tie_type_model_by_T.png", width=12, height=12, path = savepath)






















