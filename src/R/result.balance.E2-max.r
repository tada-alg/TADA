require('ggplot2')
require('reshape2')
setwd('/Users/erfan/Main/oasis/test_augmentation/results/')

d<-read.csv('data/results.balance.direction1.csv',sep=' ', header=F)
d$V12<-as.factor(d$V12)
d$V13<-as.factor(d$V13)
d[d$V12=="33.67","V12"]<-"33.66"
d[d$V13=="33.67","V13"]<-"33.66"

levels(d$V12)<-list("10-vs-90"="10.9",
                    "20-vs-80"="20.8",
                    "33-vs-66"="33.66")

levels(d$V13)<-list("10-vs-90"="10.9",
                    "20-vs-80"="20.8",
                    "33-vs-66"="33.66",
                    "50-vs-50"="50.5",
                    "80-vs-20"="80.2")


levels(d$V10)<-list("RF"="rf", "NN"="nn")
dw<-reshape(d, idvar = c("V1","V2","V3","V4","V5","V6","V7","V8","V10","V11","V12","V13"), timevar = "V9", direction = "wide")
dw$diff<-dw$V14.augmented - dw$V14.original

d1<-d[d$V1=="binomial" & d$V9=="original" & d$V2 == "phylogeny",]

d3<-droplevels(d1)
levels(d3$V1)[1]<-"no-augmentation"
d3$V2<-"non"
d3$V4<-"non"
d3$V5<-"non"
d<-droplevels(d[d$V9 != "original",])
d<-rbind(d,d3)
d$method<-as.factor(paste(d$V1,d$V2,d$V4,sep="-"))

dw$method<-as.factor(paste(dw$V1,dw$V2,dw$V4,sep="-"))


levels(dw$method)<-list("no augmentation"="no-augmentation-non-non",
                        "Ind-Downsampling"="resampling-downsampling-non-non-non",
                        "Ind-Resampling"="resampling-subsampling-non-non-non",
                        "ADASYN-40" = "SMOTE-non-ADASYN.40",
                        "SMOTE-40" = "SMOTE-non-SMOTE.40",
                        "ADASYN" = "SMOTE-non-ADASYN.5",
                        "SMOTE" = "SMOTE-non-SMOTE.5",
                        "TADA-TVSV-C"="beta-binomial-from-data-non",
                        "TADA-SV-C"="binomial-from_data-non-non-non",
                        "TADA-TVSV-m"="beta_binomial_latest-phylogeny-100-0.5-0",
                        "TADA-SV"="binomial-phylogeny-non-non-non")

levels(d$method)<-list("no augmentation"="no-augmentation-non-non",
                       "Ind-Downsampling"="resampling-downsampling-non-non-non",
                       "Ind-Resampling"="resampling-subsampling-non-non-non",
                       "ADASYN-40" = "SMOTE-non-ADASYN.40",
                       "SMOTE-40" = "SMOTE-non-SMOTE.40",
                       "ADASYN" = "SMOTE-non-ADASYN.5",
                       "SMOTE" = "SMOTE-non-SMOTE.5",
                       "TADA-TVSV-C"="beta-binomial-from-data-non",
                       "TADA-SV-C"="binomial-from_data-non-non-non",
                       "TADA-TVSV-m"="beta_binomial_latest-phylogeny-100-0.5-0",
                       "TADA-SV"="binomial-phylogeny-non-non-non")




d$h<-as.factor(paste(paste(d$V10,d$V12," "), d$V13,sep="\n"))
d[d$method=="Ind-Downsampling","V5"] = 0
ggplot(d[d$V8=="AUC"& d$method %in% c("no augmentation", "Ind-Downsampling", "ADASYN","SMOTE","TADA-SV","TADA-TVSV-m") &
           d$V5 != 20 & 
           d$V13 %in% c("10-vs-90", 
                        "20-vs-80", 
                        "33-vs-66"),], 
       aes(x=method,y=V14, shape=V5, color=method))+stat_summary(alpha=0.8)+
  facet_grid(V10~V12,scales="free")+
  theme_bw() +xlab("Augmentation method")+ylab("AUC")+
theme(axis.text.x=element_text(angle=45,hjust=1),text=element_text(size=14),
      legend.position = "bottom", legend.box = 'vertical')+
  scale_color_manual(name="",values = c("#d95f02","#1b9e77","#7570b3","#7570b3","#e7298a","#e7298a"))+
  scale_shape_discrete(name="", labels=c("Balance","Balance++","No Augmentation"))+
  scale_x_discrete(name="",labels=c("no augmentation", "downsampling", "ADASYN", "SMOTE", "TADA-TVSV-m", "TADA-SV"))
ggsave('figures/swaped_balance_auc.pdf',width= 6.6, height=6.6)




