require('ggplot2')
require('reshape2')
setwd('/Users/erfan/Main/oasis/test_augmentation/results/')

d<-read.csv('data/results.balance.0.22.csv',sep=' ', header=F)
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
dw<-reshape(d, idvar = c("V1","V2","V3","V4","V5","V6","V7","V8","V10","V11","V12","V13","V14"), timevar = "V9", direction = "wide")
dw$diff<-dw$V15.augmented - dw$V15.original

d1<-d[d$V1=="beta-binomial" & d$V9=="original" & d$V2 == "phylogeny",]

d3<-droplevels(d1)
levels(d3$V1)[1]<-"no-augmentation"
d3$V2<-"non"
d3$V4<-"non"
d3$V5<-"non"
d<-droplevels(d[d$V9 != "original",])
d<-rbind(d,d3)
d[d$V1=="resampling","V5"] = 0
d$method<-as.factor(paste(d$V1,d$V2,d$V4,sep="-"))

dw$method<-as.factor(paste(dw$V1,dw$V2,dw$V4,sep="-"))


levels(dw$method)<-list("no augmentation"="no-augmentation-non-non",
                        "Ind-Downsampling"="resampling-downsampling-non",
                        "Ind-Resampling"="resampling-subsampling-non", 
                        "ADASYN"="SMOTE-non-ADASYN.5",
                        "SMOTE"="SMOTE-non-SMOTE.5",
                        "Phy-BetaBinomial-Class"="beta-binomial-from-data-non",
                        "Phy-Binomial-Class"="binomial-from-data-non",
                        "Phy-BetaBinomial-Samp"="beta-binomial-phylogeny-100-0.5-0",
                        "Phy-Binomial-Samp"="binomial-phylogeny-non")

d<-droplevels(d[!d$method %in% c("SMOTE-non-SMOTE.40","SMOTE-non-ADASYN.40"),])

levels(d$method)<-list("no augmentation"="no-augmentation-non-non",
                      "Ind-Downsampling"="resampling-downsampling-non",
                      "Ind-Resampling"="resampling-subsampling-non", 
                      "ADASYN"="SMOTE-non-ADASYN.5",
                      "SMOTE"="SMOTE-non-SMOTE.5",
                      "Phy-BetaBinomial-Class"="beta-binomial-from-data-non",
                      "Phy-Binomial-Class"="binomial-from-data-non",
                      "Phy-BetaBinomial-Samp"="beta-binomial-phylogeny-100-0.5-0",
                      "Phy-Binomial-Samp"="binomial-phylogeny-non")


d$h<-as.factor(paste(paste(d$V10,d$V12," "), d$V13,sep="\n"))

ggplot(d[d$V8=="AUC"& d$V14 == "original" & 
           d$V5 != 20 & 
           d$method %in% c("no augmentation","Ind-Downsampling","ADASYN","SMOTE","Phy-BetaBinomial-Samp","Phy-Binomial-Samp") &
           d$V13 %in% c("10-vs-90", 
                        "20-vs-80", 
                        "33-vs-66"),], 
       aes(x=method,y=V15, shape=V5,color=method))+stat_summary(alpha=0.8)+
  facet_grid(V10~V12,scales="free_y")+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),text=element_text(size=14),
        legend.position = "bottom", legend.box = 'vertical')+
  scale_color_manual(name="",values = c("#d95f02","#1b9e77","#7570b3","#7570b3","#e7298a","#e7298a"))+
  scale_x_discrete(labels=c("no augmentation", "downsampling", "ADASYN", "SMOTE", "TADA-TVSV-m", "TADA-SV"))+
  scale_shape_discrete(labels=c("Balance","Balance++","No Augmentation"))
ggsave("figures/balance-unbiased.pdf",width=6.6, height=6.6)


ggplot(d[d$V8=="AUC"& d$V14 == "original" & 
           d$V5 != 20 & d$method %in% c("no augmentation",
                                        "Ind-Downsampling","Phy-BetaBinomial-Samp","Phy-Binomial-Samp","ADASYN","SMOTE")& 
           d$V13 %in% c("80-vs-20", 
                        "20-vs-80", 
                        "50-vs-50"),],
       aes(x=method,y=V15, color=method, shape=V5))+stat_summary(alpha=0.8)+
  facet_grid(V10~V13,scales="free")+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),text=element_text(size=14),
        legend.position = "bottom", legend.box = 'vertical')+
  scale_color_manual(name="",values = c("#d95f02","#1b9e77","#7570b3","#7570b3","#e7298a","#e7298a"))+
  scale_shape_discrete(labels=c("Balance","Balance++","No Augmentation"))+
  scale_x_discrete(labels=c("no augmentation", "downsampling", "ADASYN", "SMOTE", "TADA-TVSV-m", "TADA-SV"))
ggsave('figures/balance-biased.pdf', width= 6.6, height=6.6)



d$noskl<-0.8


ggplot(d[d$V8=="accuracy"& d$V14 == "original" & 
           d$V5 != 20 & d$V10 == "RF" &
           d$method %in% c("no augmentation","Phy-BetaBinomial-Samp", "Phy-Binomial-Samp") &
           d$V13 %in% c("10-vs-90", 
                        "20-vs-80", 
                        "33-vs-66"),], 
       aes(x=method,y=V15, color=V5))+stat_summary(alpha=0.8,size=0.3)+
  facet_wrap(V12~.)+
  theme_bw() +xlab("")+ylab("Acc.")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+geom_hline(yintercept = 0.66,color="red")+
  scale_x_discrete(labels=c("no augmentation","TADA-TVSV-m","TADA-SV"))+
  scale_color_brewer(name="",palette = "Dark2",labels=c("Balance","Balance++","No augmentation"))
ggsave('figures/balance-unbiased-acc.pdf', width= 6.6, height=3.8)




dw$h<-as.factor(paste(paste(dw$V10,dw$V12," "), dw$V13,sep="\n"))







dcast(dw[dw$V8=="AUC"& dw$V14 == "original" & 
           dw$V5 != 20 & 
           dw$method %in% c("no augmentation","Ind-Downsampling","Phy-BetaBinomial-Samp","Phy-Binomial-Samp") &
           dw$V13 %in% c("10-vs-90", 
                         "20-vs-80", 
                         "33-vs-66"),], method+V5+V10+V12~.,value.var="diff",fun.aggregate = mean)

t<-d[d$V8=="AUC"& d$V14 == "original" & d$method=="Phy-Binomial-Samp"& d$V10=="RF"&
       d$V5 != 20 & 
       d$method %in% c("no augmentation","Ind-Downsampling","Phy-BetaBinomial-Samp","Phy-Binomial-Samp") &d$V12 == "20-vs-80"&
       d$V13 %in% c("10-vs-90", 
                    "20-vs-80", 
                    "33-vs-66") & d$V9=="augmented",]


dw<-reshape(t, idvar = c("V1","V2","V3","V4","V6","V7","V8","V9","V10","V11","V12","V13","V14"), timevar = "V5", direction = "wide")



t<-droplevels(d[d$V8 == "AUC" & d$V14 == "original" & d$method %in% 
       c("Phy-Binomial-Samp","SMOTE") & d$V10== "RF" & d$V5!=20 &
     d$V13 %in% c("10-vs-90", "20-vs-80",  "33-vs-66") & d$V9=="augmented" & d$V5=="50", c(1,3,5,6,7,8,9,10,11,12,13,15) ])
dw<-reshape(t, idvar = c("V3","V5","V6","V7","V8","V9","V10","V11","V12","V13"), timevar = "V1", direction = "wide")
dw$diff<-dw$V15.binomial - dw$V15.SMOTE
t.test(dw[dw$V12=="33-vs-66",]$diff)
t.test(dw[dw$V12=="20-vs-80",]$diff)
t.test(dw[dw$V12=="10-vs-90",]$diff)




t<-droplevels(d[d$V8 == "AUC" & d$V14 == "original" & d$method %in% 
                  c("Phy-Binomial-Samp","ADASYN") & d$V10== "RF" & d$V5!=20 &
                  d$V13 %in% c("10-vs-90", "20-vs-80",  "33-vs-66") & d$V9=="augmented" & d$V5=="50", c(1,3,5,6,7,8,9,10,11,12,13,15) ])
dw<-reshape(t, idvar = c("V3","V5","V6","V7","V8","V9","V10","V11","V12","V13"), timevar = "V1", direction = "wide")
dw$diff<-dw$V15.binomial - dw$V15.SMOTE
t.test(dw[dw$V12=="33-vs-66",]$diff)
t.test(dw[dw$V12=="20-vs-80",]$diff)
t.test(dw[dw$V12=="10-vs-90",]$diff)



t<-d[d$V8=="AUC"& d$V14 == "original" & d$method=="SMOTE"& d$V10=="RF"&
       d$V5 != 20  &d$V12 == "20-vs-80"&
       d$V13 %in% c("10-vs-90", 
                    "20-vs-80", 
                    "33-vs-66") & d$V9=="augmented",]


t<-droplevels(dw[dw$V8 == "AUC" & dw$V14 == "original" & dw$V4  == "SMOTE.5" & 
                  dw$V10== "RF" & dw$V5!=20 &
                  dw$V13 %in% c("10-vs-90", "20-vs-80",  "33-vs-66") & dw$V5=="50", ])

t$diff<-t$V15.augmented-t$V15.original
t.test(t[t$V12=="33-vs-66",]$diff)
t.test(t[t$V12=="20-vs-80",]$diff)
t.test(t[t$V12=="10-vs-90",]$diff)


t<-droplevels(dw[dw$V8 == "AUC" & dw$V14 == "original" & dw$V4  == "ADASYN.5" & 
                   dw$V10== "RF" & dw$V5!=20 &
                   dw$V13 %in% c("10-vs-90", "20-vs-80",  "33-vs-66") & dw$V5=="50", ])

t$diff<-t$V15.augmented-t$V15.original
t.test(t[t$V12=="33-vs-66",]$diff)
t.test(t[t$V12=="20-vs-80",]$diff)
t.test(t[t$V12=="10-vs-90",]$diff)

