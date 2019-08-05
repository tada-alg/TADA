require('ggplot2')
require('reshape2')
setwd('/Users/erfan/Main/oasis/test_augmentation/results/')
d<-read.csv('data/final_results/E1.final.csv',sep=' ', header=F)

d<-droplevels(d[d$V3%in%c("IBD", "bmi"),])

levels(d$V9)<-list( "RF"="rf","NN"="nn")
levels(d$V3)<-list("IBD"="IBD","BMI category"="bmi")
d<-droplevels(d[d$V2%in%c("non","phylogeny"),])
dw<-reshape(d[d$V7=="AUC",], idvar = c("V1","V2","V3","V4","V5","V6","V7","V9","V10"), timevar = "V8", direction = "wide")
dw$diff<-dw$V11.augmented - dw$V11.original


# ggplot(data=d[d$V8 == 5 & d$V1 == "only_beta" & d$V10 =="AUC",  ],
#        aes(x=reorder(model,V15),y=V15,fill=V12))+geom_boxplot()+facet_wrap(V4~.,scales="free_y")+
#   theme_bw()+theme(axis.text.x = element_text(angle=45,hjust=1))

d1<-droplevels(d[d$V8=="original"  & d$V5==50 & d$V1 == "beta-binomial" & d$V4== "100-0.5-0" & d$V2 == "phylogeny",])

d3<-droplevels(d1)
levels(d3$V1)[1]<-"no-augmentation"
levels(d3$V2)[1]<-"non"
levels(d3$V4)[1]<-"non-non-non"
levels(d3$V5)[1]<-"non"
d<-droplevels(d[d$V8 != "original",])
d<-rbind(d,d3)
dw$method<-as.factor(paste(dw$V1,dw$V2,dw$V4,sep="-"))
d$method<-as.factor(paste(d$V1,d$V2,d$V4,sep="-"))


levels(dw$method)<-list("no augmentation"="no-augmentation-non-non-non-non",
                        "Resampling"="resampling-non-non-non-non",
                        "SV"="binomial-phylogeny-non-non-non",
                        "TVSV-1"="beta-binomial-clustering-1-phylogeny-100-0.5-1",
                        "TVSV-4"="beta-binomial-clustering-4-phylogeny-100-0.5-1",
                        "TVSV-8"="beta-binomial-clustering-8-phylogeny-100-0.5-1",
                        "TVSV-40"="beta-binomial-clustering-40-phylogeny-100-0.5-1",
                        "TVSV-m"="beta-binomial-phylogeny-100-0.5-0")

levels(d$method)<-list("no augmentation"="no-augmentation-non-non-non-non",
                       "Resampling"="resampling-non-non-non-non",
                       "ADASYN-40"="SMOTE-non-ADASYN.40",
                       "ADASYN-5"="SMOTE-non-ADASYN.5",                             
                       "SMOTE-40"="SMOTE-non-SMOTE.40",                             
                       "SMOTE-5"="SMOTE-non-SMOTE.5",
                       "TADA-TVSV-1"="beta-binomial-clustering-1-phylogeny-100-0.5-1",
                       "TADA-TVSV-4"="beta-binomial-clustering-4-phylogeny-100-0.5-1",
                       "TADA-TVSV-8"="beta-binomial-clustering-8-phylogeny-100-0.5-1",
                       "TADA-TVSV-40"="beta-binomial-clustering-40-phylogeny-100-0.5-1",
                       "TADA-TVSV-m"="beta-binomial-phylogeny-100-0.5-0",
                       "TADA-SV"="binomial-phylogeny-non-non-non")

d$V5<-as.factor(d$V5)
d$levelaug<-as.factor(paste(d$V9,d$V5,sep="."))
levels(d$levelaug)<-list( "NN (no aug.)"="NN.non", "NN (5x)"="NN.5",
                          "NN (20x)" = "NN.20", "NN (25x)"="NN.25","NN (50x)" = "NN.50", 
                          "RF (no aug.)"="RF.non", "RF (5x)"="RF.5",
                          "RF (20x)" = "RF.20", "RF (25x)"="RF.25", "RF (50x)" = "RF.50")

dw$V5<-as.factor(dw$V5)
dw$levelaug<-as.factor(interaction(dw$V9,dw$V5))
levels(dw$levelaug)<-list(  "NN (no aug.)"="NN.non", "NN (5x)"="NN.5",
                            "NN (20x)" = "NN.20", "NN (25x)"="NN.25","NN (50x)" = "NN.50", 
                            "RF (no aug.)"="RF.non", "RF (5x)"="RF.5",
                            "RF (20x)" = "RF.20", "RF (25x)"="RF.25", "RF (50x)" = "RF.50")
d$col<-"no augmentation"
d[grepl("TADA-TVSV",d$method),"col"]<-"TADA"
d[d$method=="TADA-SV", "col"]<-"TADA"
d[d$method %in% c("SMOTE-5","SMOTE-40" ), "col"]<-"SMOTE"
d[d$method %in% c("ADASYN-5", "ADASYN-40"), "col"]<-'ADASYN'
d$col<-as.factor(d$col)

dw$col<-"no augmentation"
dw[grepl("TADA-TVSV",dw$method),15]<-"TADA-TVSV"
dw[dw$method=="TADA-SV", 15]<-"SV"
dw$col<-as.factor(dw$col)

ggplot(data=d[(d$V5 == "25" | d$V5=="20" | d$V5 == "50" | d$V5 == "5"|d$V5 == "non" )
              & d$V7=="AUC" & d$method %in% c("TADA-SV", "TADA-TVSV-m"), ],
       aes(x=method,y=V11,colour=V5,shape=V5==25&method=="TADA-TVSV-m"))+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  stat_summary(position=position_dodge(0.5))+facet_grid(V3~V9,scales = "free_y")+
  scale_shape_discrete(name="",labels=c("k!=q","k=q"))+
  scale_color_brewer(name="k×q",palette = "Dark2")
ggsave('figures/different-k-and-q-auc.pdf', width= 5, height=6.3)


d<-droplevels(d[!((d$method == "SV" | d$method == "TVSV-m") & d$V6 == "0"), ])

ggplot(data=d[(d$V5 == "50" | d$V5=="5"| d$V5 == "non") & 
                d$V7=="AUC" & (!d$method %in% c("ADASYN-40", "SMOTE-40")) & 
                !d$method == "Resampling", ],
       aes(x=method,y=V11,colour=method,shape=interaction(V5,method!="no augmentation")))+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=90,hjust=1),text = element_text(size=14),
        legend.position = "bottom",legend.box="vertical")+
  stat_summary(position=position_dodge(width=0.6))+facet_grid(V3~V9,scales = "free_y")+
  scale_x_discrete(name="",labels=c("no augmentation", 
                                    "ADASYN", "SMOTE", "TADA-TVSV-1", "TADA-TVSV-4",
                                    "TADA-TVSV-8", "TADA-TVSV-40", "TADA-TVSV-m","TADA-SV"))+
  scale_shape_discrete(name="k×q",labels=c("non","5","50"))+
  scale_color_manual(name="",values = c("#d95f02","#7570b3","#7570b3","#e7298a",
                                        "#e7298a","#e7298a","#e7298a","#e7298a","#e7298a"),
                     labels=c("no augmentation", "ADASYN","SMOTE",
                              "TADA","TADA","TADA","TADA",
                              "TADA","TADA"))
  ggsave('figures/compare-methods-AUC-final.pdf', width= 6.4, height=8)

ggplot(data=d[(d$V5 == "5" | d$V5 == "non") & d$V7=="accuracy" & !d$method == "Resampling" , ],
       aes(x=method,y=V11,colour=col))+
  theme_bw() +xlab("")+ylab("Acc")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  stat_summary()+facet_grid(V3~V9,scales = "free_y")+
  scale_color_brewer(name="",palette = "Dark2")
ggsave('figures/compare-methods-acc-final.pdf', width= 5, height=6.3)

d$col2<-"no augmentation"
d[d$V5=="5" & d$method != "no augmentation","col2"]<-"5"
d[d$V5=="20" & d$method != "no augmentation","col2"]<-"20"
d[d$V5=="50" & d$method != "no augmentation","col2"]<-"50"

d$col2<-as.factor(d$col2)
d$col2<-factor(d$col2,levels=c("no augmentation","5","20","50"))
ggplot(data=d[d$V7=="AUC" & !d$method == "Resampling" & d$method %in% c("no augmentation", "SV","TVSV-m"), ],
       aes(x=method,y=V11,color=col2))+
  theme_bw() +xlab("")+ylab("AUC")+
  theme(axis.text.x=element_text(angle=45,hjust=1),
        legend.position = "bottom")+
  stat_summary(position=position_dodge(width=0.4))+facet_grid(V3~V9,scales = "free_y")+
  scale_color_brewer(name="",palette = "Dark2")
ggsave('figures/compare-methods-supplementary-AUC-final.pdf', width= 5, height=6.3)

### p-value for IBD RF
t.test(dw[dw$method =="TVSV-m" & dw$V7=="AUC" & !dw$method == "Resampling" & dw$V9 == "RF" & dw$V5 %in% c("5") & dw$V3== "IBD",]$diff)
nrow(d[d$method =="TADA-TVSV-m" & d$V7=="AUC" & !d$method == "Resampling" & d$V9 == "RF" & d$V5 %in% c("50") & d$V3== "IBD",])

mean(d[d$method =="TADA-TVSV-m" & d$V7=="AUC"& d$V9 == "RF" & d$V5 %in% c("50") & d$V3== "BMI category",]$V11)
mean(d[d$method =="TADA-TVSV-m" & d$V7=="AUC"& d$V9 == "RF" & d$V5 %in% c("50") & d$V3== "IBD",]$V11)

nrow(d[d$method =="TADA-TVSV-m" & d$V7=="AUC"& d$V9 == "RF" & d$V5 %in% c("5") & d$V3== "BMI category",])


dcast(dw[dw$V7=="AUC" & dw$V5== "5" & dw$method !="Resampling",],method+V9+V5+V3~.,value.var="diff",fun.aggregate = mean)

t.test(dw[dw$method =="TVSV-m" & dw$V7=="AUC"& dw$V9 == "NN" & dw$V5 %in% c("50") & dw$V3== "IBD",]$diff)
dcast(dw[(dw$V5 == "5" ) & dw$V7=="AUC" & dw$method == "TVSV-m" , ],method+col+V3+V9~.,value.var="diff",fun.aggregate = mean)
dcast(d[d$V5 == "5"  & d$V7=="AUC" & d$method == "TVSV-m" , ],method+col+V3+V9~.,value.var="V11",fun.aggregate = mean)


dt<-droplevels(d[d$method %in% c("TADA-SV","ADASYN-5") & 
                   d$V5 == 50 & d$V8 == "augmented" &  
                   d$V9=="RF" & d$V3=="IBD" & d$V7=="AUC",])
nrow(dt)
dw<-reshape(dt[,c(1,6,7,8,9,10,11)], 
            idvar = c("V6","V7","V8","V9","V10"), timevar = "V1", direction = "wide")
dw$diff<-dw$V11.binomial-dw$V11.SMOTE
t.test(dw$diff)



dt<-droplevels(d[d$method %in% c("TADA-SV","SMOTE-5") & 
                   d$V5 == 50 & d$V8 == "augmented" &  
                   d$V9=="RF" & d$V3=="IBD" & d$V7=="AUC",])
nrow(dt)
dw<-reshape(dt[,c(1,6,7,8,9,10,11)], 
            idvar = c("V6","V7","V8","V9","V10"), timevar = "V1", direction = "wide")
dw$diff<-dw$V11.binomial-dw$V11.SMOTE
t.test(dw$diff)


t<-dw[dw$V4 %in% c("SMOTE.5") & 
       dw$V5 == 50 &  
       dw$V9=="RF" & dw$V3=="IBD" & dw$V7=="AUC",]

nrow(t)
t.test(t$diff)


t<-dw[dw$V4 %in% c("ADASYN.5") & 
        dw$V5 == 50 &  
        dw$V9=="RF" & dw$V3=="IBD" & dw$V7=="AUC",]

nrow(t)
t.test(t$diff)


t<-dw[dw$V4 %in% c("SMOTE.5") & 
        dw$V5 == 50 &  
        dw$V9=="RF" & dw$V3=="BMI category" & dw$V7=="AUC",]

nrow(t)
t.test(t$diff)


t<-dw[dw$V4 %in% c("ADASYN.5") & 
        dw$V5 == 50 &  
        dw$V9=="RF" & dw$V3=="BMI category" & dw$V7=="AUC",]

nrow(t)
t.test(t$diff)

