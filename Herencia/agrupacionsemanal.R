## install.packages("epictools")
library(epitools) 

# Cargar ubicación de datos
#setwd("/media/a41618b0-11e2-4046-b29d-8081b0d65296/Lo-ultimo/SerieTemporal_Rn222/scripts/BoxPlot/fast_load")
setwd("/media/cardenas/KINGSTON/Lo-ultimo/TimeSeriesWork/SerieTemporalRn222/scripts/BoxPlot/fast_load")

## 
# setwd("E:/Lo-ultimo/TimeSeriesWork/SerieTemporalRn222/scripts/BoxPlot/fast_load")


# Leer los datos de todos los años de una solo fichero.
# Previamente se ha hecho un cat para agruparlos.
datos_raw<-read.csv("datos.csv", header=F, dec=".",sep=",", stringsAsFactors=F)

str(datos_raw)
datos_raw$V1
datos_raw$V2
length(datos_raw$V2)
tail(datos_raw$V2)

# Extraer semanas
semanas<- as.week(datos_raw$V1, format="%d/%m/%Y")
meses <- as.month(datos_raw$V1, format="%d/%m/%Y")

semanas
meses

# Extraer el número de semana en formato universal
semanas$stratum

meses$stratum2
meses$cmon

# semanas$week
# weekdays(fechas)
# week(fechas)
# head(fechas)

# Se forma un data.frame con 
# la fecha en formato humano, 
# en formato universal y
# el nivel del radon
datos <- data.frame( 
                     "nWeek"=semanas$stratum,
                     "idWeek"=semanas$week,
                     "Rn"= datos_raw$V2)
datos

summary(datos)
str(datos)

# Grafico del boxplot
boxplot(Rn~nWeek,data=datos)

boxplot(Rn~idWeek, data=datos)

# Se obtiene la mediana del Rn para cada semana. 
r1<-with(datos, tapply(Rn, nWeek, median))

# r1[which(r1==-1)] # identificación de las semanas con mediana -1 (semanas sin valores)

r1[which(r1==-1)]<- mean(r1) # relleno de las medianas con -1 por la media de todos los valores

# salvar las medianas 
medianasRn<- as.numeric(r1)
medianasRn
length(medianasRn)

paste(as.character(medianasRn), collapse=", ")

# Escribir los datos de salida para despues hacer los boxplot semanales.
write.table(datos, file="../Rn_weekly_2.txt", quote = FALSE, sep = ",", col.names = T, row.names=FALSE)

##

P = abs(2*fft(medianasRn)/100)^2

P= abs(fft(medianasRn))
abs(P)
f=1:(length(medianasRn))/length(medianasRn)
length(f)
length(medianasRn)
plot(f, abs(P), type='o', xlab = 'frequencia', ylab='periograma', ylim=c(0,2000), xlim = c(0.001,0.09)) 
plot(f, log(P), type='o', xlab = 'frequencia', ylab='periograma') 

1/0.023



ssp<- spectrum(medianasRn)
plot(ssp)


f = 0:100/100
plot(f, P[1:101], type='o', xlab = 'frequencia', ylab='periograma') 
plot(f, log(P[1:101]), type='o', xlab = 'frequencia', ylab='periograma') 

# plot.ts(P)
# plot.ts(log(P))

# Repetimos el analisis con las medianas mensuales
medianasmensualesRn<-c(87.0, 99.0, 87.0, 79.0, 73.0, 63.0, 
                     63.0, 76.5, 80.0, 92.0, 83.0, 76.0, 
                     74.0, 87.0, 88.0, 82.0, 74.0, 63.0, 
                     67.0, 67.0, 68.0, 75.0, 78.0, 96.0, 
                     86.0, 110.0, 95.0, 80.0, 73.0, 60.0, 
                     75.0, 73.0, 72.0, 71.0, 79.0, 97.0, 
                     79.0, 68.0, 77.0, 83.0, 74.0, 73.0, 
                     65.0, 77.0, 75.0, 71.0, 84.0, 88.0, 
                     73.0, 82.0, 91.0, 88.0, 63.0, 70.0, 
                     71.0, 66.0, 73.0, 80.0, 83.0, 99.0)
plot(medianasmensualesRn)

P= abs(fft(medianasmensualesRn))

f=1:(length(medianasmensualesRn))/length(medianasmensualesRn)
length(f)
length(medianasmensualesRn)
plot(f, abs(P), type='o', xlab = 'frequencia', ylab='periograma', ylim=c(0,350), xlim = c(0.05,0.5)) 

1/0.12

del<-1 # sampling interval
x<- medianasmensualesRn
x<- medianasRn
x.spec <- spectrum(x,span=10,plot=FALSE)
spx <- x.spec$freq/del
spy <- 2*x.spec$spec
plot(spy~spx,xlab="frequency",ylab="spectral density",type="l")


spectrum(medianasmensualesRn)
spectrum(medianasRn)

P = abs(2*fft(medianasmensualesRn)/100)^2
P= abs(fft(medianasmensualesRn))
plot(P, ylim=c(0,400))
P

ssp<- spectrum(medianasmensualesRn)
plot(ssp)

print(sort(ssp$spec ))


f = 0:50/100
plot(f, P[1:51], type='o', xlab = 'frequencia', ylab='periograma') 

# algunas pruebas con datos aleatorios.
datos<-seq(1, 1000, by = 0.5)
plot(datos)
datos <- datos[sample(1:length(datos)) ]
datos
plot (datos)

P = abs(2*fft(datos)/100)^2

f = 0:50/100
plot(f, P[1:51], type='o', xlab = 'frequencia', ylab='periograma') 



########
x<- seq(-3,3,length=20)
x

y<-cbind(n=dnorm(x),t=dt(x,df=10))
y

library(spatialEco)

kl.divergence(y)

kl.divergence(y[,1:2])[3:3]

y
y[,1:2]
