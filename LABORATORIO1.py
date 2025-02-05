
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.io import loadmat
from math import sqrt

# por que no me busca el archivo?
#sera las librerias? 
x=loadmat('100m.mat')

#esto es para mostrar los valores que necesitamos
#se estandarizado en mV la señal 
ecg=(x['val']-0)/200
ecg=np.transpose(ecg)

fs=360
tiem_muest=1/fs
t=np.linspace(0,np.size(ecg),np.size(ecg))*tiem_muest

# Graficar la señal ECG con mejoras en la visualización
plt.figure(figsize=(10, 5))  # Tamaño más grande para mejor visualización
plt.plot(t, ecg, color='blue', linewidth=1.2, label="Señal ECG")  # Línea azul y más gruesa
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')  # Etiqueta del eje X
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')  # Etiqueta del eje Y
plt.title("Electrocardiograma (ECG)", fontsize=14, fontweight='bold', color='darkred')  # Título en negrita y color
plt.grid(True, linestyle="--", alpha=0.6)  # Agregar cuadrícula con línea punteada
plt.legend(loc="upper right")  # Agregar leyenda
plt.show()
#plt.plot(t,ecg)
#plt.show()


#histograma
# librerias que vamos a usar
#import pandas as pd contiene funciones que nos ayudan en el analisis de datos
#import matplotlib.pyplot as plt
#plt.hist(ecg.flatten(), 15, color="yellow", ec="black")
plt.hist(ecg.flatten(), bins=50, color="yellow", ec="black")
plt.xlabel("Valores del ECG (mV)", fontsize=12,fontweight='bold')
plt.ylabel("Frecuencia", fontsize=12,fontweight='bold')
plt.title("Histograma del ECG", fontsize=14, fontweight='bold', color='black')
plt.show()

def desviacion_estandar(valores, media):
    suma = 0
    for valor in valores:
        suma += (valor - media) **2
        
    radicando = suma /(len(valores)-1)
    
    return sqrt(radicando)

def calcular_media(valores):
    suma = 0
    for valor in valores:
        suma +=valor
        
    return suma /len(valores) 

media = calcular_media(ecg.flatten())

resultados = desviacion_estandar(ecg.flatten(), media)

coeficiente_de_variacion = (resultados/media)
print('La desviación estandar es: {}'.format(resultados))
print('La media es: {}'.format(media))
print('El coeficiente de variación es: {}'.format(coeficiente_de_variacion))
print('El tipo de distribución del histograma es sesgado hacia la izquierda')

media_con_func = np.mean(ecg.flatten())
desviacion_estan_func = np.std(ecg.flatten())
print ('La media utilizando funciones de librerias es: {}'.format(media_con_func))
print ('La desviación estandar utilizando funciones de librerias es: {}'.format(desviacion_estan_func))




# Añadir ruido gaussiano (Alta frecuencia)
frecuencia1=1000
desviacion_estandar_ecg = np.std(ecg.flatten())  # Desviación estándar de la señal
ruido = np.random.normal(0, desviacion_estandar_ecg, ecg.shape)+ 0.5 * np.sin(2 * np.pi * frecuencia1 * t).reshape(ecg.shape)
ecg_ruidoso = ecg + ruido


# Graficar la señal con ruido gaussiano
plt.figure(figsize=(10, 5))
plt.plot(t, ecg_ruidoso, color='darkred', linewidth=1.2, label="Señal ECG con Ruido Gaussiano")
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')
plt.title("ECG Ruido Gaussiano", fontsize=14, fontweight='bold', color='black')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.show()

# Calcular el SNR del ruido gaussiano
E_s2 = np.mean(ecg**2)  # Energía de la señal (E[s^2])
o_2 = np.var(ruido)  # Varianza del ruido
N = len(ecg.flatten())  # Número de muestras
SNR = E_s2 / (o_2 * N)

print('El SNR de la señal con ruido gaussiano en alta frecuencia es: {} dB'.format(SNR))




# Añadir ruido gaussiano (Baja frecuencia)
frecuencia1=50
desviacion_estandar_ecg = np.std(ecg.flatten())  # Desviación estándar de la señal
ruido = np.random.normal(0, desviacion_estandar_ecg, ecg.shape)+ 0.5 * np.sin(2 * np.pi * frecuencia1 * t).reshape(ecg.shape)
ecg_ruidoso = ecg + ruido


# Graficar la señal con ruido gaussiano
plt.figure(figsize=(10, 5))
plt.plot(t, ecg_ruidoso, color='purple', linewidth=1.2, label="Señal ECG con Ruido Gaussiano")
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')
plt.title("ECG Ruido Gaussiano (100Hz)", fontsize=14, fontweight='bold', color='black')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.show()

# Calcular el SNR del ruido gaussiano
E_s2 = np.mean(ecg**2)  # Energía de la señal (E[s^2])
o_2 = np.var(ruido)  # Varianza del ruido
N = len(ecg.flatten())  # Número de muestras
SNR = E_s2 / (o_2 * N)

print('El SNR de la señal con ruido gaussiano en baja frecuencia es: {} dB'.format(SNR))





# Añadir ruido impulsivo
prob = 0.05  # Probabilidad de impulsos
amplitud = 5 * np.std(ecg)  # Ajustar amplitud del ruido
ruido_impulsivo = (np.random.rand(*ecg.shape) < prob) * (2 * amplitud * (np.random.rand(*ecg.shape) - 0.5))
ecg_ruidoso_impulso = ecg + ruido_impulsivo

# Graficar la señal con ruido impulsivo
plt.figure(figsize=(10, 5))
plt.plot(t, ecg_ruidoso_impulso, color='green', linewidth=1.2, label="Señal ECG con Ruido Impulsivo")
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')
plt.title("ECG con Ruido Impulsivo", fontsize=14, fontweight='bold', color='black')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.show()

# Calcular el SNR ruido impulso
E_s2_imp = np.mean(ecg**2)  # Energía de la señal (E[s^2])
o_2_imp = np.var(ruido_impulsivo)  # Varianza del ruido
SNR_impulso = E_s2_imp / (o_2_imp * N)

print('El SNR de la señal con ruido impulso de alta frecuencia es: {} dB'.format(SNR_impulso))




# Añadir ruido impulsivo (Baja frecuencia)
prob = 0.05  # Probabilidad de impulsos
amplitud = 5 * np.std(ecg)  # Ajustar amplitud del ruido
ruido_impulsivo = (np.random.rand(*ecg.shape) < prob) * (0.9 * amplitud * (np.random.rand(*ecg.shape) - 0.5))
ecg_ruidoso_impulso = ecg + ruido_impulsivo

# Graficar la señal con ruido impulsivo
plt.figure(figsize=(10, 5))
plt.plot(t, ecg_ruidoso_impulso, color='brown', linewidth=1.2, label="Señal ECG con Ruido Impulsivo")
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')
plt.title("ECG con Ruido Impulsivo", fontsize=14, fontweight='bold', color='black')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.show()

# Calcular el SNR ruido impulso
E_s2_imp = np.mean(ecg**2)  # Energía de la señal (E[s^2])
o_2_imp = np.var(ruido_impulsivo)  # Varianza del ruido
SNR_impulso = E_s2_imp / (o_2_imp * N)

print('El SNR de la señal con ruido impulso de baja frecuencia es: {} dB'.format(SNR_impulso))





# hacemos el ruido tipo artefacto (Alta frecuencia)
ruido_artefacto = np.random.normal(0, desviacion_estandar_ecg, ecg.shape) +  1* np.sin(2 * np.pi * 50* t).reshape(ecg.shape)
ecg_ruido_artefacto= ruido_artefacto + ecg

# Graficamos la señal con el ruido artefacto
plt.figure(figsize=(10, 5))
plt.plot(t,ecg_ruido_artefacto, color='blue', linewidth=1.2, label="Señal ECG con Ruido artefacto")
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')
plt.title("ECG con Ruido artefacto", fontsize=14, fontweight='bold', color='black')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.show()

# Calcular el SNR, ruido artefacto
E_s2_artefacto = np.mean(ecg**2)  # Energía de la señal (E[s^2])
o_2_artefacto = np.var(ruido_artefacto)  # Varianza del ruido
SNR_artefacto = E_s2_artefacto / (o_2_artefacto * N)

# mostrar el SNR, ruido artefacto
print("El SNR de la señal con ruido artefacto en alta frecuencia es: {} dB".format(SNR_artefacto))



# hacemos el ruido tipo artefacto (Baja frecuencia)
ruido_artefacto = np.random.normal(0, desviacion_estandar_ecg, ecg.shape) + 0.05 * np.sin(2 * np.pi * 50* t).reshape(ecg.shape)
ecg_ruido_artefacto= ruido_artefacto + ecg

# Graficamos la señal con el ruido artefacto
plt.figure(figsize=(10, 5))
plt.plot(t,ecg_ruido_artefacto, color='orange', linewidth=1.2, label="Señal ECG con Ruido artefacto")
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')
plt.title("ECG con Ruido artefacto", fontsize=14, fontweight='bold', color='black')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.show()

# Calcular el SNR, ruido artefacto
E_s2_artefacto = np.mean(ecg**2)  # Energía de la señal (E[s^2])
o_2_artefacto = np.var(ruido_artefacto)  # Varianza del ruido
SNR_artefacto = E_s2_artefacto / (o_2_artefacto * N)

# mostrar el SNR, ruido artefacto
print("El SNR de la señal con ruido artefacto en baja frecuencia es: {} dB".format(SNR_artefacto))

