# ANÁLISIS DE SEÑALES ECG CON RUIDO

Este repositorio presenta el proceso y explicación del código, de una señal fisiológica obtenida de PhysioNet, importada y visualizada en Python por medio del compilador Spyder, cálculando datos estadísticos descriptivos tales como la media, desviación estándar, coeficiente de variación e histogramas y comparando los resultados obtenidos manualmente y con funciones predefinidas. Posteriormente, se analiza la relación señal-ruido al contaminar la señal con ruido gaussiano, ruido impulsivo y de tipo artefacto y su impacto.

## Uso de librerias
```
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.io import loadmat
from math import sqrt
```
`matplotlib.pyplot` Se usa para poder graficar la señal ECG y su histograma.
`numpy (np)` Permite realizar las operaciones matemáticas.
`scipy.io  loadmat` Esta nos permite cargar archivos de MATLAB, en este caso nuestra señal fue extraida de la base de datos Physionet en formato `.mat`.
`math sqrt` Se usa para calcular las raices cuadradas.

## Carga de la señal

```
x=loadmat('100m.mat')
ecg=(x['val']-0)/200
ecg=np.transpose(ecg)
```
`loadmat('100m.mat')` Es un archivo de tipo `.mat` que es donde se encuentra cargada la señal a utilizar.
`(x['val']-0)/200` Se usa para extraer los datos de la señal, y esta se divide en 200 para convertirla en (mV).
`np.transpose(ecg)` Ajusta la disposición de los datos, para que estos se alineen adecuadamente.

```
fs=360
tiem_muest=1/fs
t=np.linspace(0,np.size(ecg),np.size(ecg))*tiem_muest
```
`fs=360` Hace referencia a la frecuencia a la que la señal es muestreada.
`tiem_muest=1/fs` Calcula el tiempo que hay entre una muestra y otra
`np.linspace(0,np.size(ecg),np.size(ecg))*tiem_muest` Esta linea genera un vector de tiempo en segundos.

## Gráfica de la señal
```
plt.figure(figsize=(10, 5))  # Tamaño más grande para mejor visualización
plt.plot(t, ecg, color='blue', linewidth=1.2, label="Señal ECG")  # Línea azul y más gruesa
plt.xlabel("Tiempo (s)", fontsize=12, fontweight='bold')  # Etiqueta del eje X
plt.ylabel("Amplitud (mV)", fontsize=12, fontweight='bold')  # Etiqueta del eje Y
plt.title("Electrocardiograma (ECG)", fontsize=14, fontweight='bold', color='darkred')  # Título en negrita y color
plt.grid(True, linestyle="--", alpha=0.6)  # Agregar cuadrícula con línea punteada
plt.legend(loc="upper right")  # Agregar leyenda
plt.show()
```
<img width="620" alt="Figure 2025-02-04 232930" src="https://github.com/user-attachments/assets/2bef727c-3043-4dae-959c-0a53e8d33c20" />


## Histograma de la señal
```
plt.hist(ecg.flatten(), bins=50, color="yellow", ec="black")
plt.xlabel("Valores del ECG (mV)", fontsize=12,fontweight='bold')
plt.ylabel("Frecuencia", fontsize=12,fontweight='bold')
plt.title("Histograma del ECG", fontsize=14, fontweight='bold', color='black')
plt.show()
```
`plt.hist(ecg.flatten(), bins=50, color="yellow", ec="black")`Aqui se grafica un histograma de los valores del ECG, con 50 divisiones o barras que hace referencia a `bins=50`.

<img width="391" alt="Figure 2025-02-04 232922" src="https://github.com/user-attachments/assets/da141738-1294-4ed0-8f8b-3a3a2bce302c" />


## Cálculo de Media y Desviación Estándar
```
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
```
Se calcula la media obteniendo el promedio de la señal,  la desviación estándar usando la fórmula estadística y el coeficiente de variación, además se muestran los resultados obtenidos.

## Cálculo con función

```
media_con_func = np.mean(ecg.flatten())
desviacion_estan_func = np.std(ecg.flatten())
print ('La media utilizando funciones de librerias es: {}'.format(media_con_func))
print ('La desviación estandar utilizando funciones de librerias es: {}'.format(desviacion_estan_func))
```

## Añadir ruido y calcular SNR
```
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

<img width="620" alt="Figure 2025-02-04 232915" src="https://github.com/user-attachments/assets/187e0c00-578b-4e4b-8cf3-698282dcd367" />



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
```
<img width="620" alt="Figure 2025-02-04 232954" src="https://github.com/user-attachments/assets/8867165b-584d-4cb7-9209-8d8b7c8e66a1" />


Principalmente se genera un ruido gaussiano con una desviación estándar muy parecida a la de la señal, se le suma una onda senoidal de 1000 Hz, que hace referencia a la `frecuencia1`.
Luego graficamos la señal que obtuvimos con el ruido, y calculamos el SNR por medio de la formula estadística.
`E_s2`Calcula la energia de la señal
`o_2` Es la varianza del ruido.
`SNR=E_s2/(o_2*N)` Esta fórmula calcula la relación señal-ruido.
Posteriormente se imprime el valor de esta relacion y se hace nuevamente para la baja frecuencia.

## Ruido Impulsivo
```
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

<img width="626" alt="Figure 2025-02-04 232851" src="https://github.com/user-attachments/assets/e3d5ea42-4072-4118-b77a-0f1ad04f8679" />



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
```
<img width="620" alt="Figure 2025-02-04 232842" src="https://github.com/user-attachments/assets/aa48076c-3546-442f-9783-e1953c63821f" />


Para este ruido, se introduce un ruido impulsivo con probabilidad de 5%
`ruido_impulsivo`Genera picos aleatorios en la señal.
Si se modifica el valor de la amplitud, disminuyendo la amplitud del ruido, la señal original se vuelve mas dominante y el SNR aumenta.
Por consiguiente se imprime el valor, y se hace el mismo proceso para la baja frecuencia.

## Ruido Artefacto
```
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

<img width="620" alt="Figure 2025-02-04 232834" src="https://github.com/user-attachments/assets/2b410ea7-c3c6-4182-90b1-469e4a5425f4" />


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

```
<img width="626" alt="Figure 2025-02-04 232812" src="https://github.com/user-attachments/assets/739a3150-fe25-41d3-af80-bb73b41f369d" />

`np.random.normal(0, desviacion_estandar_ecg, ecg.shape)` Se genera ruido gaussiano con media 0 y una desviación estándar igual a la de la señal ECG original.
`1 * np.sin(2 * np.pi * 50 * t).reshape(ecg.shape)` Se añade una señal senoidal de 50 Hz, lo que representa una interferencia de alta frecuencia.
La suma de estos dos términos forma el ruido artefacto.
`ecg_ruido_artefacto = ruido_artefacto + ecg`El ruido generado se suma a la señal original.
Se calcula nuevamente el SNR, se imprime y se repite el proceso para baja frecuencia.
```
`np.random.normal(0, desviacion_estandar_ecg, ecg.shape)` Se genera ruido gaussiano con media 0 y una desviación estándar igual a la de la señal ECG original.
`1 * np.sin(2 * np.pi * 50 * t).reshape(ecg.shape)` Se añade una señal senoidal de 50 Hz, lo que representa una interferencia de alta frecuencia.
La suma de estos dos términos forma el ruido artefacto.
Se calcula nuevamente el SNR, se imprime y se repite el proceso para baja frecuencia.
`ecg_ruido_artefacto = ruido_artefacto + ecg`El ruido generado se suma a la señal original.



## Resultados Impresos de los calculos hechos en el código:

La desviación estandar es: 0.12253942948749433
La media es: -0.20317361111111126
El coeficiente de variación es: -0.603126699463348
El tipo de distribución del histograma es sesgado hacia la izquierda
La media utilizando funciones de librerias es: -0.2031736111111111
La desviación estandar utilizando funciones de librerias es: 0.12252240894022379
El SNR de la señal con ruido gaussiano en alta frecuencia es: 0.00011238064617759935 dB
El SNR de la señal con ruido gaussiano en baja frecuencia es: 0.00011055939379764979 dB
El SNR de la señal con ruido impulso de alta frecuencia es: 0.0025503172624808906 dB
El SNR de la señal con ruido impulso de baja frecuencia es: 0.013968682057630513 dB
El SNR de la señal con ruido artefacto en alta frecuencia es: 3.0434201403664637e-05 dB
El SNR de la señal con ruido artefacto en baja frecuencia es: 0.0009267138968601924 dB


## Bibliografía
-Las imágenes obtenidas en este trabajo fueron generadas a partir del código implementado en Python, utilizando la librería Matplotlib para la visualización de señales fisiológicas. 
-La señal de ECG utilizada en los análisis fue descargada de la base de datos PhysioNet, una plataforma de acceso libre que proporciona registros fisiológicos para investigación biomédica.
[1]	“PhysioBank ATM”, Physionet.org. [En línea]. Disponible en: https://archive.physionet.org/cgi-bin/atm/ATM. 
