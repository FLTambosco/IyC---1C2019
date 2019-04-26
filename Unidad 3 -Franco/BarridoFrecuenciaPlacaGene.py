# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:40:46 2019

@author: Franco
"""

try:
    from lantz import MessageBasedDriver, Feat, ureg
    from lantz.core import mfeats
    import time
    import pyaudio
    import wave
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
except: 
    print('Ups! una librería no se importó :(')



class Generador(MessageBasedDriver):
    
    @Feat(read_once=True)
    def i_generador(self):
        return self.query('*IDN?')
    
    f_generador = mfeats.QuantityFeat('SOUR1:FUNC:FREQ?', 'SOUR1:FREQ:FIX {}', units='hertz', limits=(1, 1E6)) #Get-Set Frecuencia

    a_generador = mfeats.QuantityFeat('SOUR1:VOLTage:AMPLitude?', 'SOUR1:VOLT:AMPLitude {}',units = 'volt') #Get-Set Amplitud
    
    
    
class Osciloscopio(MessageBasedDriver):
    
    @Feat(read_once=True)
    def i_osciloscopio(self):
        return self.query('*IDN?')
    
    bt_osciloscopio = mfeats.QuantityFeat('HOR:MAIN:SCA?','HOR:DEL:SCA {}',units = 's', limits = (0.01,100))
    
    @Feat(units = 'Hz')
    def get_frec(self):
        return self.query('MEASU:MEAS{}:VAL?'.format(2))
    
    @Feat(units = 'volts')
    def set_scale(self):
        self.write('CH1:SCA {}') #No sé si es así, preg desp!!!
    


def Seno(t,A,B,W):
    return A * np.sin(t*W) + B




Fs = [1,2,3,4,5]
A = 1

Fenv = [] 
Frec = []




#Esto es para grabar el audio:
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5


for k in range(len(Fs)):

    with Generador.via_usb('') as G:
        try:
            G.f_generador= Fs[k] * ureg.hertz
            G.a_generador= A * ureg.volt
        except: 
            print('Hubo un problema con el ajuste de parametros del Generador')
    
    with Osciloscopio.via_usb('') as O:
        try:
            O.bt_osciloscopio= 5 * (1/Fs[k]) * ureg.seconds #Le pongo una base como para que agarre 5 picos (creo)     
            O.set_scale = (A + A/2) * ureg.volt #NECESITO LA FUNCION PARA PONERLE LA ESCALA DE AMPLITUD
        except: 
            print('Hubo un problema con el ajuste de parametros del Osciloscopio')  
        
    Fenv.append(O.get_frec)
    

#Ahora voy a grabar:
        
    WAVE_OUTPUT_FILENAME = "SeñalRecibida.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Grabando...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Se ha grabado el audio con exito")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
#Ahora voy a sacar los parametros:

    file = 'SeñalRecibida.wav'

    wav_file = wave.open(file,'r')

#Extract Raw Audio from Wav File
    signal = wav_file.readframes(-1)
    if wav_file.getsampwidth() == 1:
        signal = np.array(np.frombuffer(signal, dtype='UInt8')-128, dtype='Int8')
    elif wav_file.getsampwidth() == 2:
        signal = np.frombuffer(signal, dtype='Int16')
    else:
        raise RuntimeError("Unsupported sample width")

# http://schlameel.com/2017/06/09/interleaving-and-de-interleaving-data-with-python/
    deinterleaved = [signal[idx::wav_file.getnchannels()] for idx in range(wav_file.getnchannels())] #Creo que aca esta la información de todos los channels, voy a agarrar el primero, pero puede ser que este mal 

#Get time from indices
    fs = wav_file.getframerate()
    DominioAudio=np.linspace(0, len(signal)/wav_file.getnchannels()/fs, num=len(signal)/wav_file.getnchannels())    
  
    
#Voy a savcar los parametros de la señal recibida: 

    p0 = [A,Fs[i],0.]

    popt,pcov = curve_fit(Seno, DominioAudio, deinterleaved[1] , p0=p0, absolute_sigma=True, sigma= 0.002)
    perr = np.sqrt(np.diag(pcov))    
    
    Frec.append(popt[1])
    
    
#Plotteo:
    
plt.style.use('ggplot')  

plt.plot(Fenv, Frec,'g.',linewidth=1.0, label="Datos Medidos")
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

plt.set_xlabel('Frecuencia Enviada (Hz)', size=20)
plt.set_ylabel('Frecuencia Recibida (Hz)',size=20)
plt.legend()

plt.show()
    
    
    
    
    
    
    
    
    
    
    
#    
#with Gen.via_usb('C034165') as gene:
#    print(gene.idn)
#    gene.amplitud_gen= 0.5 * ureg.volt
#    gene.frec_gen = 300 * ureg.hertz
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
'''
- Generador manda una señal : f1, A
- Osciloscopio se ajusta a A y a f1
- Osciloscopio agarra frecuencia y amplitud y lo pone en un vector
- Placa de audio agarra señal
- Estaría bueno que se pueda ajustar una senoidal y me de la frecuencia y amplitud
- Repetir
'''