# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
from lantz import MessageBasedDriver
#from lantz import Feat
from lantz.core import Feat
#(Osciloscopio,Generador)
#('USB0::0x0699::0x0363::C108011::INSTR', 'USB0::0x0699::0x0346::C034166::INSTR')

'''
class TDS1002b(MessageBasedDriver):

    MANUFACTURER_ID = '0x699'
    MODEL_CODE = '0x363'

    @Feat(read_once=True)
    def idn(self):
        return self.query('*IDN?')
'''

'''
class Osci(MessageBasedDriver):
    amplitud = mfeats.QuantityFeat()
'''

class Osc(MessageBasedDriver):
    
    MANUFACTURER_ID = '0x0699'
    MODEL_CODE = '0x0363'
    
    @Feat(read_once=True)
    def idn(self):
        return self.query('*IDN?')
    
    @Feat()
    def parametros(self):
        return self.resource.query_ascii_values('WFMPRE:XZE?;XIN?;', separator=';' )
    
    @Feat()
    def timebase(self):
        return 1111
    
    @timebase.setter
    def timebase(self, seconds):
        self.write('HOR:DEL:SCA {}'.format(seconds))
        

import visa
visa.log_to_screen()
    
with Osc.via_usb('C108011') as osci:
    print(osci.idn)
    print(osci.parametros)
    print(osci.idn)
    osci.timebase = 28
    t = osci.timebase
    