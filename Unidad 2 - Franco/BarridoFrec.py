from lantz import MessageBasedDriver, Feat, ureg
from lantz.core import mfeats


class Osc(MessageBasedDriver):
    
    MANUFACTURER_ID = '0x0699'
    MODEL_CODE = '0x0363'
    
    @Feat(read_once=True)
    def idn(self):
        return self.query('*IDN?')
    #set_query = MessageBasedDriver.write
    timebase = mfeats.QuantityFeat('HOR:MAIN:SCA?','HOR:DEL:SCA {}',units = 's', limits = (0.01,100))
    
    @Feat(units = 'Hz')
    def frec(self):
        return self.query('MEASU:MEAS{}:VAL?'.format(2))
    
    
class Gen(MessageBasedDriver):    
    
    MANUFACTURER_ID = '0x0699'
    MODEL_CODE = '0x0346'

    @Feat(read_once=True)
    def idn(self):
        return self.query('*IDN?')
    
    #frecuencia = mfeats.QuantityFeat('SOUR{1}:FUNC:FREQ?'.format(1),'SOUR{}:FREQ{}'.format(1,),units = 'Hz', limits = (1,1E6))
    set_query = MessageBasedDriver.write
    frecgen = mfeats.QuantityFeat('SOUR1:FUNC:FREQ?','SOUR1:FREQ{}',units = 'Hz', limits = (1,1E6))
    

with Gen.via_usb('C034166') as gene:  
    print(gene.idn)
    gene.frecgen = 1000
    
    
with Osc.via_usb('C108011') as osci:
    print(osci.idn)
    print(osci.timebase)
    #osci.timebase = 0.5 *ureg.seconds
    print(osci.frec)
    
