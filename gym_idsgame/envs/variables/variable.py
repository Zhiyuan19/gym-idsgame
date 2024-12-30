import configparser
import os

class VariableConfig:
    """
    Game constants with .ini file persistence
    """
    def __init__(self, filepath='config.ini'):
        #self.filepath = filepath
        self.filepath = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(self.filepath)
        self.priority = int(self.config.get('GameVariables', 'priority', fallback=1))
        self.protocol = self.config.get('GameVariables', 'protocol', fallback='TCP')
        self.h3counter_1_priority = int(self.config.get('GameVariables', 'h3counter_1_priority', fallback=0))
        self.h3counter_2_priority = int(self.config.get('GameVariables', 'h3counter_2_priority', fallback=0))
        self.h3counter_3_priority = int(self.config.get('GameVariables', 'h3counter_3_priority', fallback=0))
        self.h3counter_4_priority = int(self.config.get('GameVariables', 'h3counter_4_priority', fallback=0))
        self.fwcounter_1_priority = int(self.config.get('GameVariables', 'fwcounter_1_priority', fallback=0))
        self.fwcounter_2_priority = int(self.config.get('GameVariables', 'fwcounter_2_priority', fallback=0))
        self.fwcounter_3_priority = int(self.config.get('GameVariables', 'fwcounter_3_priority', fallback=0))
        self.fwcounter_4_priority = int(self.config.get('GameVariables', 'fwcounter_4_priority', fallback=0))
        self.wscounter_1_priority = int(self.config.get('GameVariables', 'wscounter_1_priority', fallback=0))
        self.wscounter_2_priority = int(self.config.get('GameVariables', 'wscounter_2_priority', fallback=0))
        self.wscounter_3_priority = int(self.config.get('GameVariables', 'wscounter_3_priority', fallback=0))
        self.wscounter_4_priority = int(self.config.get('GameVariables', 'wscounter_4_priority', fallback=0))
        self.macounter_1_priority = int(self.config.get('GameVariables', 'macounter_1_priority', fallback=0))
        self.macounter_2_priority = int(self.config.get('GameVariables', 'macounter_2_priority', fallback=0))
        self.macounter_3_priority = int(self.config.get('GameVariables', 'macounter_3_priority', fallback=0))
        self.macounter_4_priority = int(self.config.get('GameVariables', 'macounter_4_priority', fallback=0))

    def save(self):
        self.config.set('GameVariables', 'priority', str(self.priority))
        self.config.set('GameVariables', 'protocol', self.protocol)
        self.config.set('GameVariables', 'h3counter_1_priority', str(self.h3counter_1_priority))
        self.config.set('GameVariables', 'h3counter_2_priority', str(self.h3counter_2_priority))
        self.config.set('GameVariables', 'h3counter_3_priority', str(self.h3counter_3_priority))
        self.config.set('GameVariables', 'h3counter_4_priority', str(self.h3counter_4_priority))
        self.config.set('GameVariables', 'fwcounter_1_priority', str(self.fwcounter_1_priority))
        self.config.set('GameVariables', 'fwcounter_2_priority', str(self.fwcounter_2_priority))
        self.config.set('GameVariables', 'fwcounter_3_priority', str(self.fwcounter_3_priority))
        self.config.set('GameVariables', 'fwcounter_4_priority', str(self.fwcounter_4_priority))
        self.config.set('GameVariables', 'wscounter_1_priority', str(self.wscounter_1_priority))
        self.config.set('GameVariables', 'wscounter_2_priority', str(self.wscounter_2_priority))
        self.config.set('GameVariables', 'wscounter_3_priority', str(self.wscounter_3_priority))
        self.config.set('GameVariables', 'wscounter_4_priority', str(self.wscounter_4_priority))
        self.config.set('GameVariables', 'macounter_1_priority', str(self.macounter_1_priority))
        self.config.set('GameVariables', 'macounter_2_priority', str(self.macounter_2_priority))
        self.config.set('GameVariables', 'macounter_3_priority', str(self.macounter_3_priority))
        self.config.set('GameVariables', 'macounter_4_priority', str(self.macounter_4_priority))
        with open(self.filepath, 'w') as configfile:
            self.config.write(configfile)
        
        #print("new ones have been saved!")


