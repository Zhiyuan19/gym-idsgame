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

    def save(self):
        self.config.set('GameVariables', 'priority', str(self.priority))
        self.config.set('GameVariables', 'protocol', self.protocol)
        with open(self.filepath, 'w') as configfile:
            self.config.write(configfile)
        
        print("new ones have been saved!")


