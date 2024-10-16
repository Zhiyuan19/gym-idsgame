from gym_idsgame.envs.variables import variable
variable_config = variable.VariableConfig()
print("Priority before change:", variable_config.priority)
variable_config.priority += 1  
variable_config.save()  
print("Priority after change:", variable_config.priority)
