import utils

command_template = 'python main_Diabetes.py --e {}  --bias_type {} --Diabetes_train_mode {} --Diabetes_test_mode {}'
p1 = ['Diabetes_LNL_II_paper']
p2 = ['II']
p3 = ['eb1', 'eb2']
p4 = ['eb1', 'eb2']

utils.run(command_template, "flexible", 1, p1, p2, p3, p4)

p3 = ['eb1_balanced', 'eb2_balanced']
p4 = ['eb1_balanced', 'eb2_balanced', 'balanced']
utils.run(command_template, "flexible", 1, p1, p2, p3, p4)

p3 = ['eb1_moderate', 'eb2_moderate']
p4 = ['eb1_moderate', 'eb2_moderate']
utils.run(command_template, "flexible", 1, p1, p2, p3, p4)