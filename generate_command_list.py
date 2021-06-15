import json

objects = ['red', 'blue', 'green', 'yellow']

with open('commands.json') as command_file:
    commands = json.load(command_file)

command_list_file = open('full_command_list.txt', 'w')
count = 0
for command in commands:
    for syntax in commands[command]['syntaxs']:
        if syntax['obj1']:
            for obj1 in objects:
                for prep in syntax['prep']:
                    if syntax['obj2']:
                        for obj2 in objects:
                            if obj1 != obj2:
                                command_list_file.write(commands[command]['name'] + ' ' +obj1 + ' ' + prep + ' ' + obj2 + '\n')
                                count += 1
                    else:
                        command_list_file.write(commands[command]['name'] + ' ' + obj1 + ' ' + prep + '\n')
                        count += 1


        else:
            for obj1 in objects:
                for prep in syntax['prep']:
                    if syntax['obj2']:
                        for obj2 in objects:
                            command_list_file.write(commands[command]['name'] + ' ' + prep + ' ' + obj2 + '\n')
                            count += 1
                    else:
                        command_list_file.write(commands[command]['name'] + ' ' + prep + '\n')
                        count += 1

print("Nb of commands :", count)
command_list_file.close()
command_file.close()