# 打开输入文件和输出文件
with open('data_all.txt', 'r') as input_file, open('data2.txt', 'w') as output_file:
    for line in input_file:
        # 分割命令行，以"|"和"&&"为分隔符
        commands = line.strip().split('|') if '|' in line else line.strip().split('&&')
        
        # 遍历分割后的命令
        for command in commands:
            # 去除首尾空格
            command = command.strip()
            
            # 如果命令非空，写入到输出文件
            if command:
                output_file.write(command + '\n')

