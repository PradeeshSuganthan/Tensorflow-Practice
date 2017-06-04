heart = "tf-stanford-tutorials-master/data/heart.txt"


with open(heart, 'r') as file:
	filedata = file.read()


filedata = filedata.replace('1', 1)

filedata = filedata.replace('0', 0)

with open(heart, 'w') as file:
	file.write(filedata)