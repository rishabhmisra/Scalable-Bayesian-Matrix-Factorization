f1= open("ua.train",'r');
f2 = open("train_sbpmf",'w')
f3= open("ua.test",'r');
f4 = open("test_sbpmf",'w');

for line in f1:
    c = line.strip().split('\t');
    f2.write(str(int(c[0])-1) + '\t' + str(int(c[1])-1) + '\t' + c[2] + '\n');

for line in f3:
    c = line.strip().split('\t');
    f4.write(str(int(c[0])-1) + '\t' + str(int(c[1])-1) + '\t' + c[2] + '\n');

f1.close();
f2.close();
f3.close();
f4.close();
