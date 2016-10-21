f1= open("sa.train_libfm",'r');
f2 = open("train_sbpmf",'w')
f3= open("sa.test_libfm",'r');
f4 = open("test_sbpmf",'w');

for line in f1:
    c = line.strip().split(' ');
    a = c[1].split(':')
    a = a[0]
    b = c[2].split(':')
    b = str(int(b[0]) - 6040)
    f2.write(a + '\t' + b + '\t' + c[0] + '\n');

for line in f3:
    c = line.strip().split(' ');
    a = c[1].split(':')
    a = a[0]
    b = c[2].split(':')
    b = str(int(b[0]) - 6040)
    f4.write(a + '\t' + b + '\t' + c[0] + '\n');

f1.close();
f2.close();
f3.close();
f4.close();
