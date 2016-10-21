f1= open("train_libfm",'r');
f2 = open("ua.train_sbpmf",'w')
f3= open("test_libfm",'r');
f4 = open("ua.test_sbpmf",'w');

for line in f1:
    c = line.strip().split(' ');
    a = c[1].split(':')
    a = a[0]
    b = c[2].split(':')
    b = b[0]
    f2.write(a + '\t' + b + '\t' + c[0] + '\n');

for line in f3:
    c = line.strip().split(' ');
    a = c[1].split(':')
    a = a[0]
    b = c[2].split(':')
    b = b[0]
    f4.write(a + '\t' + b + '\t' + c[0] + '\n');

f1.close();
f2.close();
f3.close();
f4.close();
