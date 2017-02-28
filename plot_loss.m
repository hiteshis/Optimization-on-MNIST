clear all;
clc;
fid=fopen('avg_cost_SGD.txt');
gid = fopen('computations.txt')
s=textscan(fid,'%f');
t=textscan(gid,'%d');
fclose(fid);
fclose(gid);
y1=s{1};
x1=t{1};
%x = 500*50*(1:length(y1));

plot(x1,y1);
hold on