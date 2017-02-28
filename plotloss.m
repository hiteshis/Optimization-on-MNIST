clear all;
clc;
fid=fopen('avg_cost_SVRGD.txt');
gid=fopen('gradients.txt');
s=textscan(fid,'%f');
t = textscan(gid, '%d');
fclose(fid);
fclose(gid);
y1=s{1}
x1 = t{1}
x1 = x1./50000
%x=[1:length(y1)];

plot(x1,y1,'b');
hold on

