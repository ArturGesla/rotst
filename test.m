a=importdata('out')
x=a(:,1)
y=a(:,2)
z=a(:,3)
A=[];
for i=1:length(x)
    A(x(i)+1,y(i)+1)=z(i);
end
rank(A)

%%
a=importdata('out');
A=a;
rank(A)
