clc; clear;
a=importdata('out')
x=a(:,1)
y=a(:,2)
z=a(:,3)
A=[];
for i=1:length(x)
    A(x(i)+1,y(i)+1)=z(i);
end
display("hello")
rank(A)
B=A;
%
% a=importdata('out');
% A=a;
% rank(A)


%%
iz=3; 
ix=1;

Nz=5;
Nx=5;
Neq=4;
ieq=0;
 ii = (iz + ix * Nz) * Neq + ieq;

                 iif = ii - ieq;
                 iifxp = iif + Nz * Neq;
                 iifxm = iif - Nz * Neq;
                 iifzp = iif + Neq;
                 iifzm = iif - Neq;

                 iig = iif + 1;
                 iigxp = iig + Nz * Neq;
                 iigxm = iig - Nz * Neq;
                 iigzp = iig + Neq;
                 iigzm = iig - Neq;

                 iih = iig + 1;
                 iihxp = iih + Nz * Neq;
                 iihxm = iih - Nz * Neq;
                 iihzp = iih + Neq;
                 iihzm = iih - Neq;

                 iip = iih + 1;
                 iipxp = iip + Nz * Neq;
                 iipxm = iip - Nz * Neq;
                 iipzp = iip + Neq;
                 iipzm = iip - Neq;
display("lol");
                 rank(B)
                  B=[B;zeros(length(A),1)'];
                  B(end,iih+1)=1;
                  %B(end,iipzp+1)=1;
                  %B(end,iipzm+1)=-1;
                  rank(B)
               