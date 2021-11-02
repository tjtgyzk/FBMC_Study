function createfigure(X1, YMatrix1)
%CREATEFIGURE(X1, YMatrix1)
%  X1:  x 数据的向量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 01-Nov-2021 17:07:47 自动生成

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 使用 semilogy 的矩阵输入创建多行
semilogy1 = semilogy(X1,YMatrix1,'LineWidth',1,'Parent',axes1);
set(semilogy1(1),'DisplayName','无迭代','Marker','square','LineStyle','-.');
set(semilogy1(2),'DisplayName','迭代1次','Marker','diamond','LineStyle','--');
set(semilogy1(3),'DisplayName','迭代两次','Marker','x');
set(semilogy1(4),'DisplayName','迭代三次','Marker','o');

% 取消以下行的注释以保留坐标区的 X 范围
% xlim(axes1,[10 40]);
% 取消以下行的注释以保留坐标区的 Y 范围
% ylim(axes1,[0.00981127595093666 0.281460978842495]);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'YMinorTick','on','YScale','log');
% 创建 legend
legend(axes1,'show');

