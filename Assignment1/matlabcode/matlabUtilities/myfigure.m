function [ax,fig] = myfigure(varargin)


if ~isempty(varargin)
    position = varargin{1};
else
    position = [300 400 700 400];
end
    
fig = figure('Position',position);
ax  = gca;
% Setting for my plots
set(ax,'Linewidth',1,'FontName','Times','FontSize',16);

end