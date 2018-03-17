function plotFeatures(data)
% Plots all features of a struct in a subplot

fields = fieldnames(data);
% We want to only plot data series, not the scalar values in struct
plotable = [];
for i = 1:length(fields)
    if length(data.(fields{i})) > 1
        plotable = [plotable i];
    end
end

plotSize = length(plotable);
figure()
for i = 1:plotSize
    ax(i) = subplot(plotSize,1,i);
    plot(data.(fields{plotable(i)}))
    title(fields{plotable(i)})
end
linkaxes(ax,'x')

end