clc; clear;

% 读取 t1_wear.csv 文件
wear_data = readtable('t1_wear.csv');
average_values = wear_data.average;  % 提取 average 列

% 读取 t1Fea.xlsx 文件中的特征数据
feature_data = readtable('t1Fea.xlsx');

% 初始化结果矩阵
num_features = size(feature_data, 2);
spearman_results = zeros(1, num_features);
p_values = zeros(1, num_features);

% 计算每个特征与 average 列的斯皮尔曼相关系数
for i = 1:num_features
    feature_values = feature_data{:, i};  % 提取第 i 个特征的列数据
    
    % 计算斯皮尔曼相关系数和 P 值
    [rho, p] = corr(average_values, feature_values, 'Type', 'Spearman');
    
    % 存储结果
    spearman_results(i) = rho;
    p_values(i) = p;
end

% 将结果存储为表格
feature_names = feature_data.Properties.VariableNames;
correlation_table = array2table([spearman_results', p_values'], ...
    'VariableNames', {'SpearmanCorrelation', 'PValue'}, ...
    'RowNames', feature_names);

% 显示相关性表格
disp(correlation_table);

% 保存结果为 Excel 文件
output_file = 't1Spearman.xlsx';
writetable(correlation_table, output_file, 'WriteRowNames', true);
fprintf('Spearman correlation results saved to %s\n', output_file);

% 可视化：生成柱状图
figure('Position', [100, 100, 1200, 600]); % 设置图像大小
bar(spearman_results, 'FaceColor', [0.2 0.6 0.8]); % 绘制柱状图

% 简写 x 轴标签
short_feature_names = cellfun(@(x) x(1:min(6, length(x))), feature_names, 'UniformOutput', false);

% 设置 x 轴标签和样式
set(gca, 'XTick', 1:num_features, 'XTickLabel', short_feature_names, 'XTickLabelRotation', 45);
xlabel('Feature Names'); % x 轴标题
ylabel('Spearman Correlation'); % y 轴标题
title('Spearman Correlation with Average Wear'); % 图表标题
grid on; % 显示网格

% 标记显著的相关性（P 值 < 0.05）
hold on;
for i = 1:num_features
    if p_values(i) < 0.05
        text(i, spearman_results(i) + 0.02, '*', 'FontSize', 12, 'HorizontalAlignment', 'center');
    end
end
hold off;