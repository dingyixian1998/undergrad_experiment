clc; clear;

% 提取时域特征函数
function features = extract_time_features(data)
    if isempty(data)
        features = NaN(1, 10); % 10 个时域特征
        return;
    end

    % 计算时域特征
    mean_value = mean(abs(data));           % 均值
    variance_value = var(data);             % 方差
    rms_value = rms(data);                  % 均方根
    rate_of_change = mean(abs(diff(data))); % 变化率
    peak_to_peak = max(data) - min(data);   % 峰峰值
    crest_factor = max(abs(data)) / rms_value; % 波形因子

    features = [
        mean_value,                         % 均值
        variance_value,                     % 方差
        skewness(data),                     % 偏度
        kurtosis(data),                     % 峰度
        max(abs(data)),                     % 最大值
        std(abs(data)),                     % 标准差
        rms_value,                          % 均方根
        rate_of_change,                     % 变化率
        peak_to_peak,                       % 峰峰值
        crest_factor                        % 波形因子
    ];
end

% 提取频域特征函数
function features = extract_frequency_features(data)
    if isempty(data)
        features = NaN(1, 6); % 6 个频域特征
        return;
    end

    freq_data = fft(data);
    freq_magnitude = abs(freq_data);
    [Pxx_den, f] = pwelch(data); % 使用 Welch 方法估计频谱

    spectral_energy = sum(freq_magnitude.^2); % 频谱能量
    spectral_entropy = -sum((Pxx_den / sum(Pxx_den)) .* log(Pxx_den / sum(Pxx_den) + eps)); % 频谱熵
    spectral_flatness = geomean(Pxx_den) / mean(Pxx_den); % 频谱平坦度

    [~, idx] = max(Pxx_den);
    dominant_freq = f(idx); % 主频率
    frequency_center = sum(f .* Pxx_den) / sum(Pxx_den); % 频率中心
    bandwidth = sqrt(sum((f - frequency_center).^2 .* Pxx_den) / sum(Pxx_den)); % 频率带宽

    features = [
        spectral_energy, spectral_entropy, spectral_flatness, ...
        dominant_freq, frequency_center, bandwidth
    ];
end

% 提取时频特征：整合小波能量、奇异谱熵和 IMF 能量
function features = extract_time_frequency_features(data)
    if isempty(data)
        features = NaN(1, 4); % 4 个时频特征
        return;
    end

    % 1. 奇异谱熵 (Singular Spectral Entropy)
    [Pxx_den, ~] = pwelch(data); % 使用 Welch 估计
    Pxx_den = Pxx_den / sum(Pxx_den); % 归一化
    spectral_entropy = -sum(Pxx_den .* log(Pxx_den + eps)); % 避免 log(0)

    % 2. IMF 能量 (IMF Energy) - 增加异常处理
    try
        imfs = emd(data); % EMD 分解
        num_imfs = size(imfs, 2);
        imf_energy = sum(imfs.^2, 1); % 每个 IMF 的能量
        total_imf_energy = sum(imf_energy); % 总能量
    catch
        warning('EMD 分解失败，填充 NaN');
        total_imf_energy = NaN;
    end

    % 3. 小波能量 (Wavelet Energy)
    [C, L] = wavedec(data, 3, 'db1'); % 小波分解
    wavelet_energy = zeros(1, 2);
    detail_coeff = detcoef(C, L, 2); % 获取第二层的细节系数
    wavelet_energy(1) = sum(detail_coeff.^2); % 计算细节系数的能量
    approx_coeff = appcoef(C, L, 'db1', 2); % 获取第二层的近似系数
    wavelet_energy(2) = sum(approx_coeff.^2); % 计算近似系数的能量
    wavelet_energy = wavelet_energy(:)';  % 转置为行向量

    % 4. 组合所有时频特征
    features = [
        spectral_entropy, total_imf_energy, wavelet_energy
    ];
end

% 特征提取与特征填充
function data = process_data(sample_num, sensor_num, base_path)
    data = {};  % Store features in a cell array

    for sample_index = 1:sample_num
        sample_features = {};  % Store features for each sample
        file_name = sprintf('c_%s_%03d.csv', base_path(end), sample_index);
        sample_path = fullfile(base_path, file_name);

        if isfile(sample_path)
            sensor_data = readmatrix(sample_path);

            for sensor_index = 1:sensor_num
                % 提取时域特征
                time_features = extract_time_features(sensor_data(:, sensor_index));
                % 提取频域特征
                frequency_features = extract_frequency_features(sensor_data(:, sensor_index));
                % 提取时频特征
                time_frequency_features = extract_time_frequency_features(sensor_data(:, sensor_index));

                % 存储每个传感器的特征
                sensor_features.time = time_features;
                sensor_features.frequency = frequency_features;
                sensor_features.time_frequency = time_frequency_features;

                % 将该传感器的特征存入样本特征中
                sample_features{end+1} = sensor_features;
            end
        else
            fprintf('File not found: %s\n', sample_path);

            % 如果文件不存在，用 NaN 填充特征
            for sensor_index = 1:sensor_num
                sensor_features.time = NaN(1, 11);          % 10 个时域特征
                sensor_features.frequency = NaN(1, 6);      % 6 个频域特征
                sensor_features.time_frequency = NaN(1, 4); % 4 个时频特征
                sample_features{end+1} = sensor_features;
            end
        end

        % 将该样本的所有传感器特征存入数据集中
        data{end+1} = sample_features;
    end
end

% 主程序：调用数据处理函数 
tic; % 计时开始

sample_num = 53; % 样本数
sensor_num = 7;   % 传感器数量
base_path = 't1'; % 文件路径

% 调用数据处理函数
t1_combined = process_data(sample_num, sensor_num, base_path);
fprintf('t1_combined shape: [%d, %d]\n', length(t1_combined), sensor_num);

% 切削序号 (用于绘图的 x 轴)
tool_wear = (1:sample_num)';

% 定义信号名称
signals_names = {'Fx', 'Fy', 'Fz', ...
                 'Vx', 'Vy', 'Vz', 'AE'};

% 特征名称（时域、频域、时频）
features_names_time = { 'Mean', 'Var', 'Skew', 'Kurt', 'Max', ...
                       'StdDev', 'RMS', 'RoC', 'P2P', 'Crest'
                       };

features_names_freq = {'SpecEn', 'SpecEnt', 'SpecFlat', ...
                        'DomFreq', 'FreqCtr', 'BW'};

features_names_time_freq = {'SingSpecEnt', 'IMFEn', 'WvltDetEn', 'WvltAppEn'};

% 整合所有特征名称
features_names = [features_names_time, features_names_freq, ...
                  features_names_time_freq];

% 创建列名称
columns = {};
for i = 1:length(signals_names)  % 7 个信号
    for j = 1:length(features_names)  % 20 个特征
        columns{end+1} = sprintf('%s - %s', signals_names{i}, features_names{j});
    end
end

% 检查列名数量是否正确
expected_columns = sensor_num * length(features_names); % 7 * 20 = 140 列
fprintf('Number of columns generated: %d\n', length(columns));

% 初始化用于存储所有样本的特征矩阵
all_features = [];

% 将提取的特征转换为二维矩阵并合并
for sample_index = 1:sample_num
    sample_features = [];
    for sensor_index = 1:sensor_num
        sensor_features = t1_combined{sample_index}{sensor_index};

        % 检查特征是否存在，并确保长度一致
        if isstruct(sensor_features) && ...
           ~isempty(sensor_features.time) && ...
           ~isempty(sensor_features.frequency) && ...
           ~isempty(sensor_features.time_frequency)

            % 拼接每个传感器的 25 个特征
            combined_features = [sensor_features.time(:)', ...
                                 sensor_features.frequency(:)', ...
                                 sensor_features.time_frequency(:)'];

            % 将该传感器的特征添加到样本特征中
            sample_features = [sample_features, combined_features];
        else
            % 如果数据缺失或为空，则用 NaN 填充，确保长度一致
            combined_features = [NaN(1, 10), NaN(1, 6), NaN(1, 4)];

            % 将 NaN 特征添加到样本特征中
            sample_features = [sample_features, combined_features];
        end
    end
    % 将每个样本的特征添加到最终的特征矩阵中
    all_features = [all_features; sample_features];
end

% 确保特征矩阵的列数与列名数量一致
fprintf('Number of features in each sample: %d\n', size(all_features, 2));
fprintf('Expected number of columns: %d\n', length(columns));

% 检查列数量是否一致
if size(all_features, 2) ~= length(columns)
    error('Mismatch between the number of columns and the number of features.');
end

% 将数据转换为表格形式并保存为 Excel 文件
data_table = array2table(all_features, 'VariableNames', columns);
output_file = 't1Fea.xlsx';
writetable(data_table, output_file);
fprintf('Features saved to %s\n', output_file);

% 可视化特征
for sensor_index = 1:sensor_num
    figure('Position', [50, 50, 1000, 1000]);
    sgtitle(sprintf('%s', signals_names{sensor_index})); % 图表标题
    num_features = length(features_names); % 每个信号的特征数
    for feature_index = 1:num_features
        subplot(5, 4, feature_index); % 创建 5x5 的子图布局
        plot(tool_wear, all_features(:, (sensor_index-1)*num_features + feature_index)); % 绘制特征图
        title(features_names{feature_index}); % 子图标题为特征名称
        xlabel('Sample Index');
        ylabel('Value');
    end
end

toc; % 计时结束
