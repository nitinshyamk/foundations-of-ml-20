% PREPROCESSING CMD LINE FOR SEX VARIABLE
% sed -i 's/M/0/g' abalone.data2
% sed -i 's/F/1/g' abalone.data2
% sed -i 's/I/2/g' abalone.data2

%% IMPORT AND PREPROCESSING
% matlab import as table abalone

aba = abalone{: , :};

% convert to binary classification of 1-9 as -1 label, rest as 1 label
aba(find(aba(:, 9) < 10), 9) = -1;
aba(find(aba(:, 9) >= 10), 9) = 1;
% SEX shouldn't be a numeric since this implies an ordering among the 
% three categories - split them up into two boolean codes isMale, 
% and isFemale, given by the ordering here. (isInfant is when both set to 0)
aba = [aba(:, 1) == 0, aba(:, 1) == 1, aba(:, 1) == 2, aba(:, 2:end)];

boolean_features = 1:3;

train_sz = 3133;
test_sz = 1044;

aba_tr_data = aba(1:train_sz, 1:10);
aba_tr_label = aba(1:train_sz, 11);

aba_tst_data = aba((train_sz + 1):end, 1:10);
aba_tst_label = aba((train_sz + 1):end, 11);


K = 10;
T = 10000;
tintervals = 100:100:10000;
cvind = crossvalidateindices(train_sz, K);
cvmodels = cell(K, 1);

cv_accuracy = zeros(K, length(tintervals));
tr_accuracy = zeros(K, length(tintervals));

for i = 1:K
    allbutifolds = findfold(cvind, i);
    ifold = cvind{i};
    
    data_tr = aba_tr_data(allbutifolds, :);
    labels_tr = aba_tr_label(allbutifolds, :);
    data_val = aba_tr_data(ifold, :);
    labels_val = aba_tr_label(ifold, :);
    
    [hts, feats, alphats] = adaboost_log2(data_tr, boolean_features, labels_tr, T);
    
    predict = @(data, t) sign(sum(apply_adaboost_predictor(data, hts(1:t), feats(1:t), alphats(1:t)), 2));
    
    get_accuracy = @(predictions, labels) sum(predictions ~= labels) / length(labels);
    
    for t = tintervals
        tr_accuracy(i, tintervals == t) = get_accuracy(predict(data_tr, t), labels_tr);
        cv_accuracy(i, tintervals == t) = get_accuracy(predict(data_val, t), labels_val);
    end
end


errorbar(tintervals, mean(tr_accuracy, 1), std(tr_accuracy, 1))
errorbar(tintervals, mean(cv_accuracy, 1), std(cv_accuracy, 1))

function [classifiers, features, alphas] = adaboost(data, boolean_features, labels, T)
[M, ~] = size(data);
if (length(labels) ~= M)
    error('Incompatible dimensions')
end
p = ones(M, 1) / M;

alphas = zeros(T, 1);
features = zeros(T, 1);
classifiers = cell(T, 1);
for t = 1:T
    [ht, epst, feature] = find_best_stump(data, boolean_features, p, labels);
    alphas(t) = 1/2 * log(( 1 - epst)/ epst);
    classifiers{t} = ht;
    features(t) = feature;
    
    yh = labels.* ht(data(:, feature));
    p = p .* exp( -1 * alphas(t) * yh) / ( 2 * (epst * (1 - epst))^0.5);
end
end

function [classifiers, features, alphas] = adaboost_log2(data, boolean_features, labels, T)
[M, ~] = size(data);
if (length(labels) ~= M)
    error('Incompatible dimensions')
end
p = ones(M, 1) / M;

alphas = zeros(T, 1);
features = zeros(T, 1);
classifiers = cell(T, 1);
predictions = zeros(M, T);
for t = 1:T
    [ht, epst, feature] = find_best_stump(data, boolean_features, p, labels);
    alphas(t) = 1/2 * log(( 1 - epst)/ epst);
    classifiers{t} = ht;
    features(t) = feature;
    
    predictions(:, t) = ht(data(:, feature)) * alphas(t);
    
    yf = labels.* sum(predictions, 2);
    weights = 1 ./ (1 + exp(yf));
    Zt = sum(weights);
    p = weights ./ Zt;
end
end

function [predictions_matrix] = apply_adaboost_predictor(data, classifiers, features, alphas)
T = length(classifiers);
if (length(features) ~= T || length(alphas) ~= T)
    error('incompatible dimensions');
end

[M, ~] = size(data);
predictions_matrix = zeros(M, T);
for t = 1:T
    ht = classifiers{t};
    ft = features(t);
    alphat = alphas(t);
    predictions_matrix(:, t) = ht(data(:, ft)) * alphat;
end
end



function [h, eps, feature] = find_best_stump(features, boolean_features, probabilities, labels)
[M, N] = size(features);
if (length(probabilities) ~= M || length(labels) ~= M)
    error('incompatible dimensions');
end

eps = 1;
for i = 1:N
    [ht, epst] = find_threshold_function(features(:, i), probabilities, labels, sum(boolean_features == i));
    err = sum(probabilities(ht(features(:, i)) ~= labels));
    if (abs(err - epst) > 0.005)
        warning(['True error significantly different from expected error, feature ', num2str(i)]);
    end
    if (err < eps)
        h = ht;
        eps = err;
        feature = i;
    end
end
end

function [h, eps] = find_threshold_function(features, probabilities, labels, isBoolean)
M = length(features);
if (length(probabilities) ~= M || length(labels) ~= M)
    error('Incompatible dimensions');
end

if (isBoolean)
    ths = [-0.5, 0.5, 1.5];
    pzeroplus = sum(probabilities((labels == 1) & (features == 0)));
    pzerominus = sum(probabilities((labels == -1) & (features == 0)));
    poneplus = sum(probabilities((labels == 1) & (features == 1)));
    poneminus = sum(probabilities((labels == -1) & (features == 1)));
    
    rightpluserrs = [pzerominus + poneminus, pzeroplus + poneminus, pzeroplus + poneplus];
    leftpluserrs = [poneplus + pzeroplus, pzerominus + poneplus, pzerominus + poneminus];
    
    [rm, rmind] = min(rightpluserrs);
    [lm, lmind] = min(leftpluserrs);
    
    if (lm < rm)
        h = @(x) sign(sign(ths(lmind) - x) * 2 + 1);
        eps = lm;
    else
        h = @(x) sign(sign(x - ths(rmind)) * 2 + 1);
        eps = rm;
    end
    return;
end

[sorted_features, ind] = sort(features);
sorted_features = [sorted_features(1) - 2; sorted_features; sorted_features(end) + 2];

sorted_probs = [0; probabilities(ind)];
sorted_labels = [0; labels(ind)];

pminus = zeros(M + 1, 1);
pplus = zeros(M + 1, 1);

for i = 2:(M + 1)
    pminus(i) = pminus(i - 1);
    pplus(i) = pplus(i - 1);
    if (sorted_labels(i) < 0)
        pminus(i) = sorted_probs(i) + pminus(i);
    elseif (sorted_labels(i) > 0)
        pplus(i) = sorted_probs(i) + pplus(i);
    end
end

eps = 1;
for i = 1:(M + 1)
    err_r = calculate_error(pplus, pminus, i, 1);
    err_l = calculate_error(pplus, pminus, i, 0);
    
    th = (sorted_features(i) + sorted_features(i + 1)) / 2;
    if (err_l <= err_r && err_l < eps)
        h = @(x) sign(sign(th - x) * 2 + 1);
        eps = err_l;
    elseif (err_r < err_l && err_r < eps)
        h = @(x) sign(sign(x - th) * 2 + 1);
        eps = err_r;
    end
end
end

function [err] = calculate_error(pplusleft, pminusleft, i, rightpos)
N = length(pplusleft);
pp = pplusleft(N);
pm = pminusleft(N);
if (rightpos > 0)
    err = pm - pminusleft(i) + pplusleft(i);
else
    err = pp - pplusleft(i) + pminusleft(i);
end
end

function cvind = crossvalidateindices(size, k)
% return indices corresponding to random k-fold cross validation on size
p = randperm(size);
cvind = cell(k, 1);
for i = 1:k
    cvind{i} = p((floor((i - 1) * size/k) + 1): floor(i * size/k))';
end
end

function indices = findfold(cvind, i)
k = length(cvind);
indices = [];
for j = 1:(i-1)
    indices = [indices; cvind{j}];
end
for j = (i+1):k
    indices = [indices; cvind{j}];
end
end


