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
aba = [aba(:, 1) == 0, aba(:, 1) == 1, aba(:, 2:end)];

train_sz = 3133;
test_sz = 1044;

aba_tr_data = aba(1:train_sz, 1:9);
aba_tr_label = aba(1:train_sz, 10);

aba_tst_data = aba((train_sz + 1):end, 1:9);
aba_tst_label = aba((train_sz + 1):end, 10);

% SCALING
% took a look at libsvm's scaling, and it's a complicated wrapper around
% mathematically simple functionality (mostly due to file handling in C)
% So to avoid taking things in & out of MATLAB, I just ensure that features
% are scaled uniformly to fall w/in [-1, +1] below

% column wise minimum and maximum
aba_train_max = max(aba_tr_data);
aba_train_min = min(aba_tr_data);

aba_tr_scl = scale(aba_tr_data, aba_train_min, aba_train_max, 1, -1);
aba_tst_scl = scale(aba_tst_data, aba_train_min, aba_train_max, 1, -1);


%% SVM CLASSIFICATION (B. 1-4)
K = 10;
cvind = crossvalidateindices(train_sz, K);
ckrange = -9:1:9;
degrees = 1:4;

percentfig = figure;
msefig = figure;
for d = degrees
    avgerror = zeros(length(ckrange), 1);
    avgmse = zeros(length(ckrange), 1);
    stderror = zeros(length(ckrange), 1);
    stdmse = zeros(length(ckrange), 1);
    for ck = ckrange
        errs = zeros(K, 1);
        mses = zeros(K, 1);
        for i = 1:K
            model = svmtrain(aba_tr_label(findfold(cvind, i), :), aba_tr_scl(findfold(cvind, i), :), ['-s 0 -t 1 -d ' num2str(d) ' -c ' num2str(2^ck)]);
            [~, accuracy, ~] = svmpredict(aba_tr_label(cvind{i}), aba_tr_scl(cvind{i}, :), model);
            errs(i) = 100 - accuracy(1);
            mses(i) = accuracy(2);
        end
        avgerror(ckrange == ck) = mean(errs);
        stderror(ckrange == ck) = std(errs);
        avgmse(ckrange == ck) = mean(mses);
        stdmse(ckrange == ck) = std(mses);
    end
    figure(percentfig);
    hold on; errorbar(ckrange, avgerror, stderror); hold off;
    figure(msefig);
    hold on; errorbar(ckrange, avgmse, stdmse); hold off;
end

figure(percentfig);
legend({'Degree 1', 'Degree 2', 'Degree 3', 'Degree 4'});
title('Percent error for polynomial kernels, C = 2^k');
xlabel('k');
ylabel('Percent error');
figure(msefig);
legend({'Degree 1', 'Degree 2', 'Degree 3', 'Degree 4'});
title('Mean Squared Error for polynomial kernels, C = 2^k');
xlabel('k');
ylabel('MSE');

%% SVM CLASSIFICATION WITH OPTIMAL C* (B. 5)
degrees = 1:6;
avgerror = zeros(length(degrees), 1);
stderror = zeros(length(degrees), 1);
tsterror = zeros(length(degrees), 1);

cvsvs = zeros(length(degrees), 1);
cvhpsvs = zeros(length(degrees), 1);
cvsvsstd = zeros(length(degrees), 1);
cvhpsvsstd = zeros(length(degrees), 1);
tstsvs = zeros(length(degrees), 1);
tsthpsvs = zeros(length(degrees), 1);
for d = degrees
    Cstar = 2^8;
    errors = zeros(K, 1);
    svs = zeros(K, 1);
    hpsvs = zeros(K, 1);
    for i = 1:K
        model = svmtrain(aba_tr_label(findfold(cvind, i), :), aba_tr_scl(findfold(cvind, i), :), ['-s 0 -t 1 -d ' num2str(d) ' -c ' num2str(Cstar)]);
        [~, accuracy, ~] = svmpredict(aba_tr_label(cvind{i}), aba_tr_scl(cvind{i}, :), model);
        errs(i) = 100 - accuracy(1);
        svs(i) = model.totalSV;
        hpsvs(i) = sum((model.sv_coef  < Cstar) & (model.sv_coef > -Cstar));
    end
    avgerror(d) = mean(errs);
    stderror(d) = std(errs);
    cvsvs(d) = mean(svs);
    cvhpsvs(d) = mean(hpsvs);
    cvsvsstd(d) = std(svs);
    cvhpsvsstd(d) = std(hpsvs);
    model = svmtrain(aba_tr_label(:, :), aba_tr_scl(:, :), ['-s 0 -t 1 -d ' num2str(d) ' -c ' num2str(Cstar)]);
    [~, accuracy, ~] = svmpredict(aba_tst_label, aba_tst_scl(:, :), model);
    tsterror(d) = 100 - accuracy(1);
    tstsvs(d) = model.totalSV;
    tsthpsvs(d) = sum((model.sv_coef  < Cstar) & (model.sv_coef > -Cstar));
end

figure;
hold on;
errorbar(degrees, avgerror, stderror);
plot(degrees, tsterror);
legend({'Cross Validation Error', 'Test Error'});
title('Percent error vs kernel degree');
xlabel('degree');
ylabel('Percent error');

figure;
hold on;
errorbar(degrees, cvsvs, cvsvsstd);
errorbar(degrees, cvhpsvs, cvhpsvsstd);
legend({'Total Support Vectors', 'Support Vectors on Hyperplanes'});
title('Number of Support Vectors vs Degree for Cross Validated Models with polynomial kernels');
xlabel('degree');
ylabel('Number of support vectors');

figure;
hold on;
plot(degrees, tstsvs);
plot(degrees, tsthpsvs);
legend({'Total Support Vectors', 'Support Vectors on Hyperplanes'});
title('Number of Support Vectors vs Degree for Full Model with polynomial kernels');
xlabel('degree of polynomial kernel');
ylabel('Number of support vectors');

%% SPARSE SVM (B. 6. d)

tst_err_sparse = zeros(length(degrees), 1);
cv_err_sparse = zeros(length(degrees), 1);
cv_err_sparse_std = zeros(length(degrees), 1);
    
for d = degrees
    dim = size(aba_tr_scl);
    aba_tr_sparse = kernelize(aba_tr_scl, aba_tr_scl, d) .* repmat(aba_tr_label', train_sz, 1);
    aba_tst_sparse = kernelize(aba_tr_scl, aba_tst_scl, d) .* repmat(aba_tr_label', test_sz, 1);
    errs = zeros(K, 1);
    Cstar = 2^8;
    for i = 1:K
        model = svmtrain(aba_tr_label(findfold(cvind, i), :), aba_tr_sparse(findfold(cvind, i), :), ['-t 0 -c ' num2str(Cstar)]);
        [~, accuracy, ~] = svmpredict(aba_tr_label(cvind{i}), aba_tr_sparse(cvind{i}, :), model);
        errs(i) = 100 - accuracy(1);
    end
    cv_err_sparse(d) = mean(errs);
    cv_err_sparse_std(d) = std(errs);
    
    model = svmtrain(aba_tr_label, aba_tr_sparse, ['-t 0 -c ' num2str(Cstar)]);
    [~, accuracy, ~] = svmpredict(aba_tst_label, aba_tst_sparse, model);
    tst_err_sparse(d) = 100 - accuracy(1);
end
figure;
hold on;
errorbar(degrees, cv_err_sparse, cv_err_sparse_std);
plot(degrees, tst_err_sparse);
legend({'Cross Validation Error', 'Test Error'});
title('Percent Error for Sparse Support Vector Solution');
xlabel('Degree');
ylabel('Percent Error');




%% HELPER METHODS
function kernelized = kernelize(training_data, data, d)
% training_data : m x f matrix of m observations of f features
% data : n x f matrix of n observations of f features
% d: degree of polynomial
% 
% kernelized: n x m matrix kernelized with respect to polynomial kernel
    dim_tr = size(training_data);
    dim_dt = size(training_data);
    if (dim_dt(2) ~= dim_tr(2))
        error('issue discovered')
    end
    % gamma = 1 / num_features
    kernelized = (1 / (dim_tr(2)) * data * training_data').^d;
end


function scaled = scale(array, minv, maxv, ul, ll)
% scale array so that each column is scaled uniformly between minv and maxv
% and takes on values of at most ul (upper limit) and at least ll (lower
% limit)
d = size(array);
if (length(minv) ~= d(2) || length(maxv) ~= d(2))
    error('incompatible dimensions')
end

maxa = repmat(maxv, d(1), 1);
mina = repmat(minv, d(1), 1);

% truncate columns according to respective min/max
mxind = array > maxa;
mnind = array < mina; 
for i = 1:d(2)
    array(find(mxind(:, i)), i) = minv(i);
    array(find(mnind(:, i)), i) = maxv(i);
end

diff = array - mina;
denom = maxv - minv;
denom(denom == 0) = 1;
denomr = repmat(denom, d(1), 1);

scaled = ll + (ul - ll) * diff ./ (denomr);
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


