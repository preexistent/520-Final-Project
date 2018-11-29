function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

pred_labels=randn(size(test_inputs,1),size(train_labels,2));

X = train_inputs;
y = train_labels;
N = size(y);
rho = corr(X(:,22:2021),y);
rho = abs(rho);
N_test = size(test_inputs);

selectNfeature = 500;
[B,I] = maxk(rho, selectNfeature, 1);
X_train = zeros(9, N(1), 21+selectNfeature);
X_test = zeros(9, N_test(1),21+selectNfeature);
for i = 1:9
    X_train(i,:,:) = [X(:,1:21),X(:,I(:,i))];
    X_test(i,:,:) = [test_inputs(:,1:21), test_inputs(:,I(:,i))];
end

y_hat1 = zeros(size(test_inputs, 1), 9);
y_pre1 = zeros(size(train_inputs, 1), 9);
for n = 1:9
    trainData = reshape(X_train(n,:,:), N(1), 21+selectNfeature);
    trainData = zscore(trainData);
    testData = reshape(X_test(n,:,:),N_test(1),21+selectNfeature);
    testData = zscore(testData);
    [Mdl, FitInfo] = fitrlinear(trainData,y(:,n));
    y_pre1(:,n) = predict(Mdl, trainData);
    y_hat1(:,n) = predict(Mdl, testData);
end

%% Random forest

RF_models = cell(1, 9);
y_hat2 = NaN(size(test_inputs, 1), 9);
y_pre2 = zeros(size(train_inputs, 1), 9);
for j = 1:9
        trainData = reshape(X_train(n,:,:), N(1), 21+selectNfeature);
        trainData = zscore(trainData);
        testData = reshape(X_test(n,:,:),N_test(1),21+selectNfeature);
        testData = zscore(testData);
        % Using 500 trees
        RF_models{1, j} = TreeBagger(500, trainData,...
            y(:, j), 'Method','regression');
        % Make prediction
        y_pre2(:, j) = predict(RF_models{1, j}, trainData); 
        y_hat2(:, j) = predict(RF_models{1, j}, testData);     
end
% pred_labels = pred_labels + y_hat;
% % pred_labels = pred_labels ./ 2;



%% PCA
% Project X onto new PC's
X = train_inputs;
Y_train = train_labels;
tweets = X(:,22:2021);
n = size(X, 1);
p = size(train_labels, 2);

% Compute PC's
% Don't need to normalize b/c all columns in same units
[coeff_tweets, score_tweets,~,~,~] = pca(tweets);
tweet_reduced_X = [X(:, 1:21), score_tweets(:, 1:35)];
tweet_reduced_X = zscore(tweet_reduced_X);

% Need to project test set onto PC's
test_tweets = test_inputs(:,22:2021);
test_tweets_PC = test_tweets * coeff_tweets(:, 1:35); % PC-reduced scores
X_test = [test_inputs(:, 1:21), test_tweets_PC];
X_test = zscore(X_test);

%% Random forest
RF_models = cell(1, p);
y_hat3 = NaN(size(test_inputs, 1), p);
y_pre3 = zeros(size(train_inputs, 1), 9);
for j = 1:p
        % Using 500 trees
        RF_models{1, j} = TreeBagger(500, tweet_reduced_X,...
            Y_train(:, j), 'Method','regression');
        % Make prediction
        y_pre3(:, j) = predict(RF_models{1, j}, tweet_reduced_X); 
        y_hat3(:, j) = predict(RF_models{1, j}, X_test);     
end
% pred_labels = pred_labels + y_hat;
% pred_labels = pred_labels ./ 2;

lr_models = cell(1, p);
y_hat4 = NaN(size(test_inputs, 1), p);
y_pre4 = zeros(size(train_inputs, 1), 9);
for j = 1:p
        lr_models{1, j} = fitrlinear(tweet_reduced_X, y(:,j));
        % Make prediction
        y_pre4(:, j) = predict(lr_models{1, j}, tweet_reduced_X);
        y_hat4(:, j) = predict(lr_models{1, j}, X_test);     
end
% pred_labels = pred_labels + y_hat;
% pred_labels = pred_labels ./3;

N = size(test_inputs);
y_hat5 = zeros(N(1),9);
y_pre5 = zeros(size(train_inputs, 1), 9);
for n = 1:9
    Mdl = fitrensemble(train_inputs(:,1:21),Y_train(:,n));
%     y_pre5(:,n) = predict(Mdl, tweet_reduced_X);
    y_hat5(:,n) = predict(Mdl, test_inputs(:,1:21));
end
% pred_labels = pred_labels + y_hat;
% pred_labels = pred_labels ./ 5;

lr_models = cell(1, 9);
for j = 1:9
        lr_models{1, j} = fitrlinear([y_pre1(:,j),  y_pre3(:,j), y_pre4(:,j)], y(:,j));
        % Make prediction
        pred_labels(:, j) = predict(lr_models{1, j}, [y_hat1(:,j),  y_hat3(:,j), y_hat4(:,j)]);     
end

pred_labels = 0.9*y_hat1 + 1.1*y_hat3 + 1.2*y_hat4 + 0.8*y_hat5;
pred_labels = pred_labels ./4;

end

