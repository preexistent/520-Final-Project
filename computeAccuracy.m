function accuracy = computeAccuracy(X, X_reconstruct, latent)

    n = size(X);
    n = n(1);
    var_sum = sum(latent);
    % var_sum = sum(var(X,0,1));
    square_sum = sum(sum((X-X_reconstruct).^2));
    accuracy = 1 - square_sum / (n*var_sum);
end