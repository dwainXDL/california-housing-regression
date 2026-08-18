% octave counterpart to regression.ipynb
% same file, same split, same model -- so identical numbers to match.
% Run: octave regression.m

1; % treat as script
more off;   % do not page the output

% helper
function r2 = fit_r2(Atr, ytr, Ate, yte)
b = (Atr' * Atr) \ (Atr' * ytr);
pred = Ate * b;
r2 = 1 - sum((yte - pred).^2) / sum((yte - mean(yte)).^2);
end

% load
data = csvread("california.csv", 1, 0); % the 1 skips the header row
n = rows(data);

FEATURES = {"MedInc", "HouseAge", "AveRooms", "AveBedrms", ...
            "Population", "AveOccup", "Latitude", "Longitude"};

X = data(:, 1:8);
y = data(:, 9);

printf("rows loaded : %d\n", n);

% split
% floor() matches Python's int(0.75 * len(df)).
ntr = floor(0.75 * n);
nte = n - ntr;

Xtr = X(1:ntr, :);   
ytr = y(1:ntr);
Xte = X(ntr+1:end, :);
yte = y(ntr+1:end);

printf("train : %d rows\n", ntr);
printf("test  : %d rows\n\n", nte);

% train: normal equation
A = [ones(ntr,1), Xtr];   % design matrix : intercept column & 8 features
b = (A' * A) \ (A' * ytr);   % b = [intercept; w1 ... w8]

% evaluate on held-back test rows
Ate = [ones(nte,1), Xte];
pred = Ate * b;   % one dot product per row
err = pred - yte;

MAE = mean(abs(err));
RMSE = sqrt(mean(err .^ 2));
R2 = 1 - sum(err.^2) / sum((yte - mean(yte)).^2);

% report
printf("Octave (normal equation), all 8 features, test set\n");
printf(" intercept : %.6f\n", b(1));
for i = 1:numel(FEATURES)
    printf("   %-12s: %.6f\n", FEATURES{i}, b(i+1));
end
printf("   MAE   : %.6f\n", MAE);
printf("   RMSE  : %.6f\n", RMSE);
printf("   R2    : %.6f\n\n", R2);

% drop one ablation
printf("drop-one ablation (test R2)\n");
printf(" all 8 : %4f\n", R2);
for i = 1:8
    keep = setdiff(1:8, i);      % every column expect i
    r2_i = fit_r2([ones(ntr,1), Xtr(:,keep)], ytr, ...
                    [ones(nte, 1), Xte(:,keep)], yte);
    printf(" drop %-9s: %.4f  (change %+.4f)\n", FEATURES{i}, r2_i, r2_i - R2);
end
printf("\n");

% final model: all 8 + 2 engineered
AveRooms  = X(:,3);
AveBedrms = X(:,4);
AveOccup  = X(:,6);

RoomsPerPerson = AveRooms ./ AveOccup;    
BedrmRatio = AveBedrms ./ AveRooms;

XE   = [X, RoomsPerPerson, BedrmRatio]; % X already holds the 8 raw columns in order
ENGN = {"MedInc","HouseAge","AveRooms","AveBedrms","Population", ...
        "AveOccup","Latitude","Longitude","RoomsPerPerson","BedrmRatio"};

AE  = [ones(ntr,1), XE(1:ntr,:)];
AEe = [ones(nte,1), XE(ntr+1:end,:)];
bE  = (AE' * AE) \ (AE' * ytr);

predE = AEe * bE;
errE  = predE - yte;
RMSEE = sqrt(mean(errE .^ 2));
R2E   = 1 - sum(errE.^2) / sum((yte - mean(yte)).^2);

printf("Octave, final model (all 8 + 2 engineered), test set\n");
printf("   intercept     : %.6f\n", bE(1));
for i = 1:numel(ENGN)
    printf("   %-14s: %.6f\n", ENGN{i}, bE(i+1));
end
printf(" RMSE : %.6f\n", RMSEE);
printf(" R2 : %.6f\n\n", R2E);

% ridge, the normal-equation
% note: this is "unscaled" ridge, will not match the Python Ridge CV, which scaled features so
% two different models
% not cross check per say
lambda = 1;
I = eye(9); I(1,1) = 0;          % never penalise intercept
b_ridge = (A' * A + lambda * I) \ (A' * ytr);

pred_r = Ate * b_ridge;
R2_r = 1 - sum((yte - pred_r).^2) / sum((yte - mean(yte)).^2);

printf("ridge (unscaled, lambda = %g)\n", lambda);
printf("   test R2   : %.6f\n", R2_r);
printf("   vs OLS    : %+.6f\n", R2_r - R2);

% predicted vs actual
figure("visible", "off");
scatter(yte, pred, 4, "filled");
hold on;
plot([0 5.6], [0 5.6], "r-", "linewidth", 1.5);
xlabel("actual MedHouseVal ($100,000s)");
ylabel("predicted MedHouseVal ($100,000s)");
title(sprintf("Octave: predicted vs actual (R2 = %.3f)", R2));
axis([0 5.6 0 5.6]);
print -dpng "pred_vs_actual_octave.png";
printf("\nwrote pred_vs_actual_octave.png\n");

% ridge in octave is unscaled while its scaled in Python. 

