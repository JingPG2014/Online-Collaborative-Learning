function [U,V,W,obj] = OCL(Y, X,W,U, r, lambda1, lambda2, batch_size,debug)
%% initialization
[m, n] = size(Y);
[d,~] = size(X);
V = zeros(r, n);
%W = rand(r,d);
%U = rand(r,m);

C = zeros(r, m);
Dinv = eye(r, r) / lambda2;

A = zeros(r, d);
Binv = eye(d) * (lambda1/lambda2);

I = eye(batch_size);
regularizer = (lambda1+lambda2)*ones(r,1);
regularizer = diag(regularizer);
%% online optimization
count = 1;
obj = [];
for t = 1: batch_size: n
    if debug
        fval = stoc_rpca_func(Y, X, W, U, V, lambda1, lambda2);
        obj = [obj fval];
        fprintf('Overall_fval = %f\n', fval);
    end
    tic;
    if t + batch_size - 1 > n
        batch_size = single(n + 1 - t);
        I = eye(batch_size);
    end
    y = full(Y(:,t:(t+batch_size-1)));
    x = full(X(:,t:(t+batch_size-1)));
    p = W * x;
    v = (U * U' + regularizer)\(lambda1*p+U*y);
    
    V(:,t:(t+batch_size-1)) = v;%store the solution
    
    %%solve for U
    Dinv = Dinv - Dinv * v * inv(I + v' * Dinv * v / batch_size) * v' * Dinv / batch_size;
    C = C + v * y' / batch_size;
    U = Dinv * C;

    %%solve for W
    Binv = Binv - Binv * x * inv(I + x' * Binv * x / batch_size) * x' * Binv / batch_size;
    A = A + v * x'/ batch_size;
    W = A * Binv;
  

    if mod(count,10) == 0
        toc;
        disp([num2str(count*100/(n/256)),'% finished']);
    end
    count = count+1;

end

end

function [fval] = stoc_rpca_func(Y, X, W, U, V, lambda1, lambda2)
fval = norm(Y-U'*V,'fro')^2 + lambda1*norm(W*X-V,'fro')^2 + lambda2 * norm(U,'fro')^2 + lambda2 * norm(V,'fro')^2 + lambda2 * norm(W,'fro')^2;

end
