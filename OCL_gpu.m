%r: the target low rank
%m: vocabulary size, e.g., number of tag
%n: total sample size
%d: visual feature dimension
%batch_size: batch size
%Y_cpu: m x n, tags
%X_cpu: d x n, images
%W_cpu: r x d, classifier in reduced label space.
%U_cpu: r x m, label embeddings
%lambda1: alpha
%lambda2: beta
%batch_size: number of samples in a mini-batch, default 256;
function [U_cpu, V_cpu, W_cpu,obj] = OCL_gpu(Y_cpu, X_cpu, W_cpu, U_cpu,r, lambda1, lambda2, batch_size,is_normalize,debug)
if ~exist('debug','var')
    debug = 0;
end
if ~exist('is_normalize','var')
    is_normalize = 0;
end


%% initialization
[m, n] = size(Y_cpu);
[d,~] = size(X_cpu);
disp([m,n,d,r]);
V_cpu = zeros(r, n, 'single');
%W_cpu = rand(r,d,'single');
W_gpu = gpuArray(W_cpu);
%U_cpu = rand(r,m,'single');
U_gpu = gpuArray(U_cpu);

C_gpu = zeros(r, m, 'single', 'gpuArray');
Dinv_gpu = eye(r, r, 'single', 'gpuArray') / single(lambda2);

A_gpu = zeros(r, d, 'single', 'gpuArray');
Binv_gpu = eye(d, 'single', 'gpuArray') * single(lambda1/lambda2);%B is Ainv_gpu

I_gpu = eye(batch_size, 'single', 'gpuArray');
regularizer_gpu = (lambda1+lambda2)*ones(r,1,'single');
regularizer_gpu = diag(regularizer_gpu);
regularizer_gpu = gpuArray(regularizer_gpu);

batch_size = single(batch_size);
%% online optimization
count = 1;
obj = [];
for t = 1: batch_size: n
    if debug
        fval = stoc_rpca_func(Y_cpu, X_cpu, W_cpu, U_cpu, V_cpu, lambda1, lambda2);
        obj = [obj fval];
        fprintf('Overall_fval = %f\n', fval);
    end
    tic;
    if t + batch_size - 1 > n
        batch_size = single(n + 1 - t);
        I_gpu = eye(batch_size, 'single', 'gpuArray');
    end
    y_cpu = full(Y_cpu(:,t:(t+batch_size-1)));
    x_cpu = full(X_cpu(:,t:(t+batch_size-1)));
    if is_normalize == 1
        x_cpu = l2_normalize(x_cpu,1);
    end
        
    y_gpu = gpuArray(single(y_cpu));
    x_gpu = gpuArray(single(x_cpu));
    p_gpu = W_gpu * x_gpu;
    v_gpu = (U_gpu * U_gpu' + regularizer_gpu)\(lambda1*p_gpu+U_gpu*y_gpu);
    
    V_cpu(:,t:(t+batch_size-1)) = gather(v_gpu);%store the solution
    
    %%solve for U, NOTE THAT we should cast the guy inside inv to DOUBLE,
    %%otherwise it is not a precise calculation, leanding to divergence
    Dinv_gpu = Dinv_gpu - Dinv_gpu * v_gpu * inv(double(I_gpu + v_gpu' * Dinv_gpu * v_gpu / batch_size)) * v_gpu' * Dinv_gpu / batch_size;
    C_gpu = C_gpu + v_gpu * y_gpu' / batch_size;
    U_gpu = Dinv_gpu * C_gpu;

    %%solve for W
    Binv_gpu = Binv_gpu - Binv_gpu * x_gpu * inv(double(I_gpu + x_gpu' * Binv_gpu * x_gpu / batch_size)) * x_gpu' * Binv_gpu / batch_size;
    A_gpu = A_gpu + v_gpu * x_gpu'/ batch_size;
    W_gpu = A_gpu * Binv_gpu;

    W_cpu = gather(W_gpu);
    U_cpu = gather(U_gpu);
   
   
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
