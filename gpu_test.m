%randomly generate 10,000 data of the dimension 1000, and 10,000 labels of
%the vocabulary size 200.
feat = rand(1000,10000);
gnd = rand(200,10000);
gnd = (gnd>0.5);

lambda1 = 0.001;

lambda2 = 1e-4;

%50-D embedding
target_dim = 50;

%specify your gpu
gpuDevice(2);

W = rand(50,4096,'single');

U = rand(50,size(gnd,1),'single');

[U_cpu, V_cpu, W_cpu,obj_val] = OCL_gpu(full(gnd), full(feat), W, U, target_dim, lambda1, lambda2, 256,1,1);
