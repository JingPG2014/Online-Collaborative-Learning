function data = l2_normalize(data,dim)
if dim == 2
    norms = 1./max(sqrt(sum(data.^2,2)),eps);
elseif dim == 1
    norms = 1./max(sqrt(sum(data.^2,1)),eps);
end
    data = bsxfun(@times,data,norms);
end