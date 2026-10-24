"""Compact nonnegative C1 biweight kernel. Other callable kernels are accepted."""
biweight(u::Real) = abs(u) < 1 ? 15/16 * (1-u*u)^2 : 0.0

function _data(X,t)
    size(X,1) == length(t) || throw(DimensionMismatch("one index per row is required"))
    size(X,1) > 3 && size(X,2) > 0 || throw(ArgumentError("insufficient data"))
    all(isfinite,X) && all(isfinite,t) || throw(ArgumentError("nonfinite data"))
end

"""
    local_design(index, point, bandwidth; degree=2, kernel=biweight)

Realized polynomial intercept weights, coefficient map and design diagnostics.
The map sends an n-vector of responses to coefficients in powers of
`(index-point)/bandwidth`. No ridge or eigenvalue clipping is applied.
"""
function local_design(t::AbstractVector, point::Real, h::Real; degree::Integer=2,kernel=biweight)
    isfinite(h) && h > 0 && isfinite(point) || throw(ArgumentError("invalid point/bandwidth"))
    degree in 0:3 || throw(ArgumentError("degree must be in 0:3"))
    all(isfinite,t) || throw(ArgumentError("nonfinite index"))
    u = (t .- point)./h
    k = Float64.(kernel.(u))
    all(x -> isfinite(x) && x >= 0,k) || throw(ArgumentError("kernel must be nonnegative"))
    count(>(0),k) >= degree+5 || throw(DomainError(point,"too few local observations"))
    V = hcat([u.^j for j in 0:degree]...)
    M = V' * (k .* V)
    condition = cond(M)
    isfinite(condition) && condition < 1e12 || throw(DomainError(condition,"singular local design"))
    C = cholesky(Symmetric(M)) \ (V' .* k')
    a = vec(C[1,:])
    leverage = vec(sum(V .* C';dims=2))
    (weights=a, map=C, basis=V, leverage=leverage, condition=condition, local_n=count(>(0),k),
     equivalent_n=1/sum(abs2,a), max_weight=maximum(abs,a))
end

# For a fixed selected support, WLS has residual maker I-H. Its row squared
# norm is the exact residual variance factor under homoskedastic errors.
# Compute it from small Gram matrices, without allocating an n by n hat matrix.
function _residual_df(X,t,point,h,J,A,B,kernel)
    selected = findall(vec(any(!iszero,B;dims=2)))
    isempty(selected) && return (coefficients=zeros(length(A),length(J)),factor=ones(size(X,1)),df=0)
    U = X[:,A[selected]]
    w = Float64.(kernel.((t .- point)./h))
    all(x -> isfinite(x) && x>=0,w) && sum(w)>0 || throw(DomainError(point,"invalid refit window"))
    w ./= sum(w)
    C = cholesky(Symmetric(U'*(w.*U)))
    coefficients = zeros(length(A),length(J))
    coefficients[selected,:] = C \ (U'*(w.*X[:,J]))
    T = U / C
    hii = w .* vec(sum(T.*U;dims=2))
    hh = vec(sum((T*(U'*(w.^2 .* U))).*T;dims=2))
    factor = 1 .- 2hii .+ hh
    all(x -> isfinite(x) && x>sqrt(eps(Float64)),factor) ||
        throw(DomainError(minimum(factor),"degenerate residual degrees of freedom"))
    (coefficients=coefficients,factor=factor,df=length(selected))
end

"""Nonnegative local second moments for a conditionally mean-zero response."""
function local_covariance(X::AbstractMatrix,t::AbstractVector,point::Real,h::Real;kernel=biweight)
    _data(X,t)
    isfinite(h) && h > 0 && isfinite(point) || throw(ArgumentError("invalid point/bandwidth"))
    w = Float64.(kernel.((t .- point)./h))
    all(x -> isfinite(x) && x >= 0,w) && sum(w) > 0 || throw(DomainError(point,"empty/invalid kernel window"))
    w ./= sum(w)
    S = X' * (w .* X)
    (covariance=Matrix(Symmetric(S)),weights=w,effective_n=1/sum(abs2,w))
end

"""
    precision_pilot(X, index, point, bandwidth; method=:scaledlasso,
                    multiplier=1, delta=2, refit=false, kernel=biweight)

Local scaled-lasso or ACLIME precision pilot. Penalties use the realized Kish
sample size, not the global n. ACLIME uses floor(effective_n) in its two stages
and its diagonal ridge. This is a localized modification; the iid ACLIME theorem
does not directly apply to kernel-weighted data. With `refit=true`, refit the
selected neighborhoods by least squares. Return the pilot and diagnostics.
"""
function precision_pilot(X,t,point,h;method::Symbol=:scaledlasso,multiplier::Real=1,
                         delta::Real=2,refit::Bool=false,kernel=biweight)
    isfinite(multiplier) && multiplier > 0 || throw(ArgumentError("invalid penalty multiplier"))
    isfinite(delta) && delta > 0 || throw(ArgumentError("invalid ACLIME constant"))
    L = local_covariance(X,t,point,h;kernel)
    S, neff = L.covariance,L.effective_n
    neff >= 5 || throw(DomainError(neff,"pilot effective sample size below 5"))
    p = size(S,1)
    if method === :scaledlasso
        P = Matrix(slassoinv(S,max(1,floor(Int,neff));
            λ₀=multiplier*sqrt(2log(max(p,2))/neff),refit,ntasks=1))
    elseif method === :aclime
        P = Matrix(aclime(S,max(1,floor(Int,neff));δ=delta*multiplier,δ₁=delta,
                         truncate=false,ntasks=1))
        refit && (P = _refit_inverse(S,P))
    else
        throw(ArgumentError("pilot must be :scaledlasso or :aclime"))
    end
    all(isfinite,P) || throw(DomainError(P,"nonfinite precision pilot"))
    (precision=P,effective_n=neff,residual=maximum(abs,S*P-I))
end

# Reuse the parent's exact homotopy solver for a single response; no full
# inverse is computed for the two common-complement regressions.
function _scaled_regression(S,A,c,lambda;refit::Bool=false)
    active = [j for j in A if S[j,j] > 0]
    ids = [active;c]
    ws = _SLWorkspace(Matrix(S[ids,ids]))
    col = _ColumnPath{Float64}(1)
    _sl_column!(ws,length(ids),Float64(lambda),refit,col)
    v = zeros(length(ids))
    v[col.rows] = col.vals
    v[end] > 0 || throw(DomainError(v[end],"nonpositive scaled-lasso residual precision"))
    beta = zeros(length(A))
    positions = Dict(j => q for (q,j) in enumerate(A))
    for (q,j) in enumerate(active)
        beta[positions[j]] = -v[q]/v[end]
    end
    beta
end

function _regression(X,t,point,h,J,A,multiplier,kernel;refit::Bool=false)
    L = local_covariance(X,t,point,h;kernel)
    L.effective_n >= 5 || throw(DomainError(L.effective_n,"pilot effective sample size below 5"))
    lambda = multiplier*sqrt(2log(max(length(A),2))/L.effective_n)
    B = hcat([_scaled_regression(L.covariance,A,c,lambda;refit) for c in J]...)
    (coefficients=B,effective_n=L.effective_n)
end

# Refit each selected neighborhood, then use the parent's minimum-absolute-entry
# symmetrization convention. Singular refits are failures. No eigenvalue repair.
function _refit_inverse(S,P)
    p = size(S,1)
    Q = zeros(p,p)
    for j in 1:p
        A = [a for a in 1:p if a != j && P[a,j] != 0]
        beta = isempty(A) ? Float64[] : cholesky(Symmetric(S[A,A])) \ S[A,j]
        variance = S[j,j]-dot(S[j,A],beta)
        variance > eps(Float64)*max(S[j,j],1) || throw(DomainError(variance,"singular neighborhood refit"))
        Q[j,j] = 1/variance
        Q[A,j] = -beta/variance
    end
    for j in 1:p,a in j+1:p
        value = abs(Q[a,j]) <= abs(Q[j,a]) ? Q[a,j] : Q[j,a]
        Q[a,j] = Q[j,a] = value
    end
    Q
end
