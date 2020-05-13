using Plots, JLD, Statistics, Distributed
pyplot()
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Regression Teacher Student")
include("function_definitions.jl")

NaNmean(x) = mean(filter(!isnan,x)) ; NaNmean(x,dimension) = mapslices(NaNmean,x,dims=dimension) ; NaNstd(x) = std(filter(!isnan,x)) ; NaNstd(x,dimension) = mapslices(NaNstd,x,dims=dimension)
### Load Variables

dayy = "2" ; param = load("Data\\parameters_day"*dayy*".jld")
PP   = param["PP"] ; Ptest = param["Ptest"] ; ν = param["νT"] ; νS = param["νS"]
dimension = param["dimension"] ; algo = param["algo"]
teachers_matrix = param["teachers_matrix"]
epochs_saved = [1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100] # % of learning

# 4D Matrices to store data // dim 1 : epochs //  dim 2 : P //  dim 3 : Teacher // dim 4 : ν
teacher_high = teachers_matrix[end][1] ; teacher_low = teachers_matrix[1][1]
test_err_matrix  = NaN*zeros(11,length(PP),teacher_high,length(ν))
exact_err_matrix = NaN*zeros(1 ,length(PP),teacher_high,length(ν))
if algo     == "CGD" epochs_matrix = NaN*zeros(1 ,length(PP),teacher_high,length(ν))
elseif algo == "GD"  epochs_matrix = 1E6*ones(1 ,length(PP),teacher_high,length(ν))
end

for i in eachindex(ν)
    if ν[i] ≤ 1
        test_err_matrix[:,:,1:teacher_low,i]  = load("Data\\data__nu="*string(ν[i])*" "*dayy*".jld")["test_err"]
        exact_err_matrix[1,:,1:teacher_low,i] = load("Data\\data__nu="*string(ν[i])*" "*dayy*".jld")["exact_err"]
        if algo == "CGD" epochs_matrix[1,:,1:teacher_low,i]    = load("Data\\data__nu="*string(ν[i])*" "*dayy*".jld")["epochs"] end
    else
        test_err_matrix[:,:,:,i]  = load("Data\\data__nu="*string(ν[i])*" "*dayy*".jld")["test_err"]
        exact_err_matrix[1,:,:,i] = load("Data\\data__nu="*string(ν[i])*" "*dayy*".jld")["exact_err"]
        if algo == "CGD" epochs_matrix[1,:,:,i]    = load("Data\\data__nu="*string(ν[i])*" "*dayy*".jld")["epochs"] end
    end
end

# Averages and Standard deviations over the different Teachers
dimm = 3
test_err_matrix_avg_teachers  = NaNmean(test_err_matrix, dimm)
test_err_matrix_std_teachers  = NaNstd(test_err_matrix,  dimm)
exact_err_matrix_avg_teachers = NaNmean(exact_err_matrix,dimm)
exact_err_matrix_std_teachers = NaNstd(exact_err_matrix, dimm)
epochs_matrix_avg_teachers    = NaNmean(epochs_matrix,   dimm)
epochs_matrix_std_teachers    = NaNstd(epochs_matrix,    dimm)

β = 2 ./dimension .*min.(ν,dimension .+ 2νS)
factor = 0.25
## Test Error vs P for several epochs
for j in 1:length(ν)
    νT = ν[j]
    plot(box=true)
    for i in 1:2:11
        display(plot!(PP,test_err_matrix_avg_teachers[i,:,1,j],ribbon=factor*test_err_matrix_std_teachers[i,:,1,j],fillalpha=.5,xaxis=:log,yaxis=:log,marker=:o,label="At $(epochs_saved[i])% of Learning"))
    end
    plot!(PP,exact_err_matrix_avg_teachers[1,:,1,j],color=:black,label="Exact Error")
    xlabel!("P")
    ylabel!("Test Error averaged over $teacher_high Teachers")
    titre(νT,νS)
    savefig("Figures\\Final Results, GD d=1\\testerror_vs_P_nuT$νT _nuS$νS.pdf")
end

## Test Error vs epochs for several P
xs = [[1,1000],[1,1000],[2,100],[2,20],[2,15]]
coeff = [0.8,0.5,1,5,10]
s = Float64.([1 2 4 6 8])
for j in 1:5
    νT = ν[j]
    plot(box=true,legend=:topright)
    for i in 1:length(PP)
        if isnan(round(epochs_matrix_std_teachers[1,i,1,j])) || round(epochs_matrix_std_teachers[1,i,1,j]) .== 0 str_std = "" else str_std =  " ± $(Int(round(epochs_matrix_std_teachers[1,i,1,j])))" end
        lab = "P = $(Int(PP[i])) , ⟨#epochs⟩ ≈ $(Int(round(epochs_matrix_avg_teachers[1,i,1,j])))"*str_std
        xx = Int.(round.(10 .^(range(0,stop=log10(epochs_matrix_avg_teachers[1,i,1,j]),length=11))))
        display(plot!(xx,test_err_matrix_avg_teachers[1:end,i,1,j],ribbon=factor*test_err_matrix_std_teachers[1:end,i,1,j],fillalpha=.5,xaxis=:log,yaxis=:log,marker=:o,color=i,label=lab))
        scatter!([xx[end],NaN],[exact_err_matrix[1,i,1,j],NaN],markershape=:utriangle,color=i, markersize=10,label=nothing)
    end
    # plot!(xs[j],coeff[j]*xs[j].^(-s[j]),color=:black,label="Slope $(Int(s[j]))")
    xlabel!("Epochs")
    ylabel!("Test Error averaged over $teacher_high Teachers")
    titre(νT,νS)
    savefig("Figures\\Final Results, GD d=1\\0testerror_vs_epochs_nuT$νT _nuS$νS.pdf")
end


##
plot(box=true) # plot err vs P for different ν
for i in 1:length(ν)
    display(plot!(PP,test_err_matrix_avg_teachers[end,:,1,i],ribbon=factor*test_err_matrix_std_teachers[end,:,1,i],fillalpha=.5,xaxis=:log,yaxis=:log,marker=:o,color=i,label="νT = $(ν[i])"))
end
plot!([1E2,2E3],3 *[1E2,2E3] .^ (-1),color=1,label="Slope -1",line=:dash)
plot!([1E2,2E3],70*([1E2,2E3] .^ (-2)),color=:2,label="Slope -2",line=:dash)
plot!([50,2000],3E3*[50,2000] .^ (-4),color=:black,label="Slope -4",line=:dash)
xlabel!("P")
ylabel!("Test Error Averaged over $teacher_high Teachers")
savefig("Figures\\testerror_vs_epochs_all_nu.pdf")

## Distribution of epochs
for i in 1:length(ν)
    νT = ν[i]
    plot(box=true)
    for j in 1:length(PP)
        display(histogram!(epochs_matrix[1,j,:,i],label="P = $(PP[j])",normalize=:pdf))
    end
    xlabel!("#Epochs in training")
    ylabel!("Epoch Distribution over (up to $teacher_high) Teachers")
    titre(νT,νS)
    savefig("Figures\\distrib_epochs_nuT_$νT.pdf")
end

## Number of epochs versus P at fixed tolerance
plot(box=true)
for i in 1:length(ν)
    display(plot!(PP,epochs_matrix_avg_teachers[1,:,1,i],ribbon=factor*epochs_matrix_std_teachers[1,:,1,i],xaxis=:log,yaxis=:log,label="νT = $(ν[i])"))
end
plot!([50, 2000],0.5*[50, 2000].^1.5,color=:black,line=:dash,label="Slope 1.5")
plot!([50, 1000],2*[50, 1000].^0.75,color=:black,label="Slope 0.75")
xlabel!("P")
ylabel!("⟨#epochs⟩")
title!("Fixed tolerance = 1E-5")
savefig("Figures\\trend_epochs_vs_P.pdf")
