using Plots, JLD, Statistics, Distributed, ColorSchemes
pyplot() ; default(:palette,ColorSchemes.tab10.colors[1:10]) ; plot()
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Regression Teacher Student")
include("function_definitions.jl")

NaNmean(x) = mean(filter(!isnan,x)) ; NaNmean(x,dimension) = mapslices(NaNmean,x,dims=dimension) ; NaNstd(x) = std(filter(!isnan,x)) ; NaNstd(x,dimension) = mapslices(NaNstd,x,dims=dimension)
### Load Variables

dayy = "2" ; param = load("Data\\parameters_day"*dayy*".jld")
PP   = param["PP"] ; Ptest = param["Ptest"] ; ν = param["νT"] ; νS = param["νS"]
dimension = param["dimension"] ; algo = param["algo"]
teachers_matrix = param["teachers_matrix"]
epochs_saved = ceil.(10 .^(range(1,stop=log10(1E5),length=101)))
dimension
# 4D Matrices to store data // dim 1 : epochs //  dim 2 : P //  dim 3 : Teacher // dim 4 : ν
teacher_high = teachers_matrix[end][1] ; teacher_low = teachers_matrix[1][1]
test_err_matrix  = NaN*zeros(101,length(PP),teacher_high,length(ν))
exact_err_matrix = NaN*zeros(1 ,length(PP),teacher_high,length(ν))
if algo     == "CGD" epochs_matrix = NaN*zeros(1 ,length(PP),teacher_high,length(ν))
elseif algo == "GD"  epochs_matrix = 1E3*ones(1 ,length(PP),teacher_high,length(ν))
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
coeff = 0.75*[1,1,1,3,3,24,8] ; s = [0.2 0.3 0.4 0.65 0.8 1.25 1.35]; factor = 0.5
# coeff = [0.8,0.85,1.1,1.7,1.3,8,10] ; s = [0.13 0.2 0.28 0.4 0.45 0.73 0.85 ] ; factor = 1/2
coeff = [0.7,0.7,0.7,0.7,0.7,0.7,0.7] ; s = [0.2 0.3 0.4 0.65 0.8 1.25 1.35]/10; factor = 0.5
plot(box=true,legend=:bottomleft)
for j in 1:length(ν)
    νT = ν[j]
    for i in 1:length(PP)
        # if isnan(round(epochs_matrix_std_teachers[1,i,1,j])) || round(epochs_matrix_std_teachers[1,i,1,j]) .== 0 str_std = "" else str_std =  " ± $(Int(round(epochs_matrix_std_teachers[1,i,1,j])))" end
        # lab = "P = $(Int(PP[i])) , ⟨#epochs⟩ ≈ $(Int(round(epochs_matrix_avg_teachers[1,i,1,j])))"*str_std
        lab = "P = $(Int(PP[i])) , ⟨#epochs⟩ ≈ $(Int(round(epochs_matrix_avg_teachers[1,i,1,j])))"
        lab = "ν = $νT"
        # xx = Int.(round.(10 .^(range(0,stop=log10(epochs_matrix_avg_teachers[1,i,1,j]),length=11))))
        display(plot!(epochs_saved,test_err_matrix_avg_teachers[1:end,i,1,j],ribbon=factor*test_err_matrix_std_teachers[1:end,i,1,j],fillalpha=.5,xaxis=:log,yaxis=:log,color=j,label=lab))
        if νT < 4
            display(plot!([1E3,1E5],coeff[j]*[1E3,1E5].^-s[j],color=j,line=:dash,label="Slope $(s[j])"))
        else
            display(plot!([5E2,1E4],coeff[j]*[5E2,1E4].^-s[j],color=j,line=:dash,label="Slope $(s[j])"))
        end
        # scatter!([epochs_saved[end],NaN],[exact_err_matrix[1,i,1,j],NaN],markershape=:utriangle,color=j, markersize=10,label="")
    end
    # plot!(xs[j],coeff[j]*xs[j].^(-s[j]),color=:black,label="Slope $(Int(s[j]))")
end
    xlabel!("Epochs")
    ylabel!("Test Error averaged over $teacher_high Teachers")
    title!("Teacher Matérn[ν] , Student Laplace , d=10")
    savefig("Figures\\Final Results, GD d=10\\dynamics_powerlawd10")
    # savefig("Figures\\Final Results, GD d=10\\dynamics_powerlawd1.pdf")


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

## Slope of dynamics (Student = Laplace)
nus = [0.5,0.75,1,1.5,2,20]
sloped1 = [0.25,0.28,0.3,0.5,0.66,1] # d = 1
sloped2 = [0.22,0.25,0.25,0.375,0.375,0.5] # d = 2
plot(nus,sloped1,marker=:o)
plot!(nus,sloped2,marker=:o)
