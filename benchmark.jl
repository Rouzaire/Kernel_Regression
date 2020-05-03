using Plots, Distributions, SpecialFunctions, LinearAlgebra, Distributed
pyplot()
include("function_definitions.jl")

## Single Run
Ptrain = 250
Ptest  = 250

νT = 8 ; νS = 1/2 ; dimension = 1

X             = generate_X(Ptrain,Ptest,dimension,νT, νS)
GramT         = ConstructGramMatrices(X,νT,νS,"Teacher")
Z             = generate_Z(X,GramT)
Xtest,Ztest   = extract_TestSet(X,Z,Ptest)
Xtrain,Ztrain = extract_TrainSet(X,Z,Ptrain)

# test_err,exact_err = GradientDescent(Xtrain,Ztrain,Xtest,Ztest,epochs,epochs_saved,η,νT,νS)
# @time test_err,exact_err,epochs = ConjugateGradientDescent(Xtrain,Ztrain,Xtest,Ztest,νT,νS)
@time test_err,exact_err,train_err = ConjugateGradientDescentFixedEpochs(Xtrain,Ztrain,Xtest,Ztest,νT,νS)

plot(box=true)
# plot!(1:100,test_err,xaxis=:log,yaxis=:log,label="P = $Ptrain")
plot!(test_err,xaxis=:log,yaxis=:log,label="Test Error")
plot!(train_err,xaxis=:log,yaxis=:log,label="Train Error")
scatter!([100,NaN],[exact_err,NaN],markershape=:utriangle, markersize=10,label=nothing,color=:black)
xlabel!("Epochs")
ylabel!("Test Error")
titre(νT,νS)
# savefig("Figures\\benchmark_CGD_Epochs_Fixed.pdf")



## Impact of jitter
res = []
test_err_exact = norm(analytic_prediction(Xtrain,Ztrain,Xtest,ConstructGramMatrices(Xtrain,νT,νS,"Student")) - Ztest)^2 / Ptest
xx = 10 .^(range(-8,stop=0,length=100))
for τ in xx
    GramS         = ConstructGramMatrices(Xtrain,νT,νS,"Student") + τ*I(length(Xtrain))
    tmp = norm(analytic_prediction(Xtrain,Ztrain,Xtest,GramS) - Ztest)^2 / Ptest
    append!(res,abs(test_err_exact - tmp)/test_err_exact)
end

# plot(box=true)
plot!(xx,res,xaxis=:log,yaxis=:log,label="P = $(length(Xtrain))")
# plot!(xx[30:60],100*xx[30:60] .^1 ,xaxis=:log,yaxis=:log,label="Slope 1",color=:black)
# xlabel!("τ [Gram → Gram + τ*Identity]")
# ylabel!("Relative Analytical error wrt τ = 0")
# title!("νT = $νT, νS = $νS")
savefig("Figures\\Impact_jitter_nu$νT.pdf")

PP = [30,50,100,200,300,500,1000,2000]
Ptest  = 1000 ; νT = 8 ; νS = 1/2 ; dimension = 1 ; nb_teacher = 1
exact_no_cond   = zeros(length(PP),nb_teacher)
exact_with_cond = zeros(length(PP),nb_teacher)
for j in 1:nb_teacher
    for i in eachindex(PP)
        println("$j/$nb_teacher , P = $(PP[i]) ")
        X             = generate_X(maximum(PP),Ptest,dimension,νT,νS)
        GramT         = ConstructGramMatrices(X,νT,νS,"Teacher")
        Z             = generate_Z(X,GramT)
        Xtest,Ztest   = extract_TestSet(X,Z,Ptest)
        Xtrain,Ztrain = extract_TrainSet(X,Z,PP[i])
        GramS = ConstructGramMatrices(Xtrain,PP[i],νT,"Student")
        exact_no_cond[i,j] = norm(analytic_prediction(Xtrain,Ztrain,Xtest,GramS) - Ztest)^2 / Ptest
        exact_with_cond[i,j] = norm(analytic_prediction(Xtrain,Ztrain,Xtest,GramS + 1E-3*I(PP[i])) - Ztest)^2 / Ptest
    end
end
plot(box=true)
plot!(PP,mean(exact_no_cond,dims=2),ribbon=0.5*std(exact_with_cond,dims=2),xaxis=:log,yaxis=:log,mark=:o,label="No Conditioning")
plot!(PP,mean(exact_with_cond,dims=2),ribbon=0.5*std(exact_with_cond,dims=2),xaxis=:log,yaxis=:log,mark=:o,label="With 1E-3 Conditioning ")
xlabel!("P")
ylabel!("Exact error")
titre(νT,νS)
savefig("Figures\\Impact_jitter_nu$νT _v2.pdf")

## Visualisation des predictions
P = 1000
νT = 3 ; νS = 1/2 ; dimension = 1
X        = sort(2π*rand(P))
Xtrain = X[1:100:end]
Xtest = [X[250],X[460],X[830]]
Gram         = ConstructGramMatrices(X,νT,νS,"Teacher")
Z      = rand(MvNormal(zeros(length(X)),Gram))
Ztrain = Z[1:100:end]
Ztest = [Z[250],Z[460],Z[830]]
GramS = ConstructGramMatrices(Xtrain,νT,νS,"Student")
predictionS = zeros(length(X)) ; invv = inv(GramS)
for i in 1:length(X)
    predictionS[i] = [kS(abs(Xtrain[j] - X[i])) for j in 1:length(Xtrain)]' * invv * Ztrain
end

plot(box=true)
plot!(X,Z,label="Underlying Teacher Kernel Realisation",color=:black)
scatter!(Xtrain,Ztrain,label="Training Set",color=:green,markersize=10)
scatter!(Xtest,Ztest,label="Testing Set",color=:red,marker=:utriangle,markersize=10)
plot!(X,predictionS,label="Prediction of the Student",color=:blue)
xlabel!("Space")
titre(νT,νS)
savefig("Figures\\vizu_pred.pdf")

colorschemes[:tab10]
palette(:auto)
palette(:)
get_color_palette(:auto, plot_color(:white), 17)
default(:color)

clibraries()
cgradients()
