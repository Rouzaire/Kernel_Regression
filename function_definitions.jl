## Some basic functions
@everywhere dist(x,y) = norm(x-y) # euclidien distance
@everywhere MSE(a,Z,gram_matrix) = norm(gram_matrix*a - Z)^2/length(a) # Loss  function (MSE between current approximation of minimum and actual function)
@everywhere dMSE(a,Z,gram_matrix) = 2/length(a)*gram_matrix*(gram_matrix*a - Z) # derivative of the loss function wrt a (the current guess)

## Costumized Title
@everywhere function titre(νT,νS)
    if isinteger(νT) νT = Int(νT) end
    if isinteger(νS) νS = Int(νS) end
    if νT == νS
        if νT == 1/2 title!("Teacher = Student = Laplace")
        else title!("Teacher = Student = Matérn [ν = $νT]")
        end
    else
        if νS == 1/2 title!("Teacher = Matérn [ν = $νT], Student = Laplace")
        else title!("Teacher = Matérn [ν = $νT], Student = Matérn [ν = $νS]")
        end
    end
end
## Matérn Covariance functions
@everywhere function k(h,ν,σ=1,ρ=1) # Teacher kernel
    ## h is the euclidian distance in real space R² (actual subspace = unit circle)
    ## ν (nu) the smoothness of the Matérn covariance function (here ν ∈ [0.5,8])
    ## ρ is the length scale
    ## σ² is the amplitude/sill of the function
    if     ν == 1/2 return σ^2*exp(-h/ρ)
    elseif ν == 3/2 return σ^2*(1+sqrt(3)*h/ρ)*exp(-sqrt(3)*h/ρ)
    elseif ν == 5/2 return σ^2*(1+sqrt(5)*h/ρ + 5/3*h^2/ρ^2)*exp(-sqrt(5)*h/ρ)
    else            return σ^2 * 2.0^(1-ν) / SpecialFunctions.gamma(ν) * (sqrt(2ν)h/ρ)^ν * SpecialFunctions.besselk(ν,Float64(sqrt(2ν)h/ρ))
    end
end


## Prediction functions
@everywhere function analytic_prediction(Xtrain,Ztrain,Xtest,Gram,νS,method="Inverse") # "à la Kriging", returns a vector
    # x0 is the point where one want to predict
    prediction = zeros(length(Xtest))
    if method == "Cholesky"
        L = LinearAlgebra.cholesky(Gram).L # lower gram matrix from LLT factorization
        for i in eachindex(Xtest)
            V = [k(dist(Xtest[i],element),νS) for element in Xtrain]
            c1 = L\Ztrain ; c2 = L\V ; prediction[i] = c1' * c2 # scalar product
        end
        return prediction
    elseif method == "Inverse"
        invGram = inv(Gram)
        for i in eachindex(Xtest)
            V = [k(dist(Xtest[i],element),νS) for element in Xtrain]
            prediction[i] = V'*invGram*Ztrain
        end
        return prediction
    else println("String Unknown. Choose among \"Cholesky\" or \"Inverse\" ")
    end
end

@everywhere function predict(Xtest,Xtrain,a,νS)
    prediction = zeros(length(Xtest))
    for i in eachindex(Xtest)
        V = [k(dist(Xtest[i],element),νS) for element in Xtrain]
        prediction[i] = a' * V
    end
    return prediction
end

## Generate x components of the whole dataset, uniformly distributed on a hypersphere of dimension d=D-1.
    # for instance, in d=1, X = unit circle in R^2
    # for instance, in d=2, X = unit sphere in R^3
@everywhere function generate_X(Ptrain, Ptest,dimension,νT,νS)
    @assert isinteger(dimension) ; @assert dimension > 0
    N = Ptrain + Ptest
    X = rand(MvNormal(zeros(dimension+1),I(dimension+1)),N)
    normX = [norm(X[:,i]) for i in 1:N]
    X_normalized = [(X[:,i] ./ normX[i]) for i in 1:N]
    return X_normalized
end

@everywhere function generate_Z(X,GramT)
    return rand(MvNormal(zeros(length(X)),GramT)) # true sequence generated with the Teacher kernel, to be approached by the Student kernel
end

@everywhere function extract_TrainSet(X,Z,Ptrain)
    Xtrain = X[1:Ptrain] ; Ztrain = Z[1:Ptrain] # the first Ptrain points become the training set
    return Xtrain,Ztrain
end
#The way these @everywhere functions are defined ensure the non-overlapping of the testing and training sets
@everywhere function extract_TestSet(X,Z,Ptest)
    Xtest = X[end-Ptest+1:end] ; Ztest = Z[end-Ptest+1:end] # the last Ptest points become the testing set
    return Xtest,Ztest
end

## Construction of Gram Matrices (Covariance Matrices for each couple of points)
@everywhere function ConstructGramMatrices(X,νT,νS,string="Both")
    P = length(X)
    if string == "Teacher"
        KT = ones(P,P)
        for i in 1:P
            for j in i+1:P
                KT[i,j] = KT[j,i] = k(dist(X[i],X[j]),νT)
            end
        end
        # One "tags" the matrix as Symmetric for efficiency in later computations
        # Note that it is even definite positive since it arises from a Matérn kernel
        GramT = Symmetric(KT)
        return EnforcePDness(GramT)
    elseif string == "Student"
        KS = ones(P,P)
        for i in 1:P
            for j in i+1:P
                KS[i,j] = KS[j,i] = k(dist(X[i],X[j]),νS)
            end
        end
        # One "tags" the matrix as Symmetric for efficiency in later computations
        # Note that it is even definite positive since it arises from a Matérn kernel
        GramS = Symmetric(KS)
        return EnforcePDness(GramS)
    elseif string == "Both"
        KS = ones(P,P) ; KT = ones(P,P)
        for i in 1:P
            for j in i+1:P
                KS[i,j] = KS[j,i] = k(dist(X[i],X[j]),νS)
                KT[i,j] = KT[j,i] = k(dist(X[i],X[j]),νT)
            end
        end
        # One "tags" the matrix as Symmetric for efficiency in later computations
        # Note that it is even definite positive since it arises from a Matérn kernel
        GramS = Symmetric(KS) ; GramT = Symmetric(KT)
        return EnforcePDness(GramT),EnforcePDness(GramS)
    else println("String Unknown. Choose among \"Teacher\",\"Student\" or \"Both\" ")
    end
end

@everywhere function EnforcePDness(A) # Note : PDness = positive-definiteness of a matrix A
    # Sometimes, errors in floating point in computations make the Gram matrix (numerically) not positive definite. In order to deal with this issue, one employs jitter.
    # For small jitter values (τ²), the properties of the rectified matrix do not change significantly
    # For consistency, we employ jitter even when unnecessary so that all simulations are done the same way

    τ2 = 1e-13 # small enough not to be seen and to work for all tested cases
    A_jitter = A + τ2*I(size(A)[1])
    @assert isposdef(A_jitter) ## if does not work, increase τ2 by a factor 10, and try again !
    return A_jitter
end
## Optimisation Algorithms
@everywhere function GradientDescent(Xtrain,Ztrain,Xtest,Ztest,νT,νS,epochs=Int(1E4),η=0.01)
    epochs_saved = ceil.(10 .^(range(1,stop=log10(epochs),length=10)))
    Ptrain = length(Xtrain) ; Ptest = length(Xtest)
    GramS = ConstructGramMatrices(Xtrain,νT,νS,"Student") # Construct Student Gram Matrix (Covariance Matrix for each couple of points in training set)
    test_err_exact = norm(analytic_prediction(Xtrain,Ztrain,Xtest,GramS,νS) - Ztest)^2 / Ptest

    ## Learning Procedure
    a = zeros(Ptrain) # initial weights, to be optimized
    test_err = [norm(predict(Xtest,Xtrain,a,νS) - Ztest)^2 / Ptest]
    for i in 1:epochs
        if i in epochs_saved
            prediction = predict(Xtest,Xtrain,a,νS)   # Compute the prediction
            append!(test_err,norm(prediction - Ztest)^2 / Ptest) # Compute the train error wrt prediction
        end
        grad_err = dMSE(a,Ztrain,GramS)
        a = a - η*grad_err
    end

    return test_err, test_err_exact , epochs
end
@everywhere function ConjugateGradientDescent(Xtrain,Ztrain,Xtest,Ztest,νT,νS,tolerance=1E-5)
    Ptrain = length(Xtrain) ; Ptest = length(Xtest)

    GramS = ConstructGramMatrices(Xtrain,νT,νS,"Student")

    test_err_exact = norm(analytic_prediction(Xtrain,Ztrain,Xtest,GramS,νS) - Ztest)^2 / Ptest

    ## Learning Procedure
    a = zeros(Ptrain) # initial weights, to be optimized
    test_err = []

    epochs = 0
    r = Ztrain - GramS*a # initial residual
    p = r                # initial direction
    while norm(r) > tolerance # norm(r)/Ptrain is equivalent to the train loss
        # Compute the train error wrt prediction
            ## Note : computed every iteration, not very efficient, but since the
            ## final number of epochs  is unknown, I haven't found a better solution.
        prediction = predict(Xtest,Xtrain,a,νS)
        append!(test_err,norm(prediction - Ztest)^2 / Ptest)

        # CGD Algo, from https://www.wikiwand.com/en/Conjugate_gradient_method#/The_resulting_algorithm
        α = (r' * r) / (p' * GramS * p)  # equivalent of learning rate η
        a = a + α*p
        rnew = r - α*GramS*p
        β = (rnew' * rnew) / ( r' * r)   # another equivalent of learning rate η
        p = rnew + β*p

        r = rnew  # update
        epochs = epochs+1
    end # while

    ## Returns :
        ## the test error at 0% of training, at ~10%, ~20%, ~30%, ... and at 100% = end of training. (11 numbers)
        ## the exact Kriging error (1 number)
        ## the number of epochs needed for convergence
    return test_err[Int.(round.(10 .^(range(0,stop=log10(epochs),length=11))))], test_err_exact, epochs
end

## Runs for a fixed number of epochs. Useful for benchmarks
@everywhere function ConjugateGradientDescentFixedEpochs(Xtrain,Ztrain,Xtest,Ztest,νT,νS,epochs=500)
    Ptrain = length(Xtrain) ; Ptest = length(Xtest)

    GramS = ConstructGramMatrices(Xtrain,νT,νS,"Student")

    test_err_exact = norm(analytic_prediction(Xtrain,Ztrain,Xtest,GramS,νS) - Ztest)^2 / Ptest

    ## Learning Procedure
    a = zeros(Ptrain) # initial weights, to be optimized
    test_err = [] ; train_err = []

    r = Ztrain - GramS*a # initial residual
    p = r                # initial direction
    for i in 1:epochs
        # Compute the train error wrt prediction
        prediction = predict(Xtest,Xtrain,a,νS)
        append!(test_err,norm(prediction - Ztest)^2 / Ptest)
        append!(train_err,norm(r)/Ptrain)
        # CGD Algo, from https://www.wikiwand.com/en/Conjugate_gradient_method#/The_resulting_algorithm
        α = (r' * r) / (p' * GramS * p)  # equivalent of learning rate η
        anew = a + α*p
        rnew = r - α*GramS*p
        β = (rnew' * rnew) / ( r' * r)   # another equivalent of learning rate η
        pnew = rnew + β*p
        p = pnew ; r = rnew ; a = anew   # update
    end
    return test_err, test_err_exact, train_err
end

## Run function : scans over P and scans over several teachers to get errorbars
@everywhere function Run(PP,Ptest,νT,νS,nb_teacher,algo,dimension)
    @assert algo in ["GD","CGD"] println(" '$algo' Minimization Algorithm Unknown. Choose among 'GD' or 'CGD'.")

    ## 3D Matrices to store data // dim 1 : epochs //  dim 2 : P //  dim 3 : Teacher
        test_err_matrix  = NaN*zeros(11,length(PP),maximum(nb_teacher))
        exact_err_matrix = NaN*zeros(1 ,length(PP),maximum(nb_teacher))
        epochs_matrix    = NaN*zeros(1,length(PP),maximum(nb_teacher))# number of epochs CGD needed to converge

    for i in eachindex(PP) # scan over P
        Ptrain = PP[i]
        for j in 1:nb_teacher[i] # scan over Teachers
            X           = generate_X(Ptrain,Ptest,dimension,νT,νS)
            GramT       = ConstructGramMatrices(X,νT,νS,"Teacher")
            Z           = generate_Z(X,GramT)
            Xtest,Ztest = extract_TestSet(X,Z,Ptest) # identical for all simulations and large enough to prevent statistical fluctuations
            Xtrain,Ztrain = extract_TrainSet(X,Z,Ptrain)

            ## Launch the simulations :
            println("$algo : ν = $νT, P = $(PP[i]), Teacher $j/$(nb_teacher[i]), Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now())))
            if     algo == "GD"  test_err_matrix[:,i,j],exact_err_matrix[1,i,j],epochs_matrix[1,i,j] = GradientDescent(Xtrain,Ztrain,Xtest,Ztest,νT,νS)
            elseif algo == "CGD" test_err_matrix[:,i,j],exact_err_matrix[1,i,j],epochs_matrix[1,i,j] = ConjugateGradientDescent(Xtrain,Ztrain,Xtest,Ztest,νT,νS)
            end
        end # end for loop over Teachers
    end # end for loop over P


    ## Save Data for later analysis
    # extension = " "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*"mn"
    extension = " "*string(Dates.day(now()))
    save("Data\\data__nu=$νT"*extension*".jld", "test_err", test_err_matrix, "exact_err", exact_err_matrix, "epochs", epochs_matrix)
end ## end function
