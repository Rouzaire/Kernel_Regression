cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Regression Teacher Student")
PP = [100,200,300] ; Ptest = 1000 ; νT = [0.5 0.75 1.0 1.5 2 4 20] ; νS = 1/2 ; nb_teacher_low = 10 ; nb_teacher_high = 10
PP = [200] ; Ptest = 1000 ; νT = [0.5 0.75 1.0 1.5 2 4 20] ; νS = 1/2 ; nb_teacher_low = 1 ; nb_teacher_high = 1 ## values for a first round
dimension = 10 ; algo = "GD"
teachers_matrix = []
for i in 1:length(νT)
    if νT[i] ≤ 1 append!(teachers_matrix,[Int.(floor.(10 .^(range(log10(nb_teacher_low) ,stop=0,length=length(PP)))))])
    else         append!(teachers_matrix,[Int.(floor.(10 .^(range(log10(nb_teacher_high),stop=0,length=length(PP)))))])
    end
end
teachers_matrix = 30*teachers_matrix
println("Number of different teachers : $(teachers_matrix)")



using Distributed
addprocs(length(νT))
@everywhere include("function_definitions.jl") ; @everywhere using Distributions, SpecialFunctions, LinearAlgebra, JLD, Dates
@time pmap(Run,[PP for i in eachindex(νT)],fill(Ptest,length(νT)),νT,fill(νS,length(νT)),teachers_matrix,fill(algo,length(νT)),fill(dimension,length(νT)))
save("Data\\parameters_day"*string(Dates.day(today()))*".jld", "PP", PP, "Ptest", Ptest, "νT", νT, "νS", νS, "teachers_matrix",teachers_matrix,"dimension",dimension,"algo",algo)
rmprocs(workers())
