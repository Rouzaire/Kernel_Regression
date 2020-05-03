PP = [30,50,100,200,300,500] ; Ptest = 500 ; νT = [0.5 1.0 2.0 4.0 8.0] ; νS = 1/2 ; nb_teacher_low = 15 ; nb_teacher_high = 100
dimension = 1 ; algo = "GD"
teachers_matrix = []
for i in 1:length(νT)
    if νT[i] ≤ 1 append!(teachers_matrix,[Int.(floor.(10 .^(range(log10(nb_teacher_low) ,stop=0,length=length(PP)))))])
    else         append!(teachers_matrix,[Int.(floor.(10 .^(range(log10(nb_teacher_high),stop=0,length=length(PP)))))])
    end
end
println("Number of different teachers : $(teachers_matrix)")

using Distributed
addprocs(length(νT))
@everywhere include("function_definitions.jl") ; @everywhere using Distributions, SpecialFunctions, LinearAlgebra, JLD, Dates
@time pmap(Run,[PP for i in eachindex(νT)],fill(Ptest,length(νT)),νT,fill(νS,length(νT)),teachers_matrix,fill(algo,length(νT)),fill(dimension,length(νT)))
save("Data\\parameters_day"*string(Dates.day(today()))*".jld", "PP", PP, "Ptest", Ptest, "νT", νT, "νS", νS, "teachers_matrix",teachers_matrix,"dimension",dimension,"algo",algo)
rmprocs(workers())
