# Comparação entre as versões OpenMP e CUDA

A diferença entre os tempos de execução é notável. Enquanto a implementação em CUDA consegue resolver os cenários mais complexos 
em menos de um segundo, a versão OpenMP oscila entre um e cinco segundos. Por conta da limitação de memória,  o gargalo é a 
transferência de dados. Em CUDA, tem-se o controle absoluto sobre a alocação da memória compartilhada e otimização de 
registradores. A utilização de diretivas no OpenMP faz com que o compilador gere um kernel genérico que muitas vezes não consegue 
mapear a memória de forma otimizada e específica. O CUDA oferece a possibilidade de ter o controle de tudo, fazendo com que um 
código bom nessa linguagem seja melhor que um código excelente que utiliza OpenMP.

O gráfico de SpeedUp quantifica a superioridade do CUDA em relação ao OpenMP, levando vantagem em todos os valores de k. Esse 
comportamento evidencia o que foi dito anteriormente em relação às duas abordagens. Também é possível identificar, de forma mais 
clara, o determinismo do algoritmo, uma vez que as implementações convergiram no número de iterações. Além disso, quando há um 
pico de lentidão no CUDA, por exemplo, também há um pico no OpenMP, evidenciando o equilíbrio da implementação do algoritmo 
proposto.