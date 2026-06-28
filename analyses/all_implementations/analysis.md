# Comparação das quatro implementações
	
No gráfico linear, é evidente a dificuldade encontrada pela versão sequencial dado o aumento do número de clusters. 
A complexidade assintótica do K-Means é dada por O(N x K x I), em que N é o número de pontos, K é o número de clusters e I 
é o número de iterações. Como todas implementações acessaram a mesma base de dados e todas chegaram ao mesmo número 
de iterações nos testes, foi o número de clusters que determinou a complexidade nessa simulação. Conforme o número de 
clusters cresce, o tempo da CPU salta de dois segundos para mais de 70 segundos. Ao operar sozinha, a CPU fica totalmente 
dependente do barramento de memória principal para buscar os pontos da base de dados repetidas vezes. No gráfico em escala 
logarítmica, é possível observar a hierarquia das soluções, em termos de tempo de execução.

Em ordem de desempenho, temos que a versão MPI foi a que obteve os melhores resultados, seguidos da versão do CUDA e da 
OpenMP-GPU, fazendo com que a versão sequencial fosse a pior em termos de resultados. As versões paralelas conseguiram 
resolver o problema mais complexo, para k igual a 13, em menos de 5 segundos, comprovando a eficácia do paralelismo para o 
problema. Em termos de SpeedUp, para valores pequenos de k, o ganho é menor por conta do tempo das transferências de dados, seja em  transferências host-device no caso GPU, seja na troca de mensagens através da rede/memória no caso do MPI. Conforme o valor 
de k cresce, a carga computacional domina os tempos de execução e o overhead é amortizado, fazendo com que o SpeedUp cresça.

No caso do MPI, o SpeedUp cresce drasticamente. Na versão sequencial, a base de dados não cabe na memória cache, forçando o 
processador a buscar os dados direto na memória RAM constantemente. No caso do MPI, o problema é dividido entre 16 processos, 
e em cada processo, o bloco ainda é dividido entre as threads. Sendo assim, cada processo fica com um pedaço que se torna 
acessível o suficiente para caber na cache. Sendo assim, o tempo que se ganha por não recorrer à memória RAM é altíssimo.