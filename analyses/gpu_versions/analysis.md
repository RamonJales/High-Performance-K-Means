# Análise das variações da versão OpenMP

Inicialmente, é importante salientar que alterar o número de threads muda primordialmente a organização lógica de como a GPU agenda o 
trabalho aos seus processadores/núcleos de execução, afetando minimamente partes de processamento. Além disso, o tamanho da base de dados define a carga 
total de trabalho. No caso do K-Means, a etapa mais custosa tende a ser a associação de cada ponto ao centroide mais próximo, pois 
cada ponto precisa ser comparado com todos os clusters. Logo, o K-Means é limitado pela memória, pois sua complexidade matemática é 
baixa e o problema de memória aumenta conforme o tamanho da base de dados. Com isso, a GPU passa a maior parte do tempo ociosa no 
aguardo de tráfego de dados. Ao alterar a configuração do bloco adicionando mais threads, o gargalo da largura de banda permanece inalterado.

Consequentemente, a eficiência piora de forma acentuada ao passo que o número de threads aumenta. Uma vez que o SpeedUp não se 
altera, pois o problema é limitado pela memória, aumentar o número de threads torna-se inútil, pois a GPU já possui paralelismo 
suficiente para ocupar seus recursos disponíveis, e novos aumentos apenas reduzem a quantidade de trabalho útil por thread, 
ou aumentam custos indiretos de escalonamento, sincronização e acesso à memória. Embora a carga de trabalho aumenta quando o 
número de clusters aumenta, isso não é o suficiente para fazer com que seja vantajoso aumentar o número de threads. Ainda assim, a 
queda de eficiência não significa necessariamente que a GPU está “piorando” em termos absolutos, mas sim que o aumento artificial do
número de threads não gerou ganho proporcional de desempenho. Para a versão em GPU, métricas como tempo total, throughput, ocupação
da GPU e custo de transferência de memória são mais representativas do que apenas a eficiência calculada por thread.