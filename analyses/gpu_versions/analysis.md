# Análise das variações da versão OpenMP

Inicialmente, é importante salientar que alterar o número de threads muda apenas a organização lógica de como a GPU agenda o 
trabalho aos seus processados. Além disso, o tamanho da base de dados define a carga total de trabalho. O K-Means é limitado pela 
memória, pois sua complexidade matemática é baixa e o problema de memória aumenta conforme o tamanho da base de dados. Com isso, 
a GPU passa a maior parte do tempo ociosa no aguardo de tráfego de dados. Ao alterar a configuração do bloco adicionando mais 
threads, o gargalo da largura de banda permanece inalterado.

Consequentemente, a eficiência piora vertiginosamente ao passo que o número de threads aumenta. Uma vez que o SpeedUp não se 
altera, pois o problema é limitado pela memória, aumentar o número de threads torna-se inútil, pois a cada aumento, cada thread 
irá trabalhar menos. Embora a carga de trabalho aumenta quando o número de clusters aumenta, isso não é o suficiente para fazer 
com que seja vantajoso aumentar o número de threads.