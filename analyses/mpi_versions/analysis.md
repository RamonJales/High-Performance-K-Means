# Análise das variações da versão MPI

Como discutido anteriormente, o particionamento dos dados com MPI, somado a utilização de threads em cada processo, faz com que 
os blocos de dados sejam acessíveis e que o ganho de velocidade em ler a cache seja altíssimo, a ponto de ser possível observar 
um SpeedUp Superlinear, por ultrapassar os limites matemáticos básicos de escalabilidade. Esse fenômeno é refletido na 
eficiência. Como o SpeedUp dispara, a eficiência acompanha esse crescimento, pois o ganho é muito alto, comprovando que o 
hardware foi subutilizado na versão sequencial devido ao estrangulamento da memória.

As curvas não crescem de forma igual, o ganho é mais conversador para valores menores de k em relação aos valores maiores. O 
padrão de mensagens do MPI tem um custo de latência muito mais alto do que a sincronização de threads nativas. Sendo assim, 
quando o valor de k é baixo, a CPU realiza os cálculos de maneira muito mais rápida. Portanto, a comunicação entre os processos 
se torna um gargalo. Quando o valor de k é alto, o volume de dados aumenta, logo, o tempo adicional para realização dos cálculos 
ofusca o overhead da troca de mensagens do MPI, fazendo com que o paralelismo seja totalmente eficiente.