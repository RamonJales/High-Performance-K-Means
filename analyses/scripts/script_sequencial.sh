#!/bin/bash
#SBATCH --job-name=Kmeans_Seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=intel-128
#SBATCH --hint=compute_bound
#SBATCH --output=kmeans_seq-%j.out

echo "=== INICIANDO JOB SEQUENCIAL K-MEANS ==="

cd $SLURM_SUBMIT_DIR

echo "Compilando o código fonte sequencial..."
g++ -O3 -march=native -Wall -Iinclude -o kmeans_seq sequential.cpp kmeans_core.cpp utils.cpp

if [ $? -ne 0 ]; then
    echo "Erro de compilação! Parando o job."
    exit 1
fi

# Configurações do Teste
DATASET="enem_dataset.csv"
CLUSTERS=(2 3 4 6 7 9 11 13)
RUNS=3

# Arquivo final de saída
OUT_CSV="resultados_sequencial_medias.csv"
echo "Version,K,Dataset_Size,Avg_Iterations,Avg_Time_Sec" > $OUT_CSV

echo "Iniciando bateria de testes com $RUNS execuções por cluster..."
echo "--------------------------------------------------------"

for K in "${CLUSTERS[@]}"; do
    echo "Rodando K = $K ($RUNS vezes)..."

    # Arquivo temporário para guardar os resultados brutos deste K
    TMP_FILE="tmp_k${K}.txt"
    > $TMP_FILE

    for ((i=1; i<=RUNS; i++)); do
        # Roda o executável e extrai apenas a linha que começa com CSV_RESULT
        ./kmeans_seq -input $DATASET -k $K | grep "^CSV_RESULT" >> $TMP_FILE
    done

    # Usa o AWK para ler o arquivo temporário, somar as colunas 5 (Iterações) e 6 (Tempo), e calcular a média
    awk -F',' '{
        sum_iter += $5;
        sum_time += $6;
        version = $2;
        k = $3;
        n = $4;
    } END {
        avg_iter = sum_iter / NR;
        avg_time = sum_time / NR;
        # Imprime a linha formatada direto no CSV final
        printf "%s,%s,%s,%.1f,%.6f\n", version, k, n, avg_iter, avg_time;
    }' $TMP_FILE >> $OUT_CSV

    # Limpa o arquivo temporário
    rm $TMP_FILE
done

echo "--------------------------------------------------------"
echo "=== JOB FINALIZADO! Resultados salvos em $OUT_CSV ==="
