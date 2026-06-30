#!/bin/bash
#SBATCH --job-name=Kmeans_CUDA
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=kmeans_cuda-%j.out

# Aumenta o limite de pilha
ulimit -s unlimited

echo "=== INICIANDO JOB CUDA K-MEANS ==="

cd $SLURM_SUBMIT_DIR

# Carregamento do módulo do compilador da NVIDIA
module load compilers/nvidia/nvhpc/24.11

echo "Compilando o código fonte CUDA..."
# O compilador nvcc é o padrão para códigos CUDA. 
# A flag -arch=sm_70 otimiza a compilação especificamente para a arquitetura Volta (V100).
# Certifique-se de que a extensão dos arquivos com código de kernel seja .cu (ex: kmeans_core_cuda.cu)
nvcc -O3 -arch=sm_70 -Iinclude -o kmeans_cuda cuda.cu kmeans_core_cuda.cu utils.cpp

if [ $? -ne 0 ]; then
    echo "Erro de compilação! Parando o job."
    exit 1
fi

# Configurações do Teste (mantidas idênticas ao script sequencial)
DATASET="enem_dataset.csv" 
CLUSTERS=(2 3 4 6 7 9 11 13)
RUNS=3

# Arquivo final de saída
OUT_CSV="resultados_cuda_medias_novav.csv"
echo "Version,K,Dataset_Size,Avg_Iterations,Avg_Time_Sec" > $OUT_CSV

echo "Iniciando bateria de testes com $RUNS execuções por cluster na GPU..."
echo "--------------------------------------------------------"

for K in "${CLUSTERS[@]}"; do
    echo "Rodando K = $K ($RUNS vezes)..."
    
    TMP_FILE="tmp_k${K}.txt"
    > $TMP_FILE

    for ((i=1; i<=RUNS; i++)); do
        ./kmeans_cuda -input $DATASET -k $K | grep "^CSV_RESULT" >> $TMP_FILE
    done
    
    # Usa o AWK para ler o arquivo temporário e calcular a média
    awk -F',' '{
        sum_iter += $5;
        sum_time += $6;
        version = $2;
        k = $3;
        n = $4;
    } END {
        avg_iter = sum_iter / NR;
        avg_time = sum_time / NR;
        printf "%s,%s,%s,%.1f,%.6f\n", version, k, n, avg_iter, avg_time;
    }' $TMP_FILE >> $OUT_CSV

    rm $TMP_FILE
done

echo "--------------------------------------------------------"
echo "=== JOB FINALIZADO! Resultados salvos em $OUT_CSV ==="
