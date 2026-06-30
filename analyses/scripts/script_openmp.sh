#!/bin/bash
#SBATCH --job-name=Kmeans_OMP_GPU_Threads
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=kmeans_gpu-%j.out

ulimit -s unlimited

echo "=== INICIANDO JOB OPENMP-GPU (TESTE DE THREADS) ==="
cd $SLURM_SUBMIT_DIR

module load compilers/nvidia/nvhpc/24.11

echo "Compilando o código fonte..."
nvc++ -fast -mp=gpu -Wall -Iinclude -o kmeans_gpu openmp_gpu.cpp kmeans_core_gpu.cpp utils.cpp

if [ $? -ne 0 ]; then
    echo "Erro de compilação! Parando o job."
    exit 1
fi

DATASET="enem_dataset.csv"
# Sugestão: Reduza os valores de K se o job demorar muito, pois faremos combinações.
CLUSTERS=(2 3 4 6 7 9 11 13)
THREADS=(2 4 8 16 32 64 128 256)
RUNS=3

OUT_CSV="resultados_gpu_novav.csv"
echo "Version,K,Dataset_Size,Threads_Per_Block,Avg_Iterations,Avg_Time_Sec" > $OUT_CSV

echo "--------------------------------------------------------"
echo "Iniciando Matriz de Testes (K x Threads)..."

for K in "${CLUSTERS[@]}"; do
    for T in "${THREADS[@]}"; do
        echo "Rodando K = $K | Threads = $T ($RUNS vezes)..."
        
        TMP_FILE="tmp_k${K}_t${T}.txt"
        > $TMP_FILE

        for ((i=1; i<=RUNS; i++)); do
            # Note a adição do parâmetro -threads $T e o novo grep
            ./kmeans_gpu -input $DATASET -k $K -threads $T | grep "^CSV_RESULT_THREADS" >> $TMP_FILE
        done
        
        # O AWK agora lê 7 colunas em vez de 6
        awk -F',' '{
            sum_iter += $6;
            sum_time += $7;
            version = $2;
            k = $3;
            n = $4;
            threads = $5;
        } END {
            if (NR > 0) {
                avg_iter = sum_iter / NR;
                avg_time = sum_time / NR;
                printf "%s,%s,%s,%s,%.1f,%.6f\n", version, k, n, threads, avg_iter, avg_time;
            }
        }' $TMP_FILE >> $OUT_CSV

        rm $TMP_FILE
    done
done

echo "--------------------------------------------------------"
echo "=== JOB FINALIZADO! Resultados salvos em $OUT_CSV ==="
