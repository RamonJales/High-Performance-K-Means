#!/bin/bash
#SBATCH --job-name=Kmeans_MPI_OMP
#SBATCH --partition=intel-128
#SBATCH --hint=compute_bound
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=kmeans_mpi-%j.out

echo "=== INICIANDO JOB HÍBRIDO MPI + OPENMP ==="

cd $SLURM_SUBMIT_DIR

# 1. Configuração do ambiente MPI e OpenMP
# Mapeia as threads do OpenMP dinamicamente para o valor escolhido no SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# (Opcional) Carregue módulos de MPI caso não estejam no PATH por padrão no seu cluster
# module load openmpi

echo "Compilando o código fonte MPI + OpenMP..."
# Usamos mpic++ para compilar códigos híbridos. A flag -fopenmp ativa as diretivas.
mpic++ -O3 -march=native -fopenmp -Wall -Iinclude -o kmeans_mpi_omp mpi_openmp.cpp kmeans_core_mpi.cpp utils.cpp

if [ $? -ne 0 ]; then
    echo "Erro de compilação! Parando o job."
    exit 1
fi

# Configurações do Teste
DATASET="enem_dataset.csv" 
CLUSTERS=(2 3 4 6 7 9 11 13)
RUNS=3

# Arquivo final de saída
OUT_CSV="resultados_mpi_omp_medias.csv"
echo "Version,K,Dataset_Size,Avg_Iterations,Avg_Time_Sec" > $OUT_CSV

echo "--------------------------------------------------------"
echo "Nós Alocados: $SLURM_JOB_NUM_NODES"
echo "Processos MPI Totais: $SLURM_NTASKS (1 por nó)"
echo "Threads OpenMP por Processo: $OMP_NUM_THREADS"
echo "Iniciando bateria de testes com $RUNS execuções por cluster..."
echo "--------------------------------------------------------"

for K in "${CLUSTERS[@]}"; do
    echo "Rodando K = $K ($RUNS vezes)..."
    
    TMP_FILE="tmp_k${K}.txt"
    > $TMP_FILE

    for ((i=1; i<=RUNS; i++)); do
        # O mpirun lê a variável do SLURM_NTASKS automaticamente para definir o número de processos
        mpirun -np $SLURM_NTASKS ./kmeans_mpi_omp -input $DATASET -k $K | grep "^CSV_RESULT" >> $TMP_FILE
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
