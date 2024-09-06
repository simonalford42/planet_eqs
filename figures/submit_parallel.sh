#!/usr/bin/env bash

# Initialize variables with default values
Ngrid=""
total_processes=""
std_flag=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --Ngrid)
            Ngrid="$2"
            shift 2
            ;;
        --total)
            total_processes="$2"
            shift 2
            ;;
        --std)
            std_flag="--std"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$Ngrid" ] || [ -z "$total_processes" ]; then
    echo "Usage: $0 --Ngrid <Ngrid> --total <total_processes> [--std]"
    exit 1
fi

# Loop through and submit jobs
for ix in $(seq 0 $((total_processes - 1))); do
    sbatch -J "p${ix}_${Ngrid}" --partition gpu run.sh --Ngrid "$Ngrid" --version 24880 --parallel_ix "$ix" --parallel_total "$total_processes" --compute
done

echo "Submitted $total_processes jobs with Ngrid=$Ngrid${std_flag:+ and --std flag}"
