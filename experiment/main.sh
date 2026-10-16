#!/usr/bin/env bash
set -e # Exit immediately if a command fails.
# set -x 
export PYTENSOR_FLAGS="compile__timeout=600,compile__wait=5"
export PYTHONWARNINGS="ignore::FutureWarning"

# nrepeat=100
ncore="${NCORES:-20}"
runtool="${RUNTOOL:-uv run}"
# runtool="python"

#-------------------------- scenario1001 without Z ------------------------------
# seed=0
# n_samples=(100)
# HRs=(1 1.5 2 2.5 3)
# p_censor=(0.1 0.4 0.7)
# rhos=(1)

# for rho in ${rhos[@]}; do
#      for n in ${n_samples[@]}; do
#           for pc in ${p_censor[@]}; do
#                for HR in ${HRs[@]}; do
#                     seed=$((seed+1))
#                     bx=$(awk -v x=$HR 'BEGIN {print log(x)}')
#                     echo "<====scenario1001======> n=$n, HR=$HR, pc=$pc, rho=$rho, seed=$seed"
#                     data_dir=./scenario1001/data/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir1=./scenario1001/result_exp/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir2=./scenario1001/result_pem2/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir3=./scenario1001/result_pem4/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir4=./scenario1001/result_true_s/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir5=./scenario1001/result_wstandar_global_rfbbp/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir6=./scenario1001/result_wstandar_study_rfbbp/wo_z_${n}_log${HR}_${pc}_${rho}
#                     eval_fn=eval_results.csv

#                     # $runtool main.py simulate -od $data_dir --seed $seed \
#                     # --n_samples $n --beta_x $bx -nr $nrepeat --p_censor $pc --rho $rho 

#                     # echo "[ANALYZE: EXP] Exponential baseline, no cutpoints"
#                     # $runtool main.py analyze -dd $data_dir -od $ana_dir1 -nc $ncore --target_accept 0.9
#                     # $runtool main.py evaluate -ad $ana_dir1 -of $eval_fn

#                     # echo "[ANALYZE: PEM2] PEM baseline, auto cutpoints, max_cuts=2"
#                     # $runtool main.py analyze -dd $data_dir -od $ana_dir2 -nc $ncore --target_accept 0.9 --auto_cutpoints --max_cuts 2
#                     # $runtool main.py evaluate -ad $ana_dir2 -of $eval_fn

#                     # echo "[ANALYZE: PEM4] PEM baseline, auto cutpoints, max_cuts=4"
#                     # $runtool main.py analyze -dd $data_dir -od $ana_dir3 -nc $ncore --target_accept 0.9 --auto_cutpoints --max_cuts 4
#                     # $runtool main.py evaluate -ad $ana_dir3 -of $eval_fn
                    
#                     # $runtool main.py analyze -dd $data_dir -od $ana_dir4 -nc $ncore --methods xtrue_c xtrue_cs xtrue_ps wstandar wstandar_s naive naive_s
#                     # $runtool main.py evaluate -ad $ana_dir4 -of $eval_fn

#                     # $runtool main.py analyze -dd $data_dir -od $ana_dir5 -nc $ncore --target_accept 0.9 --standardize_w global
#                     # $runtool main.py evaluate -ad $ana_dir5 -of $eval_fn

#                     # $runtool main.py analyze -dd $data_dir -od $ana_dir6 -nc $ncore --target_accept 0.9 --standardize_w study
#                     # $runtool main.py evaluate -ad $ana_dir6 -of $eval_fn
#                done
#           done
#      done
# done

# $runtool main.py summarize -efp "./scenario1001/result_exp/wo_z_*/eval_results.csv" -of ./scenario1001/result_exp/summary_1001_exp.xlsx \
#      -sp rho p_censor betax

# $runtool main.py summarize -efp "./scenario1001/result_pem2/wo_z_*/eval_results.csv" -of ./scenario1001/result_pem2/summary_1001_pem2.xlsx \
#      -sp rho p_censor betax

# $runtool main.py summarize -efp "./scenario1001/result_pem4/wo_z_*/eval_results.csv" -of ./scenario1001/result_pem4/summary_1001_pem4.xlsx \
#      -sp rho p_censor betax

# $runtool main.py summarize -efp "./scenario1001/result_true_s/wo_z_*/eval_results.csv" -of ./scenario1001/result_true_s/summary_1001.xlsx \
#      -sp rho p_censor betax

# $runtool main.py summarize -efp "./scenario1001/result_wstandar_global_rfbbp/wo_z_*/eval_results.csv" -of ./scenario1001/result_wstandar_global_rfbbp/summary_1001.xlsx \
#      -sp rho p_censor betax

# $runtool main.py summarize -efp "./scenario1001/result_wstandar_study_rfbbp/wo_z_*/eval_results.csv" -of ./scenario1001/result_wstandar_study_rfbbp/summary_1001.xlsx \
#      -sp rho p_censor betax


#-------------------------- scenario1002 without Z ------------------------------
# nrepeat=100
# seed=0
# n_samples=(100 200 500 1000)
# HRs=(2 2.5 3)
# p_censor=(0.1)
# # rhos=(1)
# rhos=(1.5)

# for rho in ${rhos[@]}; do
#      for n in ${n_samples[@]}; do
#           for pc in ${p_censor[@]}; do
#                for HR in ${HRs[@]}; do
#                     seed=$((seed+1))
#                     bx=$(awk -v x=$HR 'BEGIN {print log(x)}')
#                     echo "<====scenario1002======> n=$n, HR=$HR, pc=$pc, rho=$rho, seed=$seed"
#                     data_dir=./scenario1002/data_sigma2e_05/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir1=./scenario1002/result_beta05_a1_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir2=./scenario1002/result_beta1_a1_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir3=./scenario1002/result_study_beta1_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir4=./scenario1002/result_pem2_study_beta1_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir5=./scenario1002/result_pem4_study_beta1_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir6=./scenario1002/result_pem6_study_beta1_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir7=./scenario1002/result_pemauto1_study_beta1_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir8=./scenario1002/result_pem8_study_beta100_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir9=./scenario1002/result_pem8_study_beta1000_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     ana_dir10=./scenario1002/result_pem6_study_beta100_a05_l05_ig31/wo_z_${n}_log${HR}_${pc}_${rho}
#                     eval_fn=eval_results.csv

                    # $runtool main.py simulate -od $data_dir --seed $seed \
                    # --n_samples $n --beta_x $bx -nr $nrepeat --p_censor $pc --rho $rho --sigma2_e 0.5 

                    # $runtool main.py analyze -dd $data_dir -od $ana_dir1 -nc $ncore --target_accept 0.9 --standardize_w global \
                    # --betax_sigma0 0.5 --a_sigma0 1.0 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0

                    # $runtool main.py analyze -dd $data_dir -od $ana_dir2 -nc $ncore --target_accept 0.9 --standardize_w global \
                    # --betax_sigma0 1.0 --a_sigma0 1.0 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0
                    # if [[ "$n" == "200" || "$n" == "1000" ]]; then
                    #      echo "Skip n=$n"
                    #      continue
                    # fi
                    # Performs reasonably well
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir3 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 1.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0
                    # $runtool main.py evaluate -ad $ana_dir3 -of $eval_fn

                    # # rho=1.5
                    # echo "[ANALYZE: PEM2] PEM baseline, auto cutpoints, max_cuts=2"
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir4 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 1.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
                    # --auto_cutpoints --max_cuts 2
                    # $runtool main.py evaluate -ad $ana_dir4 -of $eval_fn

                    # echo "[ANALYZE: PEM4] PEM baseline, auto cutpoints, max_cuts=4"
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir5 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 1.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
                    # --auto_cutpoints --max_cuts 4
                    # $runtool main.py evaluate -ad $ana_dir5 -of $eval_fn

                    # echo "[ANALYZE: PEM6] PEM baseline, auto cutpoints, max_cuts=6"
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir6 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 1.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
                    # --auto_cutpoints --max_cuts 6
                    # $runtool main.py evaluate -ad $ana_dir6 -of $eval_fn

                    # echo "[ANALYZE: PEMauto1] PEM baseline, auto cutpoints, max_cuts=30"
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir7 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 1.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
                    # --auto_cutpoints --study_event_cutpoint_constraint --max_cuts 30 --min_events_per_interval 10
                    # $runtool main.py evaluate -ad $ana_dir7 -of $eval_fn

                    # echo "[ANALYZE: PEM8] PEM baseline, auto cutpoints, max_cuts=8"
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir8 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 100.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
                    # --auto_cutpoints --max_cuts 8
                    # $runtool main.py evaluate -ad $ana_dir8 -of $eval_fn

                    # echo "[ANALYZE: PEM8] PEM baseline, auto cutpoints, max_cuts=8"
                    # $runtool main.py analyze -dd $data_dir -od $ana_dir9 -nc $ncore --target_accept 0.9 --standardize_w study \
                    # --betax_sigma0 1000.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
                    # --auto_cutpoints --max_cuts 8
                    # $runtool main.py evaluate -ad $ana_dir9 -of $eval_fn

                    # echo "[ANALYZE: PEM6] PEM baseline, auto cutpoints, max_cuts=6"
#                     $runtool main.py analyze -dd $data_dir -od $ana_dir10 -nc $ncore --target_accept 0.9 --standardize_w study \
#                     --betax_sigma0 100.0 --a_sigma0 0.5 --l_sigma0 0.5 --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
#                     --auto_cutpoints --max_cuts 6
#                     $runtool main.py evaluate -ad $ana_dir10 -of $eval_fn
#                done
#           done
#      done
# done

# $runtool main.py summarize -efp "./scenario1002/result_beta1_a1_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_beta1_a1_l05_ig31/summary_1002_exp.xlsx \
#      -sp n_samples p_censor betax
# $runtool main.py summarize -efp "./scenario1002/result_study_beta1_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_study_beta1_a05_l05_ig31/summary_1002_exp.xlsx \
#      -sp n_samples p_censor betax
# $runtool main.py summarize -efp "./scenario1002/result_pem2_study_beta1_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pem2_study_beta1_a05_l05_ig31/summary_1002_pem2.xlsx \
#      -sp n_samples p_censor betax
# $runtool main.py summarize -efp "./scenario1002/result_pem4_study_beta1_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pem4_study_beta1_a05_l05_ig31/summary_1002_pem4.xlsx \
#      -sp n_samples p_censor betax
# $runtool main.py summarize -efp "./scenario1002/result_pem6_study_beta1_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pem6_study_beta1_a05_l05_ig31/summary_1002_pem6.xlsx \
#      -sp n_samples p_censor betax

# $runtool main.py summarize -efp "./scenario1002/result_pemauto1_study_beta1_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pemauto1_study_beta1_a05_l05_ig31/summary_1002_pemauto1.xlsx \
#      -sp n_samples p_censor betax

# $runtool main.py summarize -efp "./scenario1002/result_pem6_study_beta100_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pem6_study_beta100_a05_l05_ig31/summary_1002_pem6.xlsx \
#      -sp n_samples p_censor betax
# $runtool main.py summarize -efp "./scenario1002/result_pem6_study_beta1000_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pem6_study_beta1000_a05_l05_ig31/summary_1002_pem6.xlsx \
#      -sp n_samples p_censor betax

# $runtool main.py summarize -efp "./scenario1002/result_pem6_study_beta100_a05_l05_ig31/wo_z_*/eval_results.csv" -of ./scenario1002/result_pem6_study_beta100_a05_l05_ig31/summary_1002_pem6.xlsx \
#      -sp n_samples p_censor betax


#-------------------------- scenario200x RFBBP-PEM simulation suite ------------------------------
# Set to one of: 2001, 2002, 2003, 2004, 2005, all
# run_scenario="${RUN_SCENARIO:-all}"
# nrepeat=20
# seed=2000
# n_samples=(100 300 500)
# HRs=(1.25 1.5 1.75 2 2.25)
# p_censor_main=(0.2 0.5)
# study_n=3
# a_vals=(-0.5 0 0.5)
# b_vals=(0.7 1.0 1.3)
# sigma2e_main=0.25
# pem_cutpoints=(0 1 2 3 4)
# pem_increasing=(0.05 0.08 0.12 0.18 0.25)
# eval_fn=eval_results.csv

# run_rfbbp_pem_analysis() {
#      local data_dir=$1
#      local ana_dir=$2
#      if [[ ! -d "$ana_dir" ]]; then
#           $runtool main.py analyze -dd "$data_dir" -od "$ana_dir" -nc $ncore \
#           --target_accept 0.9 --standardize_w study \
#           --betax_sigma0 2.5 --a_sigma0 0.5 --l_sigma0 0.5 \
#           --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
#           --auto_cutpoints --study_event_cutpoint_constraint --max_cuts 20 --min_events_per_interval 10
#           $runtool main.py evaluate -ad "$ana_dir" -of "$eval_fn"
#      else
#           echo "[SKIP ANALYZE] $ana_dir already exists"
#      fi
# }

# simulate_pem_if_needed() {
#      local data_dir=$1
#      local n=$2
#      local HR=$3
#      local pc=$4
#      local hazards=("${@:5}")
#      local bx
#      bx=$(awk -v x=$HR 'BEGIN {print log(x)}')
#      seed=$((seed+1))
#      if [[ ! -d "$data_dir" ]]; then
#           $runtool main.py simulate -od "$data_dir" --seed "$seed" \
#           --survival_model pem --n_studies $study_n --n_samples "$n" \
#           --a "${a_vals[@]}" --b "${b_vals[@]}" --sigma2_e "$sigma2e_main" \
#           --beta_0 0 0 0 --beta_x "$bx" -nr $nrepeat --p_censor "$pc" \
#           --pem_cutpoints "${pem_cutpoints[@]}" --pem_hazards "${hazards[@]}"
#      else
#           echo "[SKIP SIMULATE] $data_dir already exists"
#      fi
# }

# if [[ "$run_scenario" == "2001" || "$run_scenario" == "all" ]]; then
#      for n in "${n_samples[@]}"; do
#           for pc in "${p_censor_main[@]}"; do
#                for HR in "${HRs[@]}"; do
#                     echo "<====scenario2001 PEM increasing truth======> n=$n, HR=$HR, pc=$pc"
#                     data_dir=./scenario2001/data/wo_z_${n}_log${HR}_${pc}_pem_inc
#                     ana_dir=./scenario2001/result_rfbbp_pem/wo_z_${n}_log${HR}_${pc}_pem_inc
#                     simulate_pem_if_needed "$data_dir" "$n" "$HR" "$pc" "${pem_increasing[@]}"
#                     run_rfbbp_pem_analysis "$data_dir" "$ana_dir"
#                done
#           done
#      done
#      $runtool main.py summarize -efp "./scenario2001/result_rfbbp_pem/wo_z_*/eval_results.csv" -of ./scenario2001/result_rfbbp_pem/summary_2001.xlsx \
#           -sp n_samples p_censor betax
# fi

# if [[ "$run_scenario" == "2002" || "$run_scenario" == "all" ]]; then
#      baseline_names=(constant increasing decreasing nonmonotone)
#      baseline_values=(
#           "0.12 0.12 0.12 0.12 0.12"
#           "0.05 0.08 0.12 0.18 0.25"
#           "0.25 0.18 0.12 0.08 0.05"
#           "0.05 0.18 0.08 0.25 0.12"
#      )
#      for idx in "${!baseline_names[@]}"; do
#           name=${baseline_names[$idx]}
#           read -r -a hazards <<< "${baseline_values[$idx]}"
#           for n in "${n_samples[@]}"; do
#                for pc in "${p_censor_main[@]}"; do
#                     for HR in "${HRs[@]}"; do
#                          echo "<====scenario2002 baseline shape======> shape=$name, n=$n, HR=$HR, pc=$pc"
#                          data_dir=./scenario2002/data_${name}/wo_z_${n}_log${HR}_${pc}
#                          ana_dir=./scenario2002/result_rfbbp_pem_${name}/wo_z_${n}_log${HR}_${pc}
#                          simulate_pem_if_needed "$data_dir" "$n" "$HR" "$pc" "${hazards[@]}"
#                          run_rfbbp_pem_analysis "$data_dir" "$ana_dir"
#                     done
#                done
#           done
#           $runtool main.py summarize -efp "./scenario2002/result_rfbbp_pem_${name}/wo_z_*/eval_results.csv" -of ./scenario2002/result_rfbbp_pem_${name}/summary_2002_${name}.xlsx \
#                -sp n_samples p_censor betax
#      done
# fi

# if [[ "$run_scenario" == "2003" || "$run_scenario" == "all" ]]; then
#      cont_models=(weibull loglogistic gompertz)
#      for model in "${cont_models[@]}"; do
#           for n in "${n_samples[@]}"; do
#                for pc in "${p_censor_main[@]}"; do
#                     for HR in "${HRs[@]}"; do
#                          bx=$(awk -v x=$HR 'BEGIN {print log(x)}')
#                          seed=$((seed+1))
#                          echo "<====scenario2003 continuous truth======> model=$model, n=$n, HR=$HR, pc=$pc"
#                          data_dir=./scenario2003/data_${model}/wo_z_${n}_log${HR}_${pc}
#                          ana_dir=./scenario2003/result_rfbbp_pem_${model}/wo_z_${n}_log${HR}_${pc}
#                          if [[ ! -d "$data_dir" ]]; then
#                               if [[ "$model" == "weibull" ]]; then
#                                    $runtool main.py simulate -od "$data_dir" --seed "$seed" \
#                                    --survival_model weibull --rho 1.5 --n_studies $study_n --n_samples "$n" \
#                                    --a "${a_vals[@]}" --b "${b_vals[@]}" --sigma2_e "$sigma2e_main" \
#                                    --beta_0 0 0 0 --beta_x "$bx" -nr $nrepeat --p_censor "$pc"
#                               elif [[ "$model" == "loglogistic" ]]; then
#                                    $runtool main.py simulate -od "$data_dir" --seed "$seed" \
#                                    --survival_model loglogistic --loglogistic_scale 1.0 --loglogistic_shape 1.5 \
#                                    --n_studies $study_n --n_samples "$n" \
#                                    --a "${a_vals[@]}" --b "${b_vals[@]}" --sigma2_e "$sigma2e_main" \
#                                    --beta_0 0 0 0 --beta_x "$bx" -nr $nrepeat --p_censor "$pc"
#                               elif [[ "$model" == "gompertz" ]]; then
#                                    $runtool main.py simulate -od "$data_dir" --seed "$seed" \
#                                    --survival_model gompertz --gompertz_scale 0.08 --gompertz_shape 0.25 \
#                                    --n_studies $study_n --n_samples "$n" \
#                                    --a "${a_vals[@]}" --b "${b_vals[@]}" --sigma2_e "$sigma2e_main" \
#                                    --beta_0 0 0 0 --beta_x "$bx" -nr $nrepeat --p_censor "$pc"
#                               fi
#                          else
#                               echo "[SKIP SIMULATE] $data_dir already exists"
#                          fi
#                          run_rfbbp_pem_analysis "$data_dir" "$ana_dir"
#                     done
#                done
#           done
#           $runtool main.py summarize -efp "./scenario2003/result_rfbbp_pem_${model}/wo_z_*/eval_results.csv" -of ./scenario2003/result_rfbbp_pem_${model}/summary_2003_${model}.xlsx \
#                -sp n_samples p_censor betax
#      done
# fi

# if [[ "$run_scenario" == "2004" || "$run_scenario" == "all" ]]; then
#      sigma2e_values=(0.0625 0.5 1.0)
#      for sigma2e in "${sigma2e_values[@]}"; do
#           for n in "${n_samples[@]}"; do
#                for pc in "${p_censor_main[@]}"; do
#                     for HR in "${HRs[@]}"; do
#                          echo "<====scenario2004 measurement error======> sigma2e=$sigma2e, n=$n, HR=$HR, pc=$pc"
#                          data_dir=./scenario2004/data_sigma2e_${sigma2e}/wo_z_${n}_log${HR}_${pc}
#                          ana_dir=./scenario2004/result_rfbbp_pem_sigma2e_${sigma2e}/wo_z_${n}_log${HR}_${pc}
#                          sigma2e_main=$sigma2e
#                          simulate_pem_if_needed "$data_dir" "$n" "$HR" "$pc" "${pem_increasing[@]}"
#                          run_rfbbp_pem_analysis "$data_dir" "$ana_dir"
#                          sigma2e_main=0.25
#                     done
#                done
#           done
#           $runtool main.py summarize -efp "./scenario2004/result_rfbbp_pem_sigma2e_${sigma2e}/wo_z_*/eval_results.csv" -of ./scenario2004/result_rfbbp_pem_sigma2e_${sigma2e}/summary_2004_sigma2e_${sigma2e}.xlsx \
#                -sp n_samples p_censor betax
#      done
# fi

# if [[ "$run_scenario" == "2005" || "$run_scenario" == "all" ]]; then
#      stress_n_samples=(50 100)
#      stress_p_censor=(0.7 0.85)
#      for n in "${stress_n_samples[@]}"; do
#           for pc in "${stress_p_censor[@]}"; do
#                for HR in "${HRs[@]}"; do
#                     echo "<====scenario2005 stress test======> n=$n, HR=$HR, pc=$pc"
#                     data_dir=./scenario2005/data/wo_z_${n}_log${HR}_${pc}_pem_inc
#                     ana_dir=./scenario2005/result_rfbbp_pem/wo_z_${n}_log${HR}_${pc}_pem_inc
#                     simulate_pem_if_needed "$data_dir" "$n" "$HR" "$pc" "${pem_increasing[@]}"
#                     run_rfbbp_pem_analysis "$data_dir" "$ana_dir"
#                done
#           done
#      done
#      $runtool main.py summarize -efp "./scenario2005/result_rfbbp_pem/wo_z_*/eval_results.csv" -of ./scenario2005/result_rfbbp_pem/summary_2005.xlsx \
#           -sp n_samples p_censor betax
# fi


#-------------------------- scenario300x final RFBBP simulation suite ------------------------------
# Set RUN_SCENARIO to one of: 3001, 3002, 3003, 3004, 3005, 3006, 300all
run_scenario="${RUN_SCENARIO:-300all}"
nrepeat="${NREPEAT:-1000}"
seed=3000
study_n=4
n_samples_300=(100 300 500)
HRs_300=(1.25 1.5 1.75 2)
HRs_sens_300=(1.5)
p_censor_300=(0.3 0.5)
p_censor_sens_300=0.5
n_sens_300=300
beta0_vals_300=(-0.5 -0.25 0.25 0.5)
a_vals=(-1.5 -0.5 0.5 1.5)
b_vals=(0.5 0.75 1.25 1.5)
sigma2e_main=0.3
pem_cutpoints=(0 1 2 3 4)
pem_increasing=(0.05 0.08 0.12 0.18 0.25)
pem_decreasing=(0.10 0.08 0.06 0.04 0.02)
pem_nonmonotone=(0.05 0.08 0.12 0.08 0.05)
x_lognorm_sigma_lar=0.4
eval_fn=eval_results.csv

run_rfbbp_pem_analysis_300() {
     local data_dir=$1
     local ana_dir=$2
     local standardize_w=${3:-study}
     if [[ ! -d "$ana_dir" ]]; then
          $runtool main.py analyze -dd "$data_dir" -od "$ana_dir" -nc $ncore \
          --target_accept 0.9 --standardize_w "$standardize_w" \
          --betax_sigma0 2.5 --a_sigma0 0.5 --l_sigma0 0.5 \
          --sigma_w_prior invgamma --sigma_w2_alpha0 3.0 --sigma_w2_gamma0 1.0 \
          --auto_cutpoints --study_event_cutpoint_constraint --max_cuts 15 --min_events_per_interval 10
          $runtool main.py evaluate -ad "$ana_dir" -of "$eval_fn"
     else
          echo "[SKIP ANALYZE] $ana_dir already exists"
     fi
}

summarize_if_needed_300() {
     local output_file=$1
     shift
     if [[ ! -f "$output_file" ]]; then
          $runtool main.py summarize "$@" -of "$output_file"
     else
          echo "[SKIP SUMMARIZE] $output_file already exists"
     fi
}

simulate_pem_if_needed_300() {
     local data_dir=$1
     local n=$2
     local HR=$3
     local pc=$4
     local sigma2e=$5
     local x_distribution=$6
     local hazards=("${@:7}")
     local bx
     bx=$(awk -v x=$HR 'BEGIN {print log(x)}')
     seed=$((seed+1))
     if [[ ! -d "$data_dir" ]]; then
          $runtool main.py simulate -od "$data_dir" --seed "$seed" \
          --survival_model pem --n_studies $study_n --n_samples "$n" \
          --beta_0 "${beta0_vals_300[@]}" --a "${a_vals[@]}" --b "${b_vals[@]}" \
          --sigma2_e "$sigma2e" --beta_x "$bx" -nr $nrepeat --p_censor "$pc" \
          --x_distribution "$x_distribution" \
          --x_lognorm_sigma "$x_lognorm_sigma_lar" \
          --pem_cutpoints "${pem_cutpoints[@]}" --pem_hazards "${hazards[@]}"
     else
          echo "[SKIP SIMULATE] $data_dir already exists"
     fi
}

simulate_continuous_if_needed_300() {
     local data_dir=$1
     local model=$2
     local n=$3
     local HR=$4
     local pc=$5
     local bx
     bx=$(awk -v x=$HR 'BEGIN {print log(x)}')
     seed=$((seed+1))
     if [[ ! -d "$data_dir" ]]; then
          if [[ "$model" == "weibull" ]]; then
               $runtool main.py simulate -od "$data_dir" --seed "$seed" \
               --survival_model weibull --rho 1.5 --n_studies $study_n --n_samples "$n" \
               --beta_0 "${beta0_vals_300[@]}" --a "${a_vals[@]}" --b "${b_vals[@]}" \
               --sigma2_e "$sigma2e_main" --beta_x "$bx" -nr $nrepeat --p_censor "$pc"
          elif [[ "$model" == "loglogistic" ]]; then
               $runtool main.py simulate -od "$data_dir" --seed "$seed" \
               --survival_model loglogistic --loglogistic_scale 1.0 --loglogistic_shape 2.0 \
               --n_studies $study_n --n_samples "$n" \
               --beta_0 "${beta0_vals_300[@]}" --a "${a_vals[@]}" --b "${b_vals[@]}" \
               --sigma2_e "$sigma2e_main" --beta_x "$bx" -nr $nrepeat --p_censor "$pc"
          elif [[ "$model" == "gompertz" ]]; then
               $runtool main.py simulate -od "$data_dir" --seed "$seed" \
               --survival_model gompertz --gompertz_scale 0.08 --gompertz_shape 0.15 \
               --n_studies $study_n --n_samples "$n" \
               --beta_0 "${beta0_vals_300[@]}" --a "${a_vals[@]}" --b "${b_vals[@]}" \
               --sigma2_e "$sigma2e_main" --beta_x "$bx" -nr $nrepeat --p_censor "$pc"
          fi
     else
          echo "[SKIP SIMULATE] $data_dir already exists"
     fi
}

if [[ "$run_scenario" == "3001" || "$run_scenario" == "300all" || "$run_scenario" == "all" ]]; then
     for n in "${n_samples_300[@]}"; do
          for pc in "${p_censor_300[@]}"; do
               for HR in "${HRs_300[@]}"; do
                    echo "<====scenario3001 main PEM increasing truth======> n=$n, HR=$HR, pc=$pc, seed=$seed"
                    data_dir=./scenario300x_1000_0.3e/scenario3001/data/wo_z_${n}_log${HR}_${pc}_pem_inc
                    ana_dir=./scenario300x_1000_0.3e/scenario3001/result_rfbbp_pem/wo_z_${n}_log${HR}_${pc}_pem_inc
                    ana_dir_w=./scenario300x_1000_0.3e/scenario3001/result_rfbbp_pem_w/wo_z_${n}_log${HR}_${pc}_pem_inc
                    simulate_pem_if_needed_300 "$data_dir" "$n" "$HR" "$pc" "$sigma2e_main" normal "${pem_increasing[@]}"
                    # run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir" study
                    run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir_w" none
               done
          done
     done
     # summarize_if_needed_300 ./scenario3001/result_rfbbp_pem/summary_3001.xlsx \
     #      -efp "./scenario3001/result_rfbbp_pem/wo_z_*/eval_results.csv" \
     #      -sp n_samples p_censor betax sigma2_e
     summarize_if_needed_300 ./scenario300x_1000_0.3e/scenario3001/result_rfbbp_pem_w/summary_3001_w.xlsx \
          -efp "./scenario300x_1000_0.3e/scenario3001/result_rfbbp_pem_w/wo_z_*/eval_results.csv" \
          -sp n_samples p_censor betax
fi

if [[ "$run_scenario" == "3002" || "$run_scenario" == "300all" || "$run_scenario" == "all" ]]; then
     sigma2e_values_300=(0.1 0.5 0.8 1)
     for sigma2e in "${sigma2e_values_300[@]}"; do
          for HR in "${HRs_sens_300[@]}"; do
               echo "<====scenario3002 measurement error======> sigma2e=$sigma2e, n=$n_sens_300, HR=$HR, pc=$p_censor_sens_300"
               data_dir=./scenario300x_1000_0.3e/scenario3002/data_sigma2e_${sigma2e}/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}_pem_inc
               ana_dir=./scenario300x_1000_0.3e/scenario3002/result_rfbbp_pem_sigma2e_${sigma2e}/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}_pem_inc
               ana_dir_w=./scenario300x_1000_0.3e/scenario3002/result_rfbbp_pem_sigma2e_${sigma2e}_w/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}_pem_inc
               simulate_pem_if_needed_300 "$data_dir" "$n_sens_300" "$HR" "$p_censor_sens_300" "$sigma2e" normal "${pem_increasing[@]}"
               # run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir" study
               run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir_w" none
          done
          # summarize_if_needed_300 ./scenario3002/result_rfbbp_pem_sigma2e_${sigma2e}/summary_3002_sigma2e_${sigma2e}.xlsx \
          #      -efp "./scenario3002/result_rfbbp_pem_sigma2e_${sigma2e}/wo_z_*/eval_results.csv" \
          #      -sp n_samples p_censor betax sigma2_e
          summarize_if_needed_300 ./scenario300x_1000_0.3e/scenario3002/result_rfbbp_pem_sigma2e_${sigma2e}_w/summary_3002_sigma2e_${sigma2e}_w.xlsx \
               -efp "./scenario300x_1000_0.3e/scenario3002/result_rfbbp_pem_sigma2e_${sigma2e}_w/wo_z_*/eval_results.csv" \
               -sp sigma2_e
     done
fi

if [[ "$run_scenario" == "3003" || "$run_scenario" == "300all" || "$run_scenario" == "all" ]]; then
     baseline_names_300=(decreasing nonmonotone)
     baseline_values_300=(
          "0.18 0.15 0.12 0.09 0.07"
          "0.08 0.15 0.22 0.15 0.08"
     )
     for idx in "${!baseline_names_300[@]}"; do
          name=${baseline_names_300[$idx]}
          read -r -a hazards <<< "${baseline_values_300[$idx]}"
          for HR in "${HRs_sens_300[@]}"; do
               echo "<====scenario3003 PEM risk shape======> shape=$name, n=$n_sens_300, HR=$HR, pc=$p_censor_sens_300"
               data_dir=./scenario300x_1000_0.3e/scenario3003/data_${name}/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}
               ana_dir=./scenario300x_1000_0.3e/scenario3003/result_rfbbp_pem_${name}/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}
               ana_dir_w=./scenario300x_1000_0.3e/scenario3003/result_rfbbp_pem_${name}_w/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}
               simulate_pem_if_needed_300 "$data_dir" "$n_sens_300" "$HR" "$p_censor_sens_300" "$sigma2e_main" normal "${hazards[@]}"
               # run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir" study
               run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir_w" none
          done
          # summarize_if_needed_300 ./scenario3003/result_rfbbp_pem_${name}/summary_3003_${name}.xlsx \
          #      -efp "./scenario3003/result_rfbbp_pem_${name}/wo_z_*/eval_results.csv" \
          #      -sp n_samples p_censor betax pem_hazards
          summarize_if_needed_300 ./scenario300x_1000_0.3e/scenario3003/result_rfbbp_pem_${name}_w/summary_3003_${name}_w.xlsx \
               -efp "./scenario300x_1000_0.3e/scenario3003/result_rfbbp_pem_${name}_w/wo_z_*/eval_results.csv" \
               -sp pem_hazards
     done
fi

if [[ "$run_scenario" == "3004" || "$run_scenario" == "300all" || "$run_scenario" == "all" ]]; then
     cont_models_300=(weibull loglogistic gompertz)
     for model in "${cont_models_300[@]}"; do
          for HR in "${HRs_sens_300[@]}"; do
               echo "<====scenario3004 survival distribution======> model=$model, n=$n_sens_300, HR=$HR, pc=$p_censor_sens_300"
               data_dir=./scenario300x_1000_0.3e/scenario3004/data_${model}/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}
               ana_dir=./scenario300x_1000_0.3e/scenario3004/result_rfbbp_pem_${model}/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}
               ana_dir_w=./scenario300x_1000_0.3e/scenario3004/result_rfbbp_pem_${model}_w/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}
               simulate_continuous_if_needed_300 "$data_dir" "$model" "$n_sens_300" "$HR" "$p_censor_sens_300"
               # run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir" study
               run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir_w" none
          done
          # summarize_if_needed_300 ./scenario3004/result_rfbbp_pem_${model}/summary_3004_${model}.xlsx \
          #      -efp "./scenario3004/result_rfbbp_pem_${model}/wo_z_*/eval_results.csv" \
          #      -sp survival_model n_samples p_censor betax
          summarize_if_needed_300 ./scenario300x_1000_0.3e/scenario3004/result_rfbbp_pem_${model}_w/summary_3004_${model}_w.xlsx \
               -efp "./scenario300x_1000_0.3e/scenario3004/result_rfbbp_pem_${model}_w/wo_z_*/eval_results.csv" \
               -sp survival_model
     done
fi

if [[ "$run_scenario" == "3005" || "$run_scenario" == "300all" || "$run_scenario" == "all" ]]; then
     for HR in "${HRs_sens_300[@]}"; do
          echo "<====scenario3005 skew-normal X======> n=$n_sens_300, HR=$HR, pc=$p_censor_sens_300"
          data_dir=./scenario300x_1000_0.3e/scenario3005/data_skewnorm/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}_pem_inc
          ana_dir=./scenario300x_1000_0.3e/scenario3005/result_rfbbp_pem_skewnorm/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}_pem_inc
          ana_dir_w=./scenario300x_1000_0.3e/scenario3005/result_rfbbp_pem_skewnorm_w/wo_z_${n_sens_300}_log${HR}_${p_censor_sens_300}_pem_inc
          simulate_pem_if_needed_300 "$data_dir" "$n_sens_300" "$HR" "$p_censor_sens_300" "$sigma2e_main" skewnorm "${pem_increasing[@]}"
          # run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir" study
          run_rfbbp_pem_analysis_300 "$data_dir" "$ana_dir_w" none
     done
     # summarize_if_needed_300 ./scenario3005/result_rfbbp_pem_skewnorm/summary_3005_skewnorm.xlsx \
     #      -efp "./scenario3005/result_rfbbp_pem_skewnorm/wo_z_*/eval_results.csv" \
     #      -sp x_distribution n_samples p_censor betax
     summarize_if_needed_300 ./scenario300x_1000_0.3e/scenario3005/result_rfbbp_pem_skewnorm_w/summary_3005_skewnorm_w.xlsx \
          -efp "./scenario300x_1000_0.3e/scenario3005/result_rfbbp_pem_skewnorm_w/wo_z_*/eval_results.csv" \
          -sp x_distribution
fi

# scenario3006: between-centre beta heterogeneity. Data are generated with
# beta_s ~ Normal(beta, tau_beta^2), while RFBBP deliberately retains the
# common-beta analysis model. Bias and coverage target the mean beta.
if [[ "$run_scenario" == "3006" || "$run_scenario" == "300all" || "$run_scenario" == "all" ]]; then
     export OMP_NUM_THREADS=1
     export OPENBLAS_NUM_THREADS=1
     export MKL_NUM_THREADS=1
     export NUMEXPR_NUM_THREADS=1

     tau_betas_3006=(${TAU_BETAS:-0.10 0.15})
     methods_3006=(${METHODS:-xtrue_c xtrue_cs xtrue_ps wstandar wstandar_s naive naive_s rfbbp})
     seed_3006="${SEED_3006:-${SEED:-3601}}"
     n_studies_3006=4
     n_per_study_3006=300
     target_hr_3006=1.5
     p_censor_3006=0.5
     beta_3006=$(awk -v x="$target_hr_3006" 'BEGIN {print log(x)}')
     root_3006="${OUTPUT_ROOT:-./scenario300x_1000_0.3e/scenario3006_beta_heterogeneity}"
     summary_3006="${root_3006}/summary_3006_beta_heterogeneity.xlsx"

     for tau_beta_3006 in "${tau_betas_3006[@]}"; do
          # Including nrepeat in the tag prevents a pilot run from being
          # silently reused for a larger final simulation.
          tag_3006="n${n_per_study_3006}_hr${target_hr_3006}_pc${p_censor_3006}_taubeta${tau_beta_3006}_r${nrepeat}"
          data_dir_3006="${root_3006}/data/${tag_3006}"
          ana_dir_3006="${root_3006}/result_rfbbp/${tag_3006}"

          if [[ ! -d "$data_dir_3006" ]]; then
               $runtool main.py simulate \
                    --output_dir "$data_dir_3006" \
                    --seed "$seed_3006" \
                    --survival_model pem \
                    --n_studies "$n_studies_3006" \
                    --n_samples "$n_per_study_3006" \
                    --beta_0 -0.5 -0.25 0.25 0.5 \
                    --a -1.5 -0.5 0.5 1.5 \
                    --b 0.5 0.75 1.25 1.5 \
                    --sigma2_e 0.3 \
                    --beta_x "$beta_3006" \
                    --tau_beta "$tau_beta_3006" \
                    --n_repeats "$nrepeat" \
                    --p_censor "$p_censor_3006" \
                    --x_distribution normal \
                    --pem_cutpoints 0 1 2 3 4 \
                    --pem_hazards 0.05 0.08 0.12 0.18 0.25
          else
               echo "[SKIP SIMULATE] $data_dir_3006 already exists"
          fi

          if [[ ! -d "$ana_dir_3006" ]]; then
               $runtool main.py analyze \
                    --data_dir "$data_dir_3006" \
                    --output_dir "$ana_dir_3006" \
                    --methods "${methods_3006[@]}" \
                    --ncores "$ncore" \
                    --target_accept 0.9 \
                    --standardize_w none \
                    --betax_sigma0 2.5 \
                    --a_sigma0 0.5 \
                    --l_sigma0 0.5 \
                    --sigma_w_prior invgamma \
                    --sigma_w2_alpha0 3.0 \
                    --sigma_w2_gamma0 1.0 \
                    --auto_cutpoints \
                    --study_event_cutpoint_constraint \
                    --max_cuts 15 \
                    --min_events_per_interval 10
               $runtool main.py evaluate \
                    --analyzed_dir "$ana_dir_3006" \
                    --output_file eval_results.csv
          else
               echo "[SKIP ANALYZE] $ana_dir_3006 already exists"
          fi
     done

     $runtool main.py summarize \
          --evaluated_file_pattern "${root_3006}/result_rfbbp/*/eval_results.csv" \
          --output_file "$summary_3006" \
          --summarize_parameters tau_beta betax n_samples p_censor

     echo "Finished scenario3006 beta-heterogeneity comparison (${tau_betas_3006[*]}; methods: ${methods_3006[*]}): $summary_3006"
fi
