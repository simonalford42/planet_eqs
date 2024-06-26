#!/usr/bin/env bash


sbatch -J p1_800 --partition gpu run.sh --Ngrid 800 --ix 1 --total 40 --std --compute
sbatch -J p21_800 --partition gpu run.sh --Ngrid 800 --ix 21 --total 40 --std --compute
sbatch -J p8_800 --partition gpu run.sh --Ngrid 800 --ix 8 --total 40 --std --compute
sbatch -J p16_800 --partition gpu run.sh --Ngrid 800 --ix 16 --total 40 --std --compute

        #    4075672       gpu   p0_800    sca63  R      21:49      1 nikola-compute-02
        #    4075724       gpu   p1_800    sca63  R       0:01      1 nikola-compute-03
        #    4075674       gpu   p2_800    sca63  R      21:49      1 nikola-compute-03
        #    4075675       gpu   p3_800    sca63  R      21:49      1 nikola-compute-04
        #    4075676       gpu   p4_800    sca63  R      21:49      1 nikola-compute-04
        #    4075677       gpu   p5_800    sca63  R      21:49      1 joachims-compute-02
        #    4075678       gpu   p6_800    sca63  R      21:49      1 joachims-compute-02
        #    4075679       gpu   p7_800    sca63  R      21:49      1 sablab-gpu-01
        #    4075726       gpu   p8_800    sca63  R       0:01      1 sablab-gpu-05
        #    4075681       gpu   p9_800    sca63  R      21:49      1 sablab-gpu-01
        #    4075682       gpu  p10_800    sca63  R      21:49      1 sablab-gpu-01
        #    4075683       gpu  p11_800    sca63  R      21:49      1 hariharan-compute-02
        #    4075684       gpu  p12_800    sca63  R      21:49      1 hariharan-compute-02
        #    4075685       gpu  p13_800    sca63  R      21:49      1 hariharan-compute-02
        #    4075686       gpu  p14_800    sca63  R      21:49      1 hariharan-compute-02
        #    4075687       gpu  p15_800    sca63  R      21:49      1 sablab-gpu-05
        #    4075727       gpu  p16_800    sca63  R       0:01      1 sablab-gpu-03
        #    4075689       gpu  p17_800    sca63  R      21:49      1 sablab-gpu-07
        #    4075690       gpu  p18_800    sca63  R      21:49      1 kim-compute-02
        #    4075691       gpu  p19_800    sca63  R      21:49      1 sablab-gpu-02
        #    4075692       gpu  p20_800    sca63  R      21:49      1 sablab-gpu-02
        #    4075725       gpu  p21_800    sca63  R       0:01      1 sablab-gpu-01
        #    4075694       gpu  p22_800    sca63  R      21:49      1 sablab-gpu-03
        #    4075695       gpu  p23_800    sca63  R      21:49      1 hinton
        #    4075696       gpu  p24_800    sca63  R      21:49      1 hinton
        #    4075697       gpu  p25_800    sca63  R      21:49      1 tripods-compute-01
        #    4075698       gpu  p26_800    sca63  R      21:49      1 tripods-compute-01
        #    4075699       gpu  p27_800    sca63  R      21:49      1 tripods-compute-01
        #    4075700       gpu  p28_800    sca63  R      21:49      1 tripods-compute-01
        #    4075701       gpu  p29_800    sca63  R      21:49      1 coecis-compute-02
        #    4075702       gpu  p30_800    sca63  R      21:49      1 coecis-compute-02
        #    4075703       gpu  p31_800    sca63  R      21:49      1 harpo
        #    4075704       gpu  p32_800    sca63  R      21:49      1 harpo
        #    4075705       gpu  p33_800    sca63  R      21:49      1 harpo
        #    4075706       gpu  p34_800    sca63  R      21:49      1 harpo
        #    4075707       gpu  p35_800    sca63  R      21:49      1 harpo
        #    4075708       gpu  p36_800    sca63  R      21:49      1 tripods-compute-02
        #    4075709       gpu  p37_800    sca63  R      21:49      1 tripods-compute-02
        #    4075710       gpu  p38_800    sca63  R      21:49      1 tripods-compute-02
        #    4075711       gpu  p39_800    sca63  R      21:49      1 tripods-compute-02

# ---------------------- Mon June 24 ----------------------
# sbatch -J p0_400 --partition gpu run.sh --Ngrid 400 --ix 0 --total 40
# sbatch -J p1_400 --partition gpu run.sh --Ngrid 400 --ix 1 --total 40
# sbatch -J p2_400 --partition gpu run.sh --Ngrid 400 --ix 2 --total 40
# sbatch -J p3_400 --partition gpu run.sh --Ngrid 400 --ix 3 --total 40
# sbatch -J p4_400 --partition gpu run.sh --Ngrid 400 --ix 4 --total 40
# sbatch -J p5_400 --partition gpu run.sh --Ngrid 400 --ix 5 --total 40
# sbatch -J p6_400 --partition gpu run.sh --Ngrid 400 --ix 6 --total 40
# sbatch -J p7_400 --partition gpu run.sh --Ngrid 400 --ix 7 --total 40
# sbatch -J p8_400 --partition gpu run.sh --Ngrid 400 --ix 8 --total 40
# sbatch -J p9_400 --partition gpu run.sh --Ngrid 400 --ix 9 --total 40

# sbatch -J p10_400 --partition gpu run.sh --Ngrid 400 --ix 10 --total 40
# sbatch -J p11_400 --partition gpu run.sh --Ngrid 400 --ix 11 --total 40
# sbatch -J p12_400 --partition gpu run.sh --Ngrid 400 --ix 12 --total 40
# sbatch -J p13_400 --partition gpu run.sh --Ngrid 400 --ix 13 --total 40
# sbatch -J p14_400 --partition gpu run.sh --Ngrid 400 --ix 14 --total 40
# sbatch -J p15_400 --partition gpu run.sh --Ngrid 400 --ix 15 --total 40
# sbatch -J p16_400 --partition gpu run.sh --Ngrid 400 --ix 16 --total 40
# sbatch -J p17_400 --partition gpu run.sh --Ngrid 400 --ix 17 --total 40
# sbatch -J p18_400 --partition gpu run.sh --Ngrid 400 --ix 18 --total 40
# sbatch -J p19_400 --partition gpu run.sh --Ngrid 400 --ix 19 --total 40

# sbatch -J p20_400 --partition gpu run.sh --Ngrid 400 --ix 20 --total 40
# sbatch -J p21_400 --partition gpu run.sh --Ngrid 400 --ix 21 --total 40
# sbatch -J p22_400 --partition gpu run.sh --Ngrid 400 --ix 22 --total 40
# sbatch -J p23_400 --partition gpu run.sh --Ngrid 400 --ix 23 --total 40
# sbatch -J p24_400 --partition gpu run.sh --Ngrid 400 --ix 24 --total 40
# sbatch -J p25_400 --partition gpu run.sh --Ngrid 400 --ix 25 --total 40
# sbatch -J p26_400 --partition gpu run.sh --Ngrid 400 --ix 26 --total 40
# sbatch -J p27_400 --partition gpu run.sh --Ngrid 400 --ix 27 --total 40
# sbatch -J p28_400 --partition gpu run.sh --Ngrid 400 --ix 28 --total 40
# sbatch -J p29_400 --partition gpu run.sh --Ngrid 400 --ix 29 --total 40

# sbatch -J p30_400 --partition gpu run.sh --Ngrid 400 --ix 30 --total 40
# sbatch -J p31_400 --partition gpu run.sh --Ngrid 400 --ix 31 --total 40
# sbatch -J p32_400 --partition gpu run.sh --Ngrid 400 --ix 32 --total 40
# sbatch -J p33_400 --partition gpu run.sh --Ngrid 400 --ix 33 --total 40
# sbatch -J p34_400 --partition gpu run.sh --Ngrid 400 --ix 34 --total 40
# sbatch -J p35_400 --partition gpu run.sh --Ngrid 400 --ix 35 --total 40
# sbatch -J p36_400 --partition gpu run.sh --Ngrid 400 --ix 36 --total 40
# sbatch -J p37_400 --partition gpu run.sh --Ngrid 400 --ix 37 --total 40
# sbatch -J p38_400 --partition gpu run.sh --Ngrid 400 --ix 38 --total 40
# sbatch -J p39_400 --partition gpu run.sh --Ngrid 400 --ix 39 --total 40

# ---------------------- Sun June 23 ----------------------
# sbatch -J p23_400 --partition gpu run.sh --Ngrid 400 --ix 23 --total 40
# sbatch -J p24_400 --partition gpu run.sh --Ngrid 400 --ix 24 --total 40
# sbatch -J p31_400 --partition gpu run.sh --Ngrid 400 --ix 31 --total 40
# sbatch -J p32_400 --partition gpu run.sh --Ngrid 400 --ix 32 --total 40
# sbatch -J p34_400 --partition gpu run.sh --Ngrid 400 --ix 34 --total 40
# sbatch -J p37_400 --partition gpu run.sh --Ngrid 400 --ix 37 --total 40
# sbatch -J p38_400 --partition gpu run.sh --Ngrid 400 --ix 38 --total 40
# sbatch -J p39_400 --partition gpu run.sh --Ngrid 400 --ix 39 --total 40

# ---------------------- Thu June 20 ----------------------
# sbatch -J p0_200 --partition gpu run.sh --Ngrid 200 --ix 0 --total 10
# sbatch -J p1_200 --partition gpu run.sh --Ngrid 200 --ix 1 --total 10
# sbatch -J p2_200 --partition gpu run.sh --Ngrid 200 --ix 2 --total 10
# sbatch -J p3_200 --partition gpu run.sh --Ngrid 200 --ix 3 --total 10
# sbatch -J p4_200 --partition gpu run.sh --Ngrid 200 --ix 4 --total 10
# sbatch -J p5_200 --partition gpu run.sh --Ngrid 200 --ix 5 --total 10
# sbatch -J p6_200 --partition gpu run.sh --Ngrid 200 --ix 6 --total 10
# sbatch -J p7_200 --partition gpu run.sh --Ngrid 200 --ix 7 --total 10
# sbatch -J p8_200 --partition gpu run.sh --Ngrid 200 --ix 8 --total 10
# sbatch -J p9_200 --partition gpu run.sh --Ngrid 200 --ix 9 --total 10

# sbatch -J p0_400 --partition gpu run.sh --Ngrid 400 --ix 0 --total 40
# sbatch -J p1_400 --partition gpu run.sh --Ngrid 400 --ix 1 --total 40
# sbatch -J p2_400 --partition gpu run.sh --Ngrid 400 --ix 2 --total 40
# sbatch -J p3_400 --partition gpu run.sh --Ngrid 400 --ix 3 --total 40
# sbatch -J p4_400 --partition gpu run.sh --Ngrid 400 --ix 4 --total 40
# sbatch -J p5_400 --partition gpu run.sh --Ngrid 400 --ix 5 --total 40
# sbatch -J p6_400 --partition gpu run.sh --Ngrid 400 --ix 6 --total 40
# sbatch -J p7_400 --partition gpu run.sh --Ngrid 400 --ix 7 --total 40
# sbatch -J p8_400 --partition gpu run.sh --Ngrid 400 --ix 8 --total 40
# sbatch -J p9_400 --partition gpu run.sh --Ngrid 400 --ix 9 --total 40

# sbatch -J p10_400 --partition gpu run.sh --Ngrid 400 --ix 10 --total 40
# sbatch -J p11_400 --partition gpu run.sh --Ngrid 400 --ix 11 --total 40
# sbatch -J p12_400 --partition gpu run.sh --Ngrid 400 --ix 12 --total 40
# sbatch -J p13_400 --partition gpu run.sh --Ngrid 400 --ix 13 --total 40
# sbatch -J p14_400 --partition gpu run.sh --Ngrid 400 --ix 14 --total 40
# sbatch -J p15_400 --partition gpu run.sh --Ngrid 400 --ix 15 --total 40
# sbatch -J p16_400 --partition gpu run.sh --Ngrid 400 --ix 16 --total 40
# sbatch -J p17_400 --partition gpu run.sh --Ngrid 400 --ix 17 --total 40
# sbatch -J p18_400 --partition gpu run.sh --Ngrid 400 --ix 18 --total 40
# sbatch -J p19_400 --partition gpu run.sh --Ngrid 400 --ix 19 --total 40

# sbatch -J p20_400 --partition gpu run.sh --Ngrid 400 --ix 20 --total 40
# sbatch -J p21_400 --partition gpu run.sh --Ngrid 400 --ix 21 --total 40
# sbatch -J p22_400 --partition gpu run.sh --Ngrid 400 --ix 22 --total 40
# sbatch -J p23_400 --partition gpu run.sh --Ngrid 400 --ix 23 --total 40
# sbatch -J p24_400 --partition gpu run.sh --Ngrid 400 --ix 24 --total 40
# sbatch -J p25_400 --partition gpu run.sh --Ngrid 400 --ix 25 --total 40
# sbatch -J p26_400 --partition gpu run.sh --Ngrid 400 --ix 26 --total 40
# sbatch -J p27_400 --partition gpu run.sh --Ngrid 400 --ix 27 --total 40
# sbatch -J p28_400 --partition gpu run.sh --Ngrid 400 --ix 28 --total 40
# sbatch -J p29_400 --partition gpu run.sh --Ngrid 400 --ix 29 --total 40

# sbatch -J p30_400 --partition gpu run.sh --Ngrid 400 --ix 30 --total 40
# sbatch -J p31_400 --partition gpu run.sh --Ngrid 400 --ix 31 --total 40
# sbatch -J p32_400 --partition gpu run.sh --Ngrid 400 --ix 32 --total 40
# sbatch -J p33_400 --partition gpu run.sh --Ngrid 400 --ix 33 --total 40
# sbatch -J p34_400 --partition gpu run.sh --Ngrid 400 --ix 34 --total 40
# sbatch -J p35_400 --partition gpu run.sh --Ngrid 400 --ix 35 --total 40
# sbatch -J p36_400 --partition gpu run.sh --Ngrid 400 --ix 36 --total 40
# sbatch -J p37_400 --partition gpu run.sh --Ngrid 400 --ix 37 --total 40
# sbatch -J p38_400 --partition gpu run.sh --Ngrid 400 --ix 38 --total 40
# sbatch -J p39_400 --partition gpu run.sh --Ngrid 400 --ix 39 --total 40

# sbatch -J p0_4 --partition gpu run.sh --Ngrid 4 --ix 0 --total 4
# sbatch -J p1_4 --partition gpu run.sh --Ngrid 4 --ix 1 --total 4
# sbatch -J p2_4 --partition gpu run.sh --Ngrid 4 --ix 2 --total 4
# sbatch -J p3_4 --partition gpu run.sh --Ngrid 4 --ix 3 --total 4

# ---------------------- Wed June 19 ----------------------
# sbatch -J megno4 --partition gpu run.sh 0 4
# sbatch -J megno20 --partition gpu run.sh 0 20
# sbatch -J megno80 --partition gpu run.sh 0 80
# sbatch -J model4 --partition gpu run.sh 1 4
# sbatch -J model20 --partition gpu run.sh 1 20
# sbatch -J model80 --partition gpu run.sh 1 80


