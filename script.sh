OUT_PREF=scene4

mkdir ${OUT_PREF}
rm ${OUT_PREF}.blend*

../general-purpose/infinigen/blender/blender -b -P single_home.py -- --out_path ${OUT_PREF}.blend --seed 42
../general-purpose/infinigen/blender/blender -b ${OUT_PREF}.blend -o ${OUT_PREF}/frame_ -a -- --cycles-device=CUDA
