# Inpletation of DRAEM on AeBAD
The impletation of DRAEM<https://github.com/VitjanZ/DRAEM> on AeBAD<https://github.com/zhangzilongc/MMR> dataset
We add program in `data_loader.py` to load data from AeBAD, while we modify the test program to prevent from out of cuda memory.
Moreover, the parameters, FLOPS and inference time could be counted in modified `test_DRAEM_aebad.py`.

---
## Quick start
Enviroment setting can be seen in [README](official_README.md). We use `python==3.8, cuda=11.8, pytorch=2.0.0`.
Train on AeBAD_S after changing the dataset path.
```
sh train_DRAEM.sh
sh test_DRAEM.sh
```
