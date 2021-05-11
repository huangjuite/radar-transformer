#! /bin/bash

cwd=$PWD

echo -e "\e[93m rl pretrained model forwar s1536_f1869509 \e[0m"
gdown --id 1E4SBpsKXjwX3q8U4gR2cBOXZ_NUIbuf0


echo -e "\e[93m radar transformer cgn \e[0m"
gdown --id 1DHpi4r74FgMIoWZkQqNZMn4E4EOQgxoA


# rl vae
echo -e "\e[93m download VAE model 0726_1557.pth \e[0m"
cd $cwd/vae
gdown --id 1B2ugYD11vKhcSiJR3loldEmOtpV3OnbQ

# rl cgan
echo -e "\e[93m download cGAN model 0827_1851.pth \e[0m"
cd $cwd/cgan
gdown --id 1KSgA1O-BKuRzav8Ew-bALgKJY490qZKO

# rl contrastive
echo -e "\e[93m download contrastive mse model mse.pth \e[0m"
cd $cwd/rdpg_torch_forward
gdown --id 1gffTWAI3hOGE09FjF0kc_of5xLprdtK1

echo -e "\e[93m radar transformer radar encoder \e[0m"
cd $cwd/transformer
gdown --id 1ggZ3JdnlXc8atXW77y-rQwMolOge4GjX
