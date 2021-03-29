#!/bin/bash

# install gdown for downloading from google-drive link
pip install -U --no-cache-dir gdown --pre

# # install unzip if necessary
# sudo apt install unzip

prev_directory=$(pwd) # current directory

mkdir -p ~/datasets/cdfsl
cd ~/datasets/cdfsl

# ISIC
gdown "https://drive.google.com/uc?id=1FhN8vgg6g0Vm6d-Q3IjeueRiOqU8SUkY"
unzip -qq ISIC.zip
rm -rf __MACOSX
rm ISIC.zip

#miniImageNet
gdown "https://drive.google.com/uc?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY"
mkdir mini-ImageNet
tar -zxf mini-imagenet.tar.gz -C mini-ImageNet/
rm mini-imagenet.tar.gz

# CropDisease
gdown "https://drive.google.com/uc?id=1UlJqQwG5e4PEHnQkBGiD8bqTUNW8t0c0"
unzip CropDiseases.zip
rm CropDiseases.zip
rm -rf __MACOSX

# EuroSAT
gdown "https://drive.google.com/uc?id=1FYZvuBePf2tuEsEaBCsACtIHi6eFpSwe"
unzip eurosat.zip
rm -rf __MACOSX
mv -f eurosat EuroSAT
rm eurosat.zip

# Resisc45
wget -O Resisc45.rar https://public.dm.files.1drv.com/y4mJvJ9x9VJcwndxnvORZAh6MNDMqSJcN9MERoOyY2X5hmwNrREZy_Po5HY0pUxETihpONOpNC6n1nv8W9TEtWrpynkhFLycyl_OjLHRkp-IuhNFn4ScZy9XFBP0yap3LvtbGfTnuJOliamDqAFxl7tgOTrfy2TTQZaiuMUmBxQ1UT_RLAkBv4mIP8gbPDzBCeZKC0oHcusUD4lgxB70Rm0RGdaF_dJLITLRTbkJNTXDoQ?access_token=EwAIA61DBAAUzl/nWKUlBg14ZGcybuC4/OHFdfEAAVBYaMzHtRL9f7DaPwuHm/ERSd3W6vL3sxCerTOtglGLmnSo%2bAfRdtGpN5t5bbAkwWWrGS1zpR%2bVtDwKzz7MzlbKjZAgODMTCekboDfGCNNWz%2bGo9xevzBovdTnLsyMPNLwD1wJn7h0%2b2Gnn%2bq1P/KsdZ7%2b34DEsPnAbwDO2fV1H98pC1WfcDYvG29DpePdnmrhr6OTi3SolWyWsPVgsWMCRbUeOzHjGIG4hTXkPoqv1hwQD/lICUsqyr//ziSJB4F/l35yHsVvFDUd7A7sJyeJ6Lw1bt1kpfQ%2b1Hlq/bnFT3zBI/gNuLxYn1atUH5xpWAro8tUDSviD%2bWn41p1rwqEDZgAACM4SSzSwqZ0/2AGSVKE/mSk91SkFJsUa1gTu34xeKYdPz8%2bkEvKIxwmvfp0Pd1phHStu30bMxmlPaFxqb4hrpCLrhKRDVVKPsU4KvaE5UVp2RRrgHyZHmM5t2MC3jXx4t8Pjf3VMREkk1a2yB1HcCQqD8FXUxAZj1jyJ3gY4C4H9UjYXcbQO3n3voQ3YHjrLbua%2befgOKo7gIxLDulDEfohAjBhlMvxKnnnbzwmf8iFguIh%2bEQz8dCNVvNjn6jmCvld7VAi7oyjzuvOFErDBbWvLXDnMRPAqbaiLmZ27X6GXD7P8fF/L%2b6vgHiIJbnaOl1mhkpn4UcPBdLLTQT%2bqCVw82Qhns4m0wgpn7zCJCO5LVwxVgJGBdUj/X3TwK/2cOS/bCaTgfQRvF6nvPCz5LJ6%2bYpcb8LDQTb8ARhLVOR0ulwGWrIi0oixQqnEUpXDD5XKd6DjgJ/nM3bGcJE34bL87w71sx8TVly8/7gjlsIieT3zlNfcO6M9EmxF%2bEdxv8DAYA6dX848hWslMQX%2bJ/mLv1dh3vf%2bro/JRHPFjvFVwBVObmXDkZxfHtk8JmTFtlSC5FrJYofxtoIVJAaYb1fFUsjeMHp6Rmu6SkJH6vjcAVweGmWEC51aW3jE60kQkv2XfFAI%3d
mkdir Resisc45
unrar e Resisc45.rar Resisc45 -idq
rm Resisc45.rar

# 102 flowers
wget https://www.robots.ox.ac.uk/\~vgg/data/flowers/102/102flowers.tgz
tar -zxf 102flowers.tgz
mkdir 102flowers
mv -f jpg 102flowers/
wget https://www.robots.ox.ac.uk/\~vgg/data/flowers/102/imagelabels.mat
mv imagelabels.mat 102flowers/
rm 102flowers.tgz

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -O 102flowers/

# DTD
wget https://www.robots.ox.ac.uk/\~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -zxf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

# Kaokore dataset (https://github.com/rois-codh/kaokore)
mkdir -p kaokore
python ${prev_directory}/data_loader/kaokore/download.py --dataset_version 1.1 --dir ${HOME}/datasets/cdfsl/kaokore

# DeepWeeds (https://github.com/AlexOlsen/DeepWeeds)
wget https://nextcloud.qriscloud.org.au/index.php/s/a3KxPawpqkiorST/download
mkdir -p DeepWeeds/images
unzip -qq download -d DeepWeeds/images
rm -rf download
wget https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv -O DeepWeeds/labels.csv

# remove if there are other files
rm *.gz
rm *.zip
rm *.tgz

# remove all hidden files
find -type f -name '.*' -exec rm {} \;

cd ${prev_directory}
