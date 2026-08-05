#!/bin/bash
#SBATCH --job-name=download_swc
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=download.log

gsutil -m cp -r gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/ /cephfs2/yyin/mcns_ad_split/