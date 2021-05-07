#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-09-05)

nj=30

stage=0
endstage=1
volume=true
basefeat=false
vad=true

topdir=data
datasets="test_all"
prefix=mfcc_20_5.0

feat_type=mfcc
feat_conf=conf/sre-mfcc-20.conf
vad_conf=conf/vad-5.0.conf

pitch=true
suffix=sp

. subtools/parse_options.sh
. subtools/path.sh

#suffix=sp
[ $volume == "true" ] && suffix=volume_$suffix

for data in $datasets ;do
srcdir=$topdir/$prefix/$data

if [[ $stage -le 0 && 0 -le $endstage ]];then
echo "[stage 0] Speed 3way"
subtools/kaldi/utils/data/perturb_data_dir_speed_3way.sh $srcdir ${srcdir}_$suffix
[ $volume == "true" ] && subtools/kaldi/utils/data/perturb_data_dir_volume.sh ${srcdir}_$suffix
subtools/correctSpeakerAfterSp3way.sh ${srcdir}_$suffix
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
echo "[stage 1] Make features"
[ $basefeat == "true" ] && subtools/makeFeatures.sh --nj $nj --pitch $pitch ${srcdir} $feat_type $feat_conf \
&& [ $vad == "true" ] && subtools/computeVad.sh --nj $nj ${srcdir} $vad_conf
subtools/makeFeatures.sh --nj $nj --pitch $pitch ${srcdir}_$suffix $feat_type $feat_conf
[ $vad == "true" ] && subtools/computeVad.sh --nj $nj ${srcdir}_$suffix $vad_conf
fi


done
