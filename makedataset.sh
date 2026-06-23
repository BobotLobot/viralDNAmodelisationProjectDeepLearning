rm -r ~/smallDataCopy/*
mkdir ~/smallDataCopy/noisy ~/smallDataCopy/noNoise ~/smallDataCopy/onlyNoiseNoStruct
ls ~/bigDataCopy/noisy/ | head -n $1 | xargs -I {} cp ~/bigDataCopy/noisy/{} ~/smallDataCopy/noisy/
ls ~/bigDataCopy/noNoise/ | head -n $1 | xargs -I {} cp ~/bigDataCopy/noNoise/{} ~/smallDataCopy/noNoise/
ls ~/bigDataCopy/onlyNoiseNoStruct | head -n $1 | xargs -I {} cp ~/bigDataCopy/onlyNoiseNoStruct/{} ~/smallDataCopy/onlyNoiseNoStruct/
