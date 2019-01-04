# OR06 HT.55 CT.92 - good results
# OR10 HT.55 CT.95 - also good
# OR12 HT.55 CT.90 - almost ideal
# OR12 TH.55 CT.85 - produces a lot of artifacts, not good
# --------- on renewed expanded road-6 dataset without retraining
# OR08 HT.50 CT.95 - ok
# OR12 HT.53 CT.96 - not very good right u-turn, not good ascending road
# OR06 HT.55 CT.93 - fast and relatively good result. Looks like retrainig on expanded dataset is necessary to achieve
#                    better results. Dataset remarking is also necessary.
# OR20 HT.57 CT.94 - retraining needed.
# Retrained on rod 66 re,apped
# OR06 HT.57 CT.94 - works bad, big gaps
# OR06 HT.60 CT.80 - smaller gaps but sky artigacts
# OR06 HT.70 CT.70 - worse than previuos
# OR06 HT.50 CT.70 - sky artifacts, some gaps but edge sticking is good
# OR06 HT.45 CT.80 - good edge sticking, some artifacts, trying ti incrsease oversamplig to achieve good horizon
# OR16 HT.43 CT.82 - a bit better horizon while very slow processing
# Trying heatmap all sources fast
# OR04 HT.42 CT.84 - not bad but some holes are present when they are not allowed and sky artifacts
# OR20 HT.45 CT.80 - *40 oversampling is very slow but gives    good horizon, some artifacts and gaps
# Trying hirez heatmap with lower CT limit
# OR20 HT.45 CT.70 - .... not very bad but no so cool

if __name__ != '__main__':
    quit()

from convroaddetector import *

_OUT_DIR = 'heatmapping/heatmap_out'
_SRC_DIR = 'heatmapping/heatmap_src'
_COMBINED_DIR = 'heatmapping/combined_out'

#_SRC_DIR = 'img'

make_heatmaps(input_path=_SRC_DIR,
              output_path=_OUT_DIR,
              combined_path=_COMBINED_DIR,
              oversampling_ratio=8,
              heat_threshold=0.35,
              cell_threshold=0.63)

# --- retrained again on expanded dataset to remove air artiacts
# OR04 HT.45 CT.70 - checkup - good edge sticking, better horizon, NO AIR ARTIFACTS!
# OR10 HT.48 CT.75 - big gaps on real road, hming aborted
# OR10 HT.45 CT.60 - no holes in real road, but edge sticking not the best
# OR10 HT.50 CT.60 - gaps in road, better edge sticking
# OR10 HT.55 CT.50 - some gaps, even better EST
# OR10 HT.50 CT.50 - gaps.
# ---- looks like gaps are unbeatable here.
# OR02 HT.47 CT.50 - small gaps on low res, needs tryout on higher res
# OR06 HT.47 CT.50 - gaps still exists
# OR02 HT.40 CT.80 - experimental - a bit better edge sticking, gaps
# OR02 HT.60 CT.20 - experimental - horizon not very good, gaps, +- works
# OR02 HT.60 CT.10 - experimental - much better edge sticking on vertical, still gappy
# OR02 HT.55 CT.10 - bad gaps, bad horizon, cool edge sticking on verticals
# OR02 HT.30 CT.90 - EXP - little gaps, sticking not very good
# OR04 HT.27 CT.90 - EST bad, gaps remians, return to HH-LC strategy
# OR04 HT.80 CT.05 - not bad but needs less HT
# OR06 HT.75 CT.08 - cropping corrected (!) EST not good
# OR06 HT.70 CT.08 - nonono it is wrong way.
# OR06 HT.45 CT.70 - REGAP? - not bad
# Trying hi-res again
# OR20 HT.43 CT.65 - results exists, wait until whole processing. Note: looks like we need some more labeled data with
                    #  mirrored images (!)
# OR20 HT.41 CT.65 - ABIITLOWER - not bad, very slow - waiting for all pics...
        # Very slow but does not worth the efforts...
# OR08 HT.37 CT.61 - nomal results
# OR08 HT.35 CT.63 - used for VAE dataset generation